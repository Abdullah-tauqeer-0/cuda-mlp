// src/mlp.cu
// CUDA-driven MLP implementation
#include <cassert>
#include "mlp.hpp"
#include "utils.hpp"

// --------------------------------------------------------------------- //
//                 ───---  Local helper kernels  ---───                  //
// --------------------------------------------------------------------- //

// --- grad_input = grad_Z · W  ---------------------------------------- //
__global__ void k_linear_grad_input(const fp32* d_gradZ,  // [batch,out]
                                    const fp32* d_W,      // [out,in]
                                    fp32*       d_gradX,  // [batch,in]
                                    size_t batch,
                                    size_t in_dim,
                                    size_t out_dim)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // sample
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // in_dim
    if (row >= batch || col >= in_dim) return;

    fp32 acc = 0.f;
    for (size_t j = 0; j < out_dim; ++j)
        acc += d_gradZ[row * out_dim + j] * d_W[j * in_dim + col];
    d_gradX[row * in_dim + col] = acc;
}

// --- accumulate dW & db  --------------------------------------------- //
__global__ void k_linear_grad_Wb(const fp32* d_X,       // [batch,in]
                                 const fp32* d_gradZ,   // [batch,out]
                                 fp32*       d_gradW,   // [out,in]
                                 fp32*       d_gradb,   // [out]
                                 size_t batch,
                                 size_t in_dim,
                                 size_t out_dim)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // out_dim idx
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // in_dim  idx
    if (row >= out_dim || col >= in_dim) return;

    fp32 accW = 0.f;
    for (size_t n = 0; n < batch; ++n)
        accW += d_gradZ[n * out_dim + row] * d_X[n * in_dim + col];

    d_gradW[row * in_dim + col] = accW;

    // One thread per row accumulates bias
    if (col == 0) {
        fp32 accb = 0.f;
        for (size_t n = 0; n < batch; ++n)
            accb += d_gradZ[n * out_dim + row];
        d_gradb[row] = accb;
    }
}

// --- SGD update & zero-grad ------------------------------------------ //
__global__ void k_sgd(fp32* param, fp32* grad,
                      fp32 lr, fp32 scale, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    param[idx] -= lr * scale * grad[idx];
    grad[idx]   = 0.f;
}

// --------------------------------------------------------------------- //
//                     ───---  MLP methods  ---───                       //
// --------------------------------------------------------------------- //
MLP::MLP(const std::vector<size_t>& dims,
         const std::vector<ActType>& acts,
         fp32 lr)
    : lr_(lr)
{
    assert(dims.size() >= 2 && acts.size() == dims.size() - 1);
    for (size_t i = 0; i + 1 < dims.size(); ++i)
        layers_.emplace_back(dims[i], dims[i + 1], acts[i]);
}

MLP::~MLP() = default;

// ---- forward --------------------------------------------------------- //
const fp32* MLP::forward(const fp32* d_x, size_t batch)
{
    const fp32* cur = d_x;

    for (auto& L : layers_) {
        size_t z_bytes = batch * L.out_dim * sizeof(fp32);

        // allocate (or reuse) caches
        if (!L.d_z)   CUDA_CHECK(cudaMalloc(&L.d_z  , z_bytes));
        if (!L.d_out) CUDA_CHECK(cudaMalloc(&L.d_out, z_bytes));

        linear_forward(cur, L.d_W, L.d_b,
                       L.d_z, batch, L.in_dim, L.out_dim);

        act_forward(L.d_z, L.d_out,
                    batch * L.out_dim, L.act);

        cur = L.d_out;           // output of this layer = input to next
    }
    return cur;                 // pointer to logits / final activations
}

// ---- backward -------------------------------------------------------- //
void MLP::backward(const fp32* d_y_true,
                   const fp32* d_grad_out,
                   size_t batch)
{
    // Start with gradient wrt network output
    const fp32* cur_grad = d_grad_out;

    // Work backward through layers
    for (int li = static_cast<int>(layers_.size()) - 1; li >= 0; --li) {
        auto& L = layers_[li];

        // grad after activation → before activation
        fp32* d_gradZ;
        CUDA_CHECK(cudaMalloc(&d_gradZ,
                              batch * L.out_dim * sizeof(fp32)));
        act_backward(L.d_out, cur_grad, d_gradZ,
                     batch * L.out_dim, L.act);

        // grads w.r.t weights, bias, and input
        fp32* d_gradX;
        CUDA_CHECK(cudaMalloc(&d_gradX,
                              batch * L.in_dim * sizeof(fp32)));

        dim3 th(16,16);
        dim3 grW((L.in_dim + th.x - 1)/th.x,
                 (L.out_dim+ th.y - 1)/th.y);
        k_linear_grad_Wb<<<grW, th>>>(
            li == 0 ? /*input layer*/ nullptr : layers_[li-1].d_out,
            d_gradZ,
            L.d_dW, L.d_db,
            batch, L.in_dim, L.out_dim);

        dim3 grX((L.in_dim + th.x - 1)/th.x,
                 (batch    + th.y - 1)/th.y);
        k_linear_grad_input<<<grX, th>>>(
            d_gradZ, L.d_W, d_gradX,
            batch, L.in_dim, L.out_dim);

        cudaFree(d_gradZ);                    // no longer needed
        if (li != static_cast<int>(layers_.size()) - 1)
            cudaFree((void*)cur_grad);        // free prev layer grad
        cur_grad = d_gradX;                   // pass to next iteration
    }
    cudaFree((void*)cur_grad);                // gradient of input not used
}

// ---- SGD step --------------------------------------------------------- //
void MLP::step(fp32 lr_override)
{
    fp32 lr = (lr_override > 0.f) ? lr_override : lr_;
    constexpr int TPB = 256;

    for (auto& L : layers_) {
        size_t w_elems = L.in_dim * L.out_dim;
        size_t b_elems = L.out_dim;
        int   blocksW  = static_cast<int>((w_elems + TPB - 1)/TPB);
        int   blocksB  = static_cast<int>((b_elems + TPB - 1)/TPB);

        // scale=1.0 (could divide by batch here if desired)
        k_sgd<<<blocksW, TPB>>>(L.d_W , L.d_dW, lr, 1.f, w_elems);
        k_sgd<<<blocksB, TPB>>>(L.d_b , L.d_db, lr, 1.f, b_elems);
    }
}
