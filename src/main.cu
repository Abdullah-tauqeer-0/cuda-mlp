// src/main.cu
// Minimal demo: train a 2-8-1 MLP on XOR with MSE loss
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "mlp.hpp"
#include "utils.hpp"

// --------------------------------------------------------------------- //
//                      device-side loss + gradient                      //
// --------------------------------------------------------------------- //
__global__ void k_mse_loss_grad(const fp32* y_pred,  // [batch,1]
                                const fp32* y_true,  // [batch,1]
                                fp32*       d_loss,  // [batch,1]  grad wrt y_pred
                                fp32*       loss_out,// scalar, accumulated
                                size_t batch)
{
    __shared__ fp32 sh_sum;
    if (threadIdx.x == 0) sh_sum = 0.f;
    __syncthreads();

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        fp32 diff      = y_pred[i] - y_true[i];
        d_loss[i]      = 2.f * diff / batch;   // d(MSE)/dy_pred
        atomicAdd(&sh_sum, diff * diff);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0)
        atomicAdd(loss_out, sh_sum / batch);
}

// --------------------------------------------------------------------- //
int main()
{
    // -------- 1. Tiny XOR dataset (4 samples, batch training) --------- //
    const std::vector<fp32> h_X = {
        0.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        1.f, 1.f
    };
    const std::vector<fp32> h_y = { 0.f, 1.f, 1.f, 0.f };

    const size_t batch = 4;
    const size_t in_dim = 2, out_dim = 1;

    fp32 *d_X, *d_y;
    CUDA_CHECK(cudaMalloc(&d_X, h_X.size()*sizeof(fp32)));
    CUDA_CHECK(cudaMalloc(&d_y, h_y.size()*sizeof(fp32)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), h_X.size()*sizeof(fp32),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), h_y.size()*sizeof(fp32),
                          cudaMemcpyHostToDevice));

    // -------- 2. Build MLP: 2-8-1 with ReLU then Sigmoid -------------- //
    std::vector<size_t> dims = {in_dim, 8, out_dim};
    std::vector<ActType> acts = {ActType::RELU, ActType::SIGMOID};
    MLP net(dims, acts, /*lr=*/1e-1f);

    // -------- 3. Buffers for loss / grad ------------------------------ //
    fp32 *d_pred, *d_grad, *d_loss_scalar;
    CUDA_CHECK(cudaMalloc(&d_grad, batch*out_dim*sizeof(fp32)));
    CUDA_CHECK(cudaMalloc(&d_loss_scalar, sizeof(fp32)));

    // -------- 4. Training loop --------------------------------------- //
    const int epochs = 3000;
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // forward
        d_pred = const_cast<fp32*>(net.forward(d_X, batch));

        // zero loss accumulator
        CUDA_CHECK(cudaMemset(d_loss_scalar, 0, sizeof(fp32)));

        // loss + grad
        constexpr int TPB = 128;
        int blocks = (batch + TPB - 1) / TPB;
        k_mse_loss_grad<<<blocks, TPB>>>(
            d_pred, d_y, d_grad, d_loss_scalar, batch);

        // backward
        net.backward(d_y, d_grad, batch);

        // update
        net.step();

        // host loss print every 200 iters
        if (epoch % 200 == 0) {
            fp32 h_loss;
            CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_scalar,
                                  sizeof(fp32), cudaMemcpyDeviceToHost));
            printf("Epoch %4d | MSE: %.6f\n", epoch, h_loss);
        }
    }

    // -------- 5. Cleanup --------------------------------------------- //
    cudaFree(d_X); cudaFree(d_y);
    cudaFree(d_grad); cudaFree(d_loss_scalar);
    printf("Training complete.\n");
    return 0;
}
