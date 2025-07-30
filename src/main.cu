// src/main.cu  (full file replaces old version)
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "mlp.hpp"
#include "utils.hpp"
#include "csv.hpp"

// ---------- Device-side MSE loss & gradient (same as before) ---------- //
__global__ void k_mse_loss_grad(const fp32* y_pred, const fp32* y_true,
                                fp32* d_loss, fp32* loss_out, size_t batch);
// … (kernel body identical to previous prompt) …

// ---------------------- Helper: upload host vectors ------------------- //
static void upload(const std::vector<fp32>& h,
                   fp32*& d, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    CUDA_CHECK(cudaMalloc(&d, h.size()*sizeof(fp32)));
    CUDA_CHECK(cudaMemcpy(d, h.data(),
                          h.size()*sizeof(fp32), kind));
}

// --------------------------------------------------------------------- //
int main(int argc, char** argv)
{
    // ---------------------------------------------------------------- //
    // 1.  Parse flags: csv path, epochs, lr (all optional)
    // ---------------------------------------------------------------- //
    std::string csv_path;
    int   epochs = 3000;
    fp32  lr     = 1e-1f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv" && i+1 < argc) csv_path = argv[++i];
        else if (arg == "--epochs" && i+1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i+1 < argc)     lr     = std::stof(argv[++i]);
        else {
            printf("Usage: %s [--csv path] [--epochs N] [--lr val]\n", argv[0]);
            return 0;
        }
    }

    // ---------------------------------------------------------------- //
    // 2.  Load data
    // ---------------------------------------------------------------- //
    std::vector<fp32> h_X, h_y;
    size_t in_dim = 2;                    // default XOR dims
    if (!csv_path.empty()) {
        load_csv(csv_path, h_X, h_y, in_dim);
    } else {
        // XOR fallback
        h_X = {0,0,  0,1,  1,0,  1,1};
        h_y = {0,1,1,0};
    }
    const size_t batch = h_y.size();
    const size_t out_dim = 1;             // binary regression

    fp32 *d_X, *d_y;
    upload(h_X, d_X);
    upload(h_y, d_y);

    // ---------------------------------------------------------------- //
    // 3.  Build network
    // ---------------------------------------------------------------- //
    std::vector<size_t> dims = {in_dim, 8, out_dim};
    std::vector<ActType> acts = {ActType::RELU, ActType::SIGMOID};
    MLP net(dims, acts, lr);

    // ---------------------------------------------------------------- //
    // 4.  Buffers
    // ---------------------------------------------------------------- //
    fp32 *d_grad, *d_loss_scalar;
    CUDA_CHECK(cudaMalloc(&d_grad,  batch*out_dim*sizeof(fp32)));
    CUDA_CHECK(cudaMalloc(&d_loss_scalar, sizeof(fp32)));

    // ---------------------------------------------------------------- //
    // 5.  Training loop
    // ---------------------------------------------------------------- //
    constexpr int TPB = 128;
    int blocks = (batch + TPB - 1) / TPB;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const fp32* d_pred = net.forward(d_X, batch);

        // zero loss
        CUDA_CHECK(cudaMemset(d_loss_scalar, 0, sizeof(fp32)));

        k_mse_loss_grad<<<blocks, TPB>>>(d_pred, d_y,
                                         d_grad, d_loss_scalar, batch);

        net.backward(d_y, d_grad, batch);
        net.step();

        if (epoch % 200 == 0) {
            fp32 h_loss;
            CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_scalar,
                                  sizeof(fp32), cudaMemcpyDeviceToHost));
            printf("Epoch %5d | MSE %.6f\n", epoch, h_loss);
        }
    }

    // ---------------------------------------------------------------- //
    // 6.  Cleanup
    // ---------------------------------------------------------------- //
    cudaFree(d_X); cudaFree(d_y);
    cudaFree(d_grad); cudaFree(d_loss_scalar);
    printf("Done.\n");
    return 0;
}
