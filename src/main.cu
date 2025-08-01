#include "dataloader.hpp"          // ➊ include the header

// --- FLAGS ------------------------------------------------------------ //
size_t batch_size = 32;            // ➋ add default
float  val_ratio  = 0.2f;

...
else if (arg == "--batch" && i+1 < argc) batch_size = std::stoul(argv[++i]);
else if (arg == "--val"   && i+1 < argc) val_ratio  = std::stof(argv[++i]);
...

// 2.  Load data (unchanged, produces h_X, h_y, in_dim, etc.)

// 2-b NEW: create DataLoader
DataLoader loader(h_X, h_y, in_dim, val_ratio);

// 4. Buffers (remove old upload code – DataLoader handles it)
float *d_pred, *d_grad, *d_loss_scalar;
CUDA_CHECK(cudaMalloc(&d_grad,  batch_size * out_dim * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_loss_scalar, sizeof(float)));

const int TPB = 128;

// 5. Training loop – iterate over mini-batches ➌
for (int epoch = 0; epoch < epochs; ++epoch) {
    size_t n_batches = loader.batches(batch_size, /*train=*/true);
    for (size_t bi = 0; bi < n_batches; ++bi) {
        auto [d_X, d_y, cur_bs] = loader.next_batch(batch_size, true);

        d_pred = const_cast<float*>(net.forward(d_X, cur_bs));

        cudaMemset(d_loss_scalar, 0, sizeof(float));
        int blocks = (cur_bs + TPB - 1) / TPB;
        k_mse_loss_grad<<<blocks, TPB>>>(d_pred, d_y,
                                         d_grad, d_loss_scalar, cur_bs);

        net.backward(d_y, d_grad, cur_bs);
        net.step();
    }

    if (epoch % 50 == 0) {      // quick val loss
        auto [d_Xv, d_yv, bs] = loader.next_batch(batch_size, false);
        d_pred = const_cast<float*>(net.forward(d_Xv, bs));
        cudaMemset(d_loss_scalar, 0, sizeof(float));
        int blocks = (bs + TPB - 1) / TPB;
        k_mse_loss_grad<<<blocks, TPB>>>(d_pred, d_yv,
                                         d_grad, d_loss_scalar, bs);
        float h_loss;
        cudaMemcpy(&h_loss, d_loss_scalar, sizeof(float),
                   cudaMemcpyDeviceToHost);
        printf("Epoch %4d | Val-MSE %.6f\n", epoch, h_loss);
    }
}
