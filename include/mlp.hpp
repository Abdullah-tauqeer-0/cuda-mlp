// include/mlp.hpp
// Simple CUDA-powered MLP — header only
// Abdullah Tauqeer, 2025-07-29
#pragma once

#include <vector>
#include <cuda_runtime.h>

//---------- Convenience ----------------------------------------------------//
using fp32 = float;                         // change to half for fp16 later
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) std::exit(code);
    }
}

//---------- Activation -----------------------------------------------------//
enum class ActType { RELU, SIGMOID, NONE };

// Forward / backward kernel prototypes (to be implemented in utils.cu)
void act_forward(const fp32* d_in, fp32* d_out,
                 size_t n, ActType type);
void act_backward(const fp32* d_in, const fp32* d_grad_out,
                  fp32* d_grad_in, size_t n, ActType type);

//---------- MLP Layer ------------------------------------------------------//
struct Layer {
    size_t in_dim, out_dim;
    ActType act;

    // Device pointers
    fp32 *d_W  = nullptr;   // [out_dim, in_dim]
    fp32 *d_b  = nullptr;   // [out_dim]
    fp32 *d_dW = nullptr;   // grads
    fp32 *d_db = nullptr;

    // Cached activations for back-prop
    fp32 *d_out = nullptr;  // output after activation
    fp32 *d_z   = nullptr;  // pre-activation linear output

    explicit Layer(size_t in_dim_, size_t out_dim_, ActType act_);
    ~Layer();

    // Disable copy, allow move
    Layer(const Layer&)            = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) noexcept        = default;
};

//---------- MLP Network ----------------------------------------------------//
class MLP {
public:
    MLP(const std::vector<size_t>& dims,
        const std::vector<ActType>& acts,
        fp32 lr = 1e-2f);

    ~MLP();

    // Forward pass: returns pointer to network output on device
    //  x: [batch, input_dim] device pointer
    const fp32* forward(const fp32* d_x, size_t batch);

    // Backward pass: y_true and loss grad expected on device
    //  y_true: [batch, output_dim] one-hot or regression targets
    //  d_loss: gradient of loss wrt network output (same shape)
    void backward(const fp32* d_y_true, const fp32* d_loss, size_t batch);

    // SGD weight update (could be Adam later)
    void step(fp32 lr_override = -1.f);

    size_t output_dim() const { return layers_.back().out_dim; }

private:
    std::vector<Layer> layers_;
    fp32 lr_;
};
