// include/utils.hpp
#pragma once
#include "mlp.hpp"

// ---- Linear layer helpers ---- //
// (Implemented in src/utils.cu)
void linear_forward(const fp32* d_X,          // [batch, in_dim]
                    const fp32* d_W,          // [out_dim, in_dim] (row-major)
                    const fp32* d_b,          // [out_dim]
                    fp32*       d_Z,          // [batch, out_dim]  (output)
                    size_t batch,
                    size_t in_dim,
                    size_t out_dim);

void add_bias(const fp32* d_b, fp32* d_Z,
              size_t batch, size_t out_dim);
