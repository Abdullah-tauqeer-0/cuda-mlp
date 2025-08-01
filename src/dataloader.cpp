#include "dataloader.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "mlp.hpp"     // for CUDA_CHECK

DataLoader::DataLoader(const std::vector<float>& X,
                       const std::vector<float>& y,
                       size_t feature_dim,
                       float val_ratio,
                       int seed)
  : h_X_(X), h_y_(y), feature_dim_(feature_dim)
{
    size_t N = y.size();
    idx_train_.resize(N);
    std::iota(idx_train_.begin(), idx_train_.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(idx_train_.begin(), idx_train_.end(), rng);

    size_t val_n = static_cast<size_t>(N * val_ratio + 0.5f);
    idx_val_.assign(idx_train_.end() - val_n, idx_train_.end());
    idx_train_.resize(N - val_n);

    max_batch_ = *std::max_element({idx_train_.size(), idx_val_.size()});

    // allocate once (largest possible)
    CUDA_CHECK(cudaMalloc(&d_X_, max_batch_ * feature_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_, max_batch_ * sizeof(float)));
}

void DataLoader::shuffle_train()
{
    std::mt19937 rng(123 + std::random_device{}());
    std::shuffle(idx_train_.begin(), idx_train_.end(), rng);
}

size_t DataLoader::batches(size_t bs, bool train) const
{
    const auto& idx = train ? idx_train_ : idx_val_;
    return (idx.size() + bs - 1) / bs;
}

std::tuple<const float*, const float*, size_t>
DataLoader::next_batch(size_t bs, bool train)
{
    auto& idx = train ? idx_train_ : idx_val_;

    if (train && iter_ == 0) shuffle_train();

    size_t start = iter_ * bs;
    size_t end   = std::min(start + bs, idx.size());
    size_t cur   = end - start;
    if (cur == 0) {           // epoch finished
        iter_ = 0;
        return next_batch(bs, train);
    }

    // gather -> host scratch buffers
    std::vector<float> h_x(cur * feature_dim_);
    std::vector<float> h_y(cur);
    for (size_t i = 0; i < cur; ++i) {
        size_t src = idx[start + i];
        std::copy_n(&h_X_[src * feature_dim_], feature_dim_,
                    &h_x[i * feature_dim_]);
        h_y[i] = h_y_[src];
    }

    CUDA_CHECK(cudaMemcpy(d_X_, h_x.data(),
               cur * feature_dim_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_, h_y.data(),
               cur * sizeof(float), cudaMemcpyHostToDevice));

    ++iter_;
    return {d_X_, d_y_, cur};
}
