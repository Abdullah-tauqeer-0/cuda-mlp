#pragma once
#include <vector>
#include <random>

class DataLoader {
public:
    DataLoader(const std::vector<float>& X,
               const std::vector<float>& y,
               size_t feature_dim,
               float val_ratio = 0.2f,
               int   seed = 42);

    /** Returns number of mini-batches for current split. */
    size_t batches(size_t batch_size, bool train) const;

    /**
     * Copies the next mini-batch to GPU (allocates once, re-uses).
     *   train == true  → training split, otherwise validation split.
     * Returns tuple (d_X, d_y, current_batch_size).
     */
    std::tuple<const float*, const float*, size_t>
    next_batch(size_t batch_size, bool train);

private:
    void shuffle_train();

    std::vector<float> h_X_, h_y_;          // host rows (row-major)
    std::vector<size_t> idx_train_, idx_val_;
    size_t iter_ = 0;                       // batch iterator

    // Persistent device buffers (reused)
    float *d_X_ = nullptr, *d_y_ = nullptr;
    size_t feature_dim_, max_batch_;
};
