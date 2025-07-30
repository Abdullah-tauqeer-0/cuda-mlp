// include/csv.hpp
#pragma once
#include <vector>
#include <string>

/**
 * Load a CSV whose last column is the numeric label.
 * Returns feature matrix (row-major) in h_X,
 * label vector in h_y, and feature dimension.
 */
void load_csv(const std::string& path,
              std::vector<float>& h_X,
              std::vector<float>& h_y,
              size_t& feature_dim);
