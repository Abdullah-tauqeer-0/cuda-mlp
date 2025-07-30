// src/csv.cpp
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "csv.hpp"

void load_csv(const std::string& path,
              std::vector<float>& h_X,
              std::vector<float>& h_y,
              size_t& feature_dim)
{
    std::ifstream fin(path);
    if (!fin.good())
        throw std::runtime_error("Unable to open CSV: " + path);

    std::string line;
    feature_dim = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ',')) row.push_back(std::stof(cell));

        if (feature_dim == 0) feature_dim = row.size() - 1;
        if (row.size() != feature_dim + 1)
            throw std::runtime_error("Inconsistent column count");

        // copy features then label
        h_X.insert(h_X.end(), row.begin(), row.end() - 1);
        h_y.push_back(row.back());
    }
}
