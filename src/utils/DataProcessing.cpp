#include "DataProcessing.h"

#include <fstream>
#include <vector>
#include <iostream>
Utils::Matrix Utils::DataProcessing::LoadCSV(const std::string &path, const char sep, bool skip_first) {
    std::ifstream file(path);
    std::string line;
    std::vector<double> values;
    long long rows = 0;
    while (std::getline(file, line)) {
        if (skip_first) {
            skip_first = false;
            continue;
        }
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, sep)) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<Utils::Matrix>(values.data(), values.size() / rows, rows).transpose();
}

Utils::Data Utils::DataProcessing::SplitDataAndTarget(const Utils::Matrix &data, size_t target_col_number) {
    Data result;
    result.target = data.col(target_col_number);
    if (target_col_number == 0) {
        result.data = data.block(0, 1, data.rows(), data.cols() - 1);
    } else if (target_col_number == data.cols() - 1) {
        result.data = data.block(0, 0, data.rows(), data.cols() - 1);
    } else {
        result.data = Matrix(data.rows(), data.cols() - 1);
        result.data << data.block(0, 0, data.rows(), target_col_number - 1),
                data.block(0, target_col_number + 1, data.rows(), data.cols() - 1);
    }
    return result;
}

Utils::Matrix Utils::DataProcessing::OneHotEncode(const Utils::Matrix &target, int classes_number) {
    Utils::Matrix result = Utils::Matrix::Zero(target.rows(), classes_number);
    for (int i = 0; i < target.rows(); ++i) {
        result(i, static_cast<int>(target(i, 0))) = 1.0;
    }
    return result;
}
