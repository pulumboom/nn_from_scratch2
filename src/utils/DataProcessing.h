#pragma once

#include <string>

#include "Eigen/Dense"
#include "ModuleInterface.h"

namespace Utils {
    using Matrix = Eigen::MatrixXd;
    
    struct Data {
        Matrix data;
        Matrix target;
    };

    class DataProcessing {
    public:
        static Matrix LoadCSV(const std::string &path, char sep = ',', bool skip_first = true);
        static Data SplitDataAndTarget(const Matrix &data, size_t target_col_number = 0);
        static Matrix OneHotEncode(const Matrix& target, int classes_number);
    };
}
