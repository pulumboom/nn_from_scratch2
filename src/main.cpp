//#include "nn/MLP.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <any>

#include "LinearLayer.h"
#include "ReLU.h"
#include "Sequential.h"

namespace {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    template<typename T>
    int foo(T x) {
        int y = x;
        y++;
        return y;
    }
}

int main() {
//    std::vector<Base::ModuleTypeErasure> mm = {Layers::LinearLayer(1, 2), AF::ReLU()};
    Layers::LinearLayer ll(2, 3);
}
