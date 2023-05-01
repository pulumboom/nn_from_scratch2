#include "MSE.h"

double MSE::Forward(Base::Matrix &input, Base::Matrix &target) {
    output_ = (input - target).array().pow(2).mean();
    return output_;
}

Base::Matrix MSE::Backward(Base::Matrix &input, Base::Matrix &target) {
    return 2 * (input - target);
}
