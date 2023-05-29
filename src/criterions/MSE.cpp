#include "MSE.h"

double Criterion::MSE::operator()(const Base::Matrix &input, const Base::Matrix &target) {
    output_ = (input - target).array().pow(2).mean();
    return output_;
}

double Criterion::MSE::Forward(const Base::Matrix &input, const Base::Matrix &target) {
    return (*this)(input, target);
}

Base::Matrix Criterion::MSE::Backward(const Base::Matrix &input, const Base::Matrix &target) {
    return 2 * (input - target);
}
