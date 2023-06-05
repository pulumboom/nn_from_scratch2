#include "MAE.h"

double Criterion::MAE::operator()(const Base::Matrix &input, const Base::Matrix &target) {
    output_ = (input - target).array().abs().mean();
    return output_;
}

double Criterion::MAE::Forward(const Base::Matrix &input, const Base::Matrix &target) {
    return (*this)(input, target);
}

Base::Matrix Criterion::MAE::Backward(const Base::Matrix &input, const Base::Matrix &target) {
    return (input.array() > target.array()).select(1, -1);
}
