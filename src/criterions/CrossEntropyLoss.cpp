#include "CrossEntropyLoss.h"

double Criterion::CrossEntropyLoss::operator()(const Base::Matrix &input, const Base::Matrix &target) {
    auto prob = sf(input).array().log().matrix();
    output_ = -(prob.array() * target.array()).sum() / input.rows();
    return output_;
}

double Criterion::CrossEntropyLoss::Forward(const Base::Matrix &input, const Base::Matrix &target) {
    return (*this)(input, target);
}

Base::Matrix Criterion::CrossEntropyLoss::Backward(const Base::Matrix &input, const Base::Matrix &target) {
    auto prob = sf(input).array().log().matrix();
    return -1 / input.rows() / sf.Backward(input, target).array();
}
