#include "CrossEntropyLoss.h"

double Criterion::CrossEntropyLoss::operator()(const Base::Matrix &input, const Base::Matrix &target) {
//   output_ = -1 /
    return 0.0;
    //todo
}

double Criterion::CrossEntropyLoss::Forward(const Base::Matrix &input, const Base::Matrix &target) {
    return (*this)(input, target);
}

Base::Matrix Criterion::CrossEntropyLoss::Backward(const Base::Matrix &input, const Base::Matrix &target) {
    return Base::Matrix();
    //todo
}