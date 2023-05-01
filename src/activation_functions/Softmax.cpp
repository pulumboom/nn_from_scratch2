#include "Softmax.h"

Base::Matrix AF::Softmax::Forward(const Base::Matrix &input) {
    output_ = input.array().exp();
    Eigen::VectorXd column_sum = output_.colwise().sum();
    output_.array().rowwise() /= column_sum.array().transpose();
    return output_;
}

Base::Matrix AF::Softmax::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
//    dinput = -output_.rowwise().transpose() * output_.array()
    // todo
}

void AF::Softmax::ResetGrad() {}

void AF::Softmax::UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) {}
