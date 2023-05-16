#include "Softmax.h"

Base::Matrix AF::Softmax::operator()(const Base::Matrix &input) {
    output_ = input.array().exp();
    Eigen::VectorXd column_sum = output_.colwise().sum();
    output_.array().rowwise() /= column_sum.array().transpose();
    return output_;
}

Base::Matrix AF::Softmax::Forward(const Base::Matrix &input) {
    return (*this)(input);
}

Base::Matrix AF::Softmax::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
//    dinput = -output_.rowwise().transpose() * output_.array()
    // todo
    return {};
}

void AF::Softmax::ResetGrad() {}

void AF::Softmax::SwitchToTrainMode() {}

void AF::Softmax::SwitchToTestMode() {}

const Base::Matrix &AF::Softmax::Output() const {
    return output_;
}

std::vector<Base::Matrix*> AF::Softmax::GetParameters() {
    return {};
}

std::vector<Base::Matrix*> AF::Softmax::GetGradients() {
    return {};
}
