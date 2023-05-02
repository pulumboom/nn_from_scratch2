#include "Sigmoid.h"

Base::Matrix AF::Sigmoid::operator()(const Base::Matrix &input) {
    output_ = 1 / (1 + input.array().exp());
    return output_;
}

Base::Matrix AF::Sigmoid::Forward(const Base::Matrix &input) {
    return (*this)(input);
}

Base::Matrix AF::Sigmoid::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
    return grad_output.array() * output_.array() * (1 - output_.array());
}

void AF::Sigmoid::ResetGrad() {}

void AF::Sigmoid::SwitchToTrainMode() {}

void AF::Sigmoid::SwitchToTestMode() {}

const Base::Matrix &AF::Sigmoid::Output() const {
    return output_;
}

std::vector<Base::Matrix*> AF::Sigmoid::GetParameters() {}

std::vector<Base::Matrix*> AF::Sigmoid::GetGradients() {}
