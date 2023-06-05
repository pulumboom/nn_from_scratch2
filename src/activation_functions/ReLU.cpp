#include "ReLU.h"

Base::Matrix AF::ReLU::operator()(const Base::Matrix &input) {
    return output_ = input.cwiseMax(0);
}

Base::Matrix AF::ReLU::Forward(const Base::Matrix &input) {
    return (*this)(input);
}

Base::Matrix AF::ReLU::Backward(const Base::Matrix &input, const Base::Matrix& grad_output) {
    return (input.array() > 0).select(grad_output, 0.0);
}

void AF::ReLU::ResetGrad() {}

void AF::ReLU::SwitchToTrainMode() {}

void AF::ReLU::SwitchToTestMode() {}

const Base::Matrix &AF::ReLU::Output() const {
    return output_;
}

std::vector<Base::Matrix*> AF::ReLU::GetParameters() {
    return {};
}

std::vector<Base::Matrix*> AF::ReLU::GetGradients() {
    return {};
}
