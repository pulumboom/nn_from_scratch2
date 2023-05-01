#include "ReLU.h"

Base::Matrix AF::ReLU::Forward(const Base::Matrix &input) {
    return input.cwiseMax(0);
}

Base::Matrix AF::ReLU::Backward(const Base::Matrix &input, const Base::Matrix& grad_output) {
    return (input.array() > 0).select(grad_output, 0.0);
}

void AF::ReLU::ResetGrad() {}

void AF::ReLU::UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) {}
