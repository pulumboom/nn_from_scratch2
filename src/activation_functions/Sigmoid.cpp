#include "Sigmoid.h"

Base::Matrix AF::Sigmoid::Forward(const Base::Matrix &input) {
    output_ = 1 / (1 + input.array().exp());
    return output_;
}

Base::Matrix AF::Sigmoid::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
    return grad_output.array() * output_.array() * (1 - output_.array());
}

void AF::Sigmoid::ResetGrad() {}

void AF::Sigmoid::UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) {}
