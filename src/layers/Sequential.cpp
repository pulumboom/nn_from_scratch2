#include "Sequential.h"

Layers::Sequential::Sequential(std::vector<Base::ModuleTypeErasure> layers) : layers_(std::move(layers)) {}

Base::Matrix Layers::Sequential::Forward(const Base::Matrix &input) {
    output_ = input;
    for (auto &layer : layers_) {
        output_ = layer->Forward(input);
    }
    return output_;
}

Base::Matrix Layers::Sequential::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
    for (int i = layers_.size() - 1; i > -1; --i) {
        if (i > 0) {
            layers_[i]->UpdateParameters(layers_[i - 1]->Output(), grad_output);
            grad_output = layers_[i]->Backward(layers_[i - 1]->Output(), grad_output);
        } else {
            layers_[i]->UpdateParameters(input, grad_output);
            grad_output = layers_[i]->Backward(input, grad_output);
        }
    }
    return grad_output;
}

void Layers::Sequential::ResetGrad() {
    for (auto &layer : layers_) {
        layer->ResetGrad();
    }
}

void Layers::Sequential::UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) {}
