#include "Sequential.h"

Base::Matrix Layers::Sequential::operator()(const Base::Matrix &input) {
    output_ = input;
    for (auto &layer : layers_) {
        output_ = layer->Forward(output_);
    }
    return output_;
}

Base::Matrix Layers::Sequential::Forward(const Base::Matrix &input) {
    return (*this)(input);
}

Base::Matrix Layers::Sequential::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
    Base::Matrix grad = grad_output;
    for (int i = layers_.size() - 1; i > -1; --i) {
        if (i > 0) {
            grad = layers_[i]->Backward(layers_[i - 1]->Output(), grad);
        } else {
            grad = layers_[i]->Backward(input, grad);
        }
    }
    return grad;
}

void Layers::Sequential::ResetGrad() {
    for (auto &layer : layers_) {
        layer->ResetGrad();
    }
}

void Layers::Sequential::SwitchToTrainMode() {
    for (auto &layer : layers_) {
        layer->SwitchToTrainMode();
    }
}

void Layers::Sequential::SwitchToTestMode() {
    for (auto &layer : layers_) {
        layer->SwitchToTestMode();
    }
}

const Base::Matrix &Layers::Sequential::Output() const {
    return output_;
}

std::vector<Base::Matrix*> Layers::Sequential::GetParameters() {
    std::vector<Base::Matrix*> parameters;
    for (auto &layer : layers_) {
        for (auto param : layer->GetParameters()) {
            parameters.push_back(param);
        }
    }
    return parameters;
}

std::vector<Base::Matrix*> Layers::Sequential::GetGradients() {
    std::vector<Base::Matrix*> gradients;
    for (auto &layer : layers_) {
        for (auto grad : layer->GetGradients()) {
            gradients.push_back(grad);
        }
    }
    return gradients;
}
