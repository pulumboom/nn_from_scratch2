#include <EigenRand/EigenRand>

#include "LinearLayer.h"

Layers::LinearLayer::LinearLayer(long long in_features, long long out_features, bool has_bias, size_t seed)
        : grad_for_weights_(Base::Matrix::Zero(in_features, out_features)),
          has_bias_(has_bias) {
    Eigen::Rand::P8_mt19937_64 urng = {seed};
    weights_ = Eigen::Rand::normal<Base::Matrix>(in_features, out_features, urng);
    if (has_bias_) {
        *bias_ = Eigen::Rand::normal<Vector>(-1, out_features, urng);
        *grad_for_bias_ = Vector::Zero(out_features);
    }
}

Layers::LinearLayer::LinearLayer(long long in_features, long long out_features, size_t seed)
        : LinearLayer(in_features, out_features, true, seed) {}

Layers::LinearLayer::LinearLayer(long long in_features, long long out_features, bool has_bias)
        : LinearLayer(in_features, out_features, has_bias, 42) {}

Layers::LinearLayer::LinearLayer(long long in_features, long long out_features)
        : LinearLayer(in_features, out_features, true) {}

Layers::LinearLayer::LinearLayer(const Layers::LinearLayer &other)
        : weights_(other.weights_),
        grad_for_weights_(other.grad_for_weights_),
        has_bias_(other.has_bias_) {
    if (has_bias_) {
        bias_ = std::make_unique<Vector>(*other.bias_);
        grad_for_bias_ = std::make_unique<Vector>(*other.grad_for_bias_);
    }
}

Base::Matrix Layers::LinearLayer::operator()(const Base::Matrix &input) {
    output_ = (input * weights_.transpose()).rowwise() + (*bias_).transpose();
    return output_;
}

Base::Matrix Layers::LinearLayer::Forward(const Base::Matrix &input) {
    return (*this)(input);
}

Base::Matrix Layers::LinearLayer::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    Forward(input);
    
    grad_for_weights_ += grad_output.transpose() * input;
    
    if (has_bias_) {
        *grad_for_bias_ += grad_output.colwise().sum();
    }
    
    return grad_output * weights_;
}

void Layers::LinearLayer::ResetGrad() {
    grad_for_weights_.setZero();

    if (has_bias_) {
        grad_for_bias_->setZero();
    }
}

void Layers::LinearLayer::SwitchToTrainMode() {
    training_mode_ = true;
}

void Layers::LinearLayer::SwitchToTestMode() {
    training_mode_ = false;
}

const Base::Matrix &Layers::LinearLayer::Output() const {
    return output_;
}

std::vector<Base::Matrix*> Layers::LinearLayer::GetParameters() {
    std::vector<Base::Matrix*> parameters;
    parameters.push_back(&weights_);
    std::vector<Base::Matrix> p;
    if (has_bias_) {
        parameters.push_back((Base::Matrix*)bias_.get());
    }
}

std::vector<Base::Matrix*> Layers::LinearLayer::GetGradients() {
    std::vector<Base::Matrix*> gradients;
    gradients.push_back(&grad_for_weights_);
    if (has_bias_) {
        gradients.push_back((Base::Matrix*)grad_for_bias_.get());
    }
}
