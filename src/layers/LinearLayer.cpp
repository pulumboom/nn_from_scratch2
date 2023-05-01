#include <EigenRand/EigenRand>

#include "LinearLayer.h"

Layers::LinearLayer::LinearLayer(long long in_features, long long out_features, bool has_bias, size_t seed)
        : grad_for_weights_(Base::Matrix::Zero(in_features, out_features)),
          has_bias_(has_bias) {
    Eigen::Rand::P8_mt19937_64 urng = { seed };
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

Base::Matrix Layers::LinearLayer::Forward(const Base::Matrix &input) {
    output_ = (input * weights_.transpose()).rowwise() + (*bias_).transpose();
    return output_;
}

Base::Matrix Layers::LinearLayer::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    Forward(input);
    return grad_output * weights_;
}

void Layers::LinearLayer::ResetGrad() {
    grad_for_weights_.setZero();

    if (has_bias_) {
        grad_for_bias_->setZero();
    }
}

void Layers::LinearLayer::UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) {
    grad_for_weights_ += grad_output.transpose() * input;

    if (has_bias_) {
        *grad_for_bias_ += grad_output.colwise().sum();
    }
}

const Base::Matrix &Layers::LinearLayer::Weights() const {
    return weights_;
}

Base::Matrix &Layers::LinearLayer::Weights() {
    return weights_;
}

const Layers::Vector &Layers::LinearLayer::Bias() const {
    return *bias_;
}

Layers::Vector &Layers::LinearLayer::Bias() {
    return *bias_;
}
