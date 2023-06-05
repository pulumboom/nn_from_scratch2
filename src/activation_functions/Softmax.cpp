#include "Softmax.h"

Base::Matrix AF::Softmax::operator()(const Base::Matrix &input) {
    output_ = (input.colwise() - input.rowwise().maxCoeff()).array().exp();
    output_.colwise() -= output_.rowwise().maxCoeff();
    Eigen::VectorXd column_sum = output_.colwise().sum();
    output_.array().rowwise() /= (column_sum.array() + 0.00001).transpose();
    return output_;
}

Base::Matrix AF::Softmax::Forward(const Base::Matrix &input) {
    return (*this)(input);
}

Base::Matrix AF::Softmax::Backward(const Base::Matrix &input, const Base::Matrix &grad_output) {
    output_ = Forward(input);
    Base::Matrix result(input.rows(), input.cols());
    result.setZero();
    for (int i = 0; i < input.rows(); ++i) {
        result.row(i) = grad_output.row(i).array() * (output_.row(i).array() * (1.0 - output_.row(i).array())).array();
    }
    return result;
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
