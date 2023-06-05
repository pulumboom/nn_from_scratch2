#include "SGD.h"

Optimizers::SGD::SGD(
        Layers::Sequential *module,
        double learning_rate,
        double momentum_coefficient,
        double weight_decay) :
        module_(module),
        learning_rate_(learning_rate),
        momentum_coefficient_(momentum_coefficient),
        weight_decay_(weight_decay) {
    params_ = module_->GetParameters();
    grads_ = module_->GetGradients();
    if (momentum_coefficient == 0) {
        return;
    }
    std::vector<Base::Matrix *> params = module_->GetParameters();
    momentum_states_.reserve(params.size());
    for (auto param: params) {
        momentum_states_.push_back(std::make_unique<Base::Matrix>(param->rows(), param->cols()));
        momentum_states_.back()->setZero();
    }
}

Optimizers::SGD::SGD(
        Layers::Sequential *module,
        double learning_rate,
        double momentum_coefficient) :
        Optimizers::SGD::SGD(module, learning_rate, momentum_coefficient, 0.) {}

Optimizers::SGD::SGD(Layers::Sequential *module, double learning_rate)
        : Optimizers::SGD::SGD(module, learning_rate, 0.) {}

Optimizers::SGD::SGD(Layers::Sequential *module) : Optimizers::SGD::SGD(module, 0.01) {}

void Optimizers::SGD::MakeStep() {
    for (int i = 0; i < params_.size(); ++i) {
        if (weight_decay_ != 0) {
            *grads_[i] += weight_decay_ * (*params_[i]);
        }
        if (momentum_coefficient_ != 0) {
            *momentum_states_[i] *= momentum_coefficient_;
            *momentum_states_[i] += *grads_[i];
            *params_[i] -= learning_rate_ * (*momentum_states_[i]);
        } else {
            *params_[i] -= learning_rate_ * (*grads_[i]);
        }
    }
}

void Optimizers::SGD::SetLearningRate(double new_learning_rate) {
    learning_rate_ = new_learning_rate;
}

double Optimizers::SGD::GetLearningRate() const {
    return learning_rate_;
}

void Optimizers::SGD::SetMomentum(double new_momentum) {
    if (momentum_coefficient_ == 0 and new_momentum != 0) {
        std::vector<Base::Matrix *> params = module_->GetParameters();
        momentum_states_.reserve(params.size());
        for (auto param: params) {
            momentum_states_.push_back(std::make_unique<Base::Matrix>(param->rows(), param->cols()));
            momentum_states_.back()->setZero();
        }
    }
    momentum_coefficient_ = new_momentum;
}

double Optimizers::SGD::GetMomentum() const {
    return momentum_coefficient_;
}

void Optimizers::SGD::SetWeightDecay(double new_weight_decay) {
    weight_decay_ = new_weight_decay;
}

double Optimizers::SGD::GetWeightDecay() const {
    return weight_decay_;
}
