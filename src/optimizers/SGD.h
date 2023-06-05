#pragma once

#include <vector>

#include "Sequential.h"

namespace Optimizers {
    class SGD {
    public:
        SGD(Layers::Sequential *module, double learning_rate, double momentum_coefficient, double weight_decay);

        SGD(Layers::Sequential *module, double learning_rate, double momentum_coefficient);

        SGD(Layers::Sequential *module, double learning_rate);

        SGD(Layers::Sequential *module);

        void MakeStep();

        void SetLearningRate(double new_learning_rate);

        double GetLearningRate() const;

        void SetMomentum(double new_momentum);

        double GetMomentum() const;

        void SetWeightDecay(double new_weight_decay);

        double GetWeightDecay() const;

    private:
        std::vector<Base::Matrix *> params_;
        std::vector<Base::Matrix *> grads_;
        std::vector<std::unique_ptr<Base::Matrix>> momentum_states_;
        Layers::Sequential *module_ = nullptr;
        double learning_rate_ = 0.01;
        double momentum_coefficient_ = 0.;
        double weight_decay_ = 0.;
    };
}
