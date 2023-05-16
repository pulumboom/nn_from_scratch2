#pragma once

#include <vector>

#include "OptimizerInterface.h"
#include "ModuleTypeErasure.h"
#include "CriterionTypeErasure.h"

namespace Optimizers {
    class SGD {
    public:
        SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion, double learning_rate,
            double momentum_coefficient, double weight_decay);

        SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion, double learning_rate,
            double momentum_coefficient);

        SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion, double learning_rate);

        SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion);

        void MakeStep();

    private:
        std::vector<Base::Matrix *> momentum_states_;
        Base::ModuleTypeErasure *module_ = nullptr;
        Base::CriterionTypeErasure *criterion_ = nullptr;
        double learning_rate_ = 0.01;
        double momentum_coefficient_ = 0.;
        double weight_decay_ = 0.;
    };
}
