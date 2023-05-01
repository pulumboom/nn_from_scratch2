#pragma once

#include "OptimizerBase.h"
#include "ModuleTypeErasure.h"

namespace Optimizers {
    class SGD : public Base::OptimizerBase {
    public:
        SGD(Base::ModuleTypeErasure module, double learning_rate, double momentum, double weight_decay);
        SGD(Base::ModuleTypeErasure module, double learning_rate, double momentum);
        SGD(Base::ModuleTypeErasure module, double learning_rate);
        SGD(Base::ModuleTypeErasure module);
        
        void MakeStep() override;

    private:
        double learning_rate_ = 0.01;
        double momentum_ = 0.;
        double weight_decay_ = 0.;
    };
}
