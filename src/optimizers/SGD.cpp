#include "SGD.h"

Optimizers::SGD::SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion, double learning_rate,
                     double momentum_coefficient, double weight_decay) :
        module_(module), criterion_(criterion), learning_rate_(learning_rate),
        momentum_coefficient_(momentum_coefficient), weight_decay_(weight_decay) {
//    for (auto &params : (*module_)->GetParameters()) {
//        momentum_states_.push_back(std::make_unique<Base::Matrix>());
//    }
}

Optimizers::SGD::SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion, double learning_rate,
                     double momentum_coefficient) : module_(module), criterion_(criterion),
                                                    learning_rate_(learning_rate),
                                                    momentum_coefficient_(momentum_coefficient) {}

Optimizers::SGD::SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion, double learning_rate)
        : module_(module), criterion_(criterion), learning_rate_(learning_rate) {}

Optimizers::SGD::SGD(Base::ModuleTypeErasure *module, Base::CriterionTypeErasure *criterion) : module_(module),
                                                                                               criterion_(criterion) {}

void Optimizers::SGD::MakeStep() {

}
