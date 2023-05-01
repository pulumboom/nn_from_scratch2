#include "SGD.h"

Optimizers::SGD::SGD(Base::ModuleTypeErasure module, double learning_rate, double momentum, double weight_decay)
        : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
    module_ = module;
}

Optimizers::SGD::SGD(Base::ModuleTypeErasure module, double learning_rate, double momentum)
        : SGD(module, learning_rate, momentum, 0) {}

Optimizers::SGD::SGD(Base::ModuleTypeErasure module, double learning_rate)
        : SGD(module, learning_rate, 0) {}

Optimizers::SGD::SGD(Base::ModuleTypeErasure module)
        : SGD(module, 0.01) {}

void Optimizers::SGD::MakeStep() {

}
