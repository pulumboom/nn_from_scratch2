#pragma once

#include <map>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "ModuleTypeErasure.h"

namespace Base {
    using Matrix = Eigen::MatrixXd;
    
    class OptimizerBase {
    public:
        void ResetGrad() {
            module_->ResetGrad();
        }
        
        virtual void MakeStep() = 0;
    protected:
        std::map<std::string, std::vector<Matrix>> states_;
        Base::ModuleTypeErasure module_;
    };
}
