#pragma once

#include <memory>
#include "OptimizerInterface.h"

namespace Base {
    class OptimizerTypeErasure {
    public:
        OptimizerTypeErasure() = default;
        
        template<typename T>
        OptimizerTypeErasure(T &&optimizer) : model_(std::make_unique<Optimizer<T>>(std::move(optimizer))) {
            
        }

//        OptimizerTypeErasure(const OptimizerTypeErasure &other) : model_(
//                other.isDefined() ? other.model_->MakeCopy_() : nullptr
//                ) {}

        OptimizerTypeErasure(OptimizerTypeErasure &&) noexcept = default;

//        OptimizerTypeErasure &operator=(const OptimizerTypeErasure &other) {
//            return *this = OptimizerTypeErasure(other);
//        }

        OptimizerTypeErasure &operator=(OptimizerTypeErasure &&) noexcept = default;

        bool isDefined() const {
            return model_ == nullptr;
        }
        
        OptimizerInterface *operator->() {
            return model_.get();
        }
        
        const OptimizerInterface *operator->() const {
            return model_.get();
        }
        
    private:
        template<typename T>
        struct Optimizer : OptimizerInterface {
            Optimizer(T &&optimizer) : optimizer_(std::move(optimizer)) {}
            
            void MakeStep() override {
                optimizer_.MakeStep();
            }
            
//            std::unique_ptr<OptimizerInterface> MakeCopy_() const override {
//                return std::make_unique<Optimizer>(*this);
//            }
            
            T optimizer_;
        };
        
        std::unique_ptr<OptimizerInterface> model_ = nullptr;
    };
}
