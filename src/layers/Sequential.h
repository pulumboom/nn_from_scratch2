#pragma once

#include <vector>
#include <any>

#include "ModuleInterface.h"
#include "ModuleTypeErasure.h"

namespace Layers {
    class Sequential {
    public:
        template<typename... Layers>
        Sequential(Layers&&... layers) {
            layers_.reserve(sizeof...(Layers));
            ([&]{
                layers_.emplace_back(std::move(layers));
            }(), ...);
        }
        
        Base::Matrix operator()(const Base::Matrix &input);
        
        Base::Matrix Forward(const Base::Matrix &input);

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &grad_output);

        void ResetGrad();
        
        void SwitchToTrainMode();
        
        void SwitchToTestMode();
        
        const Base::Matrix &Output() const;
        
        std::vector<Base::Matrix*> GetParameters();
        
        std::vector<Base::Matrix*> GetGradients();

    private:
        Base::Matrix output_;
        std::vector<Base::ModuleTypeErasure> layers_;
    };
}
