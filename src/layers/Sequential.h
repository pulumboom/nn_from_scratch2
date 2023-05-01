#pragma once

#include <vector>
#include <any>

#include "ModuleBase.h"
#include "ModuleTypeErasure.h"

namespace Layers {
    class Sequential : public Base::ModuleBase {
    public:
        template<typename... Layers>
        Sequential(Layers&&... layers) {
            layers_.reserve(sizeof...(Layers));
            auto i = {layers_.emplace_back(std::forward<Layers>(layers))...};
        }
        Sequential(std::vector<Base::ModuleTypeErasure> layers);

        Base::Matrix Forward(const Base::Matrix &input) override;

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &grad_output) override;

        void ResetGrad() override;

        void UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) override;

    private:
        std::vector<Base::ModuleTypeErasure> layers_;
    };
}
