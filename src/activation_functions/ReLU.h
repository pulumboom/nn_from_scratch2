#pragma once

#include "ModuleBase.h"

namespace AF {
    class ReLU : public Base::ModuleBase {
    public:
        Base::Matrix Forward(const Base::Matrix &input) override;

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &grad_output) override;

        void ResetGrad() override;

        void UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) override;
    };
}
