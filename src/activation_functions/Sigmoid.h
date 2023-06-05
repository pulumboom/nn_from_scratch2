#pragma once

#include "ModuleInterface.h"

namespace AF {
    class Sigmoid {
    public:
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
    };
}
