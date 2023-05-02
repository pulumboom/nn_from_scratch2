#pragma once

#include <memory>

#include "Eigen/Dense"

namespace Base {
    using Matrix = Eigen::MatrixXd;

    class ModuleInterface {
    public:
        virtual ~ModuleInterface() = default;

        virtual const Matrix &operator()(const Matrix &) = 0;

        virtual const Matrix &Forward(const Matrix &) = 0;

        virtual const Matrix &Backward(const Matrix &, const Matrix &) = 0;

        virtual void ResetGrad() = 0;

        virtual void SwitchToTrainMode() = 0;

        virtual void SwitchToTestMode() = 0;

        virtual const Matrix &Output() const = 0;
        
        virtual std::vector<Matrix*> GetParameters() = 0;
        
        virtual std::vector<Matrix*> GetGradients() = 0;
        
        virtual std::unique_ptr<ModuleInterface> MakeCopy_() const = 0;
    };
}
