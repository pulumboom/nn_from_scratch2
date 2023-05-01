#pragma once

#include <memory>

#include "Eigen/Dense"

namespace Base {
    using Matrix = Eigen::MatrixXd;

    class ModuleBase {
    public:
        virtual ~ModuleBase() = default;

        Matrix operator()(const Matrix &input) {
            return Forward(input);
        }

        virtual Matrix Forward(const Matrix &) = 0;

        virtual Matrix Backward(const Matrix &, const Matrix &) = 0;

        virtual void ResetGrad() = 0;

        virtual void UpdateParameters(const Matrix &, Matrix &) = 0;

        void SwitchToTrainMode() {
            training_mode_ = true;
        }

        void SwitchToEvalMode() {
            training_mode_ = false;
        }

        const Matrix &Output() const {
            return output_;
        }

        virtual std::unique_ptr<ModuleBase> MakeCopy_() const = 0;

    protected:
        Matrix output_;
        bool training_mode_ = true;
    };
}
