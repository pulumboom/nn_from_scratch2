#pragma once

#include "Eigen/Dense"

namespace Base {
    using Matrix = Eigen::MatrixXd;

    class CriterionBase {
    public:
        double operator()(Matrix &input, Matrix &target) {
            return Forward(input, target);
        }

        virtual double Forward(Matrix &input, Matrix &target) = 0;

        virtual Matrix Backward(Matrix &input, Matrix &target) = 0;

    protected:
        double output_ = 0;
    };
}
