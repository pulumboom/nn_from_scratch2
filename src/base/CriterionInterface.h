#pragma once

#include <memory>

#include "Eigen/Dense"

namespace Base {
    using Matrix = Eigen::MatrixXd;

    class CriterionInterface {
    public:
        virtual ~CriterionInterface() = default;
        
        virtual double operator()(const Matrix &, const Matrix &) = 0;

        virtual double Forward(const Matrix &, const Matrix &) = 0;

        virtual Matrix Backward(const Matrix &, const Matrix &) = 0;
        
//        virtual std::unique_ptr<CriterionInterface> MakeCopy_() const = 0;
    };
}
