#pragma once

#include "CriterionTypeErasure.h"

namespace Criterion {
    class MSE {
    public:
        double operator()(const Base::Matrix &input, const Base::Matrix &target);

        double Forward(const Base::Matrix &input, const Base::Matrix &target);

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &target);

    private:
        double output_ = 0;
    };

}
