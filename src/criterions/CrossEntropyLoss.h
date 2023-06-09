#pragma once

#include "CriterionInterface.h"

#include "Softmax.h"

namespace Criterion {
    class CrossEntropyLoss {
    public:
        double operator()(const Base::Matrix &input, const Base::Matrix &target);

        double Forward(const Base::Matrix &input, const Base::Matrix &target);

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &target);

    private:
        double output_;
        AF::Softmax sf;
    };
}
