#pragma once

#include "CriterionInterface.h"

class CrossEntropyLoss {
public:
    double operator()(const Base::Matrix &input, const Base::Matrix &target);

    double Forward(const Base::Matrix &input, const Base::Matrix &target);

    Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &target);

private:
    double output_;
};
