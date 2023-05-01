#pragma once

#include "CriterionBase.h"

class CrossEntropyLoss : public Base::CriterionBase {
public:
    double Forward(Base::Matrix &input, Base::Matrix &target) override;
    Base::Matrix Backward(Base::Matrix &input, Base::Matrix &target) override;

private:
//todo
};
