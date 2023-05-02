#pragma once

#include <memory>
#include "CriterionInterface.h"

namespace Base {
    class CriterionTypeErasure {
    public:
        CriterionTypeErasure() = default;

        template<typename T>
        CriterionTypeErasure(T &&criterion) : model_(
                std::make_unique<Criterion<T>>(std::move(std::forward<T>(criterion)))
        ) {}

        CriterionTypeErasure(const CriterionTypeErasure &other) : model_(
                other.isDefined() ? other.model_->MakeCopy_() : nullptr) {}

        CriterionTypeErasure(CriterionTypeErasure &&) noexcept = default;

        CriterionTypeErasure &operator=(const CriterionTypeErasure &other) {
            return *this = CriterionTypeErasure(other);
        }

        CriterionTypeErasure &operator=(CriterionTypeErasure &&) noexcept = default;

        bool isDefined() const {
            return model_ == nullptr;
        }

        CriterionInterface *operator->() {
            return model_.get();
        }

        const CriterionInterface *operator->() const {
            return model_.get();
        }

    private:
        template<typename T>
        struct Criterion : CriterionInterface {
            Criterion(T criterion) : criterion_(std::move(criterion)) {}

            double operator()(const Matrix &input, const Matrix &target) override {
                return criterion_(input, target);
            }

            double Forward(const Matrix &input, const Matrix &target) override {
                return criterion_.Forward(input, target);
            }

            Matrix Backward(const Matrix &input, const Matrix &target) {
                return criterion_.Backward(input, target);
            }

            std::unique_ptr<CriterionInterface> MakeCopy_() const override {
                return std::make_unique<Criterion>(*this);
            }

            T criterion_;
        };

        std::unique_ptr<CriterionInterface> model_ = nullptr;
    };
}
