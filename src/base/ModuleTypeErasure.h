#pragma once

#include <memory>
#include "ModuleInterface.h"

namespace Base {
    class ModuleTypeErasure {
    public:
        ModuleTypeErasure() = default;

        template<typename T>
        ModuleTypeErasure(T &&layer) : model_(
                std::make_unique<Module<T>>(std::move(std::forward<T>(layer)))
        ) {}

        ModuleTypeErasure(const ModuleTypeErasure &other) : model_(
                other.isDefined() ? other.model_->MakeCopy_() : nullptr) {}

        ModuleTypeErasure(ModuleTypeErasure &&) noexcept = default;

        ModuleTypeErasure &operator=(const ModuleTypeErasure &other) {
            return *this = ModuleTypeErasure(other);
        }

        ModuleTypeErasure &operator=(ModuleTypeErasure &&) noexcept = default;

        bool isDefined() const {
            return model_ == nullptr;
        }

        ModuleInterface *operator->() {
            return model_.get();
        }

        const ModuleInterface *operator->() const {
            return model_.get();
        }

    private:
        template<typename T>
        struct Module : ModuleInterface {
            Module(T layer) : layer_(std::move(layer)) {}

            const Matrix &operator()(const Matrix &input) override {
                return layer_(input);
            }

            const Matrix &Forward(const Matrix &input) override {
                return layer_(input);
            }

            const Matrix &Backward(const Matrix &input, const Matrix &grad_output) override {
                return layer_.Backward(input, grad_output);
            }

            void ResetGrad() override {
                layer_.ResetGrad();
            }

            void SwitchToTrainMode() override {
                layer_.SwitchToTrainMode();
            }

            void SwitchToTestMode() override {
                layer_.SwitchToTestMode();
            }

            const Matrix &Output() const override {
                return layer_.Output();
            }

            std::vector<Matrix *> GetParameters() override {
                return layer_.GetParameters();
            }

            std::vector<Matrix *> GetGradients() override {
                return layer_.GetGradients();
            }

            std::unique_ptr<ModuleInterface> MakeCopy_() const override {
                return std::make_unique<Module>(*this);
            }

            T layer_;
        };

        std::unique_ptr<ModuleInterface> model_ = nullptr;
    };
}
