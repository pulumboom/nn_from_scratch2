#pragma once

#include <memory>

namespace Base {
    class ModuleTypeErasure {
    public:
        ModuleTypeErasure() = default;

        template<typename Layer>
        ModuleTypeErasure(Layer layer) : model_(
                std::make_unique<Layer>(std::move(std::forward(layer)))
        ) {}

        ModuleTypeErasure(const ModuleTypeErasure &other) : model_(
                other.isDefined() ? other.model_->MakeСopy_() : nullptr) {}

        ModuleTypeErasure &operator=(const ModuleTypeErasure &other) {
            return *this = ModuleTypeErasure(other);
        }

        ModuleTypeErasure(ModuleTypeErasure &&) noexcept = default;

        ModuleTypeErasure &operator=(ModuleTypeErasure &&) noexcept = default;

        bool isDefined() const {
            return model_ == nullptr;
        }

        ModuleBase *operator->() {
            return model_.get();
        }

        const ModuleBase *operator->() const {
            return model_.get();
        }

    private:
        std::unique_ptr<ModuleBase> model_ = nullptr;
    };
}
