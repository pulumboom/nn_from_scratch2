#pragma once

#include <memory>

#include "ModuleBase.h"

namespace Layers {
    using Vector = Eigen::VectorXd;

    class LinearLayer : public Base::ModuleBase {
    public:
        LinearLayer(long long in_features, long long out_featuers, bool has_bias, size_t seed);
        LinearLayer(long long in_features, long long out_features, size_t seed);
        LinearLayer(long long in_features, long long out_features, bool has_bias);
        LinearLayer(long long in_features, long long out_features);

        Base::Matrix Forward(const Base::Matrix &input) override;

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &grad_output) override;

        void ResetGrad() override;

        void UpdateParameters(const Base::Matrix &input, Base::Matrix &grad_output) override;

        const Base::Matrix &Weights() const;
        Base::Matrix& Weights();

        const Vector &Bias() const;
        Vector& Bias();

    private:
        Base::Matrix weights_; // in_features x out_features
        Base::Matrix grad_for_weights_; // in_features x out_features
        std::unique_ptr<Vector> bias_ = nullptr; // out_features x 1
        std::unique_ptr<Vector> grad_for_bias_ = nullptr; // out_features x 1
        bool has_bias_ = true;
    };
}
