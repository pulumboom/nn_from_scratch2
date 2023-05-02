#pragma once

#include <memory>

#include "ModuleInterface.h"

namespace Layers {
    using Vector = Eigen::VectorXd;

    class LinearLayer {
    public:
        LinearLayer(long long in_features, long long out_features, bool has_bias, size_t seed);
        LinearLayer(long long in_features, long long out_features, size_t seed);
        LinearLayer(long long in_features, long long out_features, bool has_bias);
        LinearLayer(long long in_features, long long out_features);
        
        LinearLayer(const LinearLayer&); 
        
        Base::Matrix operator()(const Base::Matrix &input);

        Base::Matrix Forward(const Base::Matrix &input);

        Base::Matrix Backward(const Base::Matrix &input, const Base::Matrix &grad_output);

        void ResetGrad();
        
        void SwitchToTrainMode();
        
        void SwitchToTestMode();
        
        const Base::Matrix &Output() const;

        std::vector<Base::Matrix*> GetParameters();
        
        std::vector<Base::Matrix*> GetGradients();

    private:
        Base::Matrix output_;
        Base::Matrix weights_; // in_features x out_features
        Base::Matrix grad_for_weights_; // in_features x out_features
        std::unique_ptr<Vector> bias_ = nullptr; // out_features x 1
        std::unique_ptr<Vector> grad_for_bias_ = nullptr; // out_features x 1
        bool has_bias_ = true;
        bool training_mode_ = false;
    };
}
