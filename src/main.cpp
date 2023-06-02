#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <iostream>
#include <vector>

#include "LinearLayer.h"
#include "ReLU.h"
#include "Softmax.h"
#include "Sequential.h"
#include "OptimizerTypeErasure.h"
#include "CriterionTypeErasure.h"
#include "DataProcessing.h"
#include "SGD.h"
#include "CrossEntropyLoss.h"

namespace {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    void Train(
            Layers::Sequential &module,
            Optimizers::SGD &optimizer,
            Criterion::CrossEntropyLoss &criterion,
            Matrix &X,
            Matrix &y,
            long long batch_size,
            int epochs,
            bool debug = true) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0;
//            for (long long batch = 0; batch < X.rows(); batch += batch_size) {
            for (long long batch = 0; batch < 20; batch += batch_size) {
                auto batch_data = X.block(
                        batch,
                        0,
                        std::min(X.rows() - batch, batch_size),
                        X.cols());
                auto batch_labels = y.block(
                        batch,
                        0,
                        std::min(X.rows() - batch, batch_size),
                        y.cols());
                auto oneHotClasses = Utils::DataProcessing::OneHotEncode(batch_labels, 10);
                  
                auto output = module.Forward(batch_data);
                module.ResetGrad();
                double loss = criterion.Forward(output, oneHotClasses);
                epoch_loss += loss;
                module.Backward(batch_data, criterion.Backward(output, oneHotClasses));
                optimizer.MakeStep();
            }
            if (debug) {
                std::cout << std::fixed << "Epoch #" << epoch + 1 << ". Loss: " << epoch_loss << std::endl;
            }
        }
    }

    void Eval(
            Layers::Sequential &module,
            Criterion::CrossEntropyLoss &criterion,
            Matrix &X,
            Matrix &y,
            long long batch_size
    ) {
        double total_loss = 0;
        for (long long batch = 0; batch < X.rows(); batch += batch_size) {
            auto batch_data = X.block(
                    batch,
                    0,
                    std::min(X.rows() - batch, batch_size),
                    X.cols()
            );
            auto batch_labels = y.block(
                    batch,
                    0,
                    std::min(X.rows() - batch, batch_size),
                    y.cols()
            );
            auto oneHotClasses = Utils::DataProcessing::OneHotEncode(batch_labels, 10);
            auto output = module.Forward(batch_data);
            total_loss += criterion.Forward(output, oneHotClasses);
        }
        std::cout << "Evaluation Loss: " << total_loss << std::endl;
    }
}

int main() {
    Matrix data = Utils::DataProcessing::LoadCSV("../data/mnist/mnist_train.csv");
    Utils::Data data_struct = Utils::DataProcessing::SplitDataAndTarget(data);

    Matrix test = Utils::DataProcessing::LoadCSV("../data/mnist/mnist_test.csv");
    Utils::Data test_struct = Utils::DataProcessing::SplitDataAndTarget(test);

    Layers::Sequential nn(Layers::LinearLayer(784, 10), AF::Softmax());
    Criterion::CrossEntropyLoss cel;
    Optimizers::SGD optimizer_sgd(&nn);
    Train(nn, optimizer_sgd, cel, test_struct.data, test_struct.target, 32, 5, true);
    Eval(nn, cel, test_struct.data, test_struct.target, 32);
}
