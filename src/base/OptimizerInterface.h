#pragma once

#include <memory>

namespace Base {
    class OptimizerInterface {
    public:
        virtual ~OptimizerInterface() = default;
        
        virtual void MakeStep() = 0;

//        virtual std::unique_ptr<OptimizerInterface> MakeCopy_() const = 0;
    };
}
