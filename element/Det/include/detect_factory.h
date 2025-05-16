#pragma once

#include <memory>

#include "detect.h"

class detect_factory {
public:
    detect_factory() = default;
    virtual ~detect_factory() = default;

    virtual std::shared_ptr<detect> getInstance(std::string, yoloType, int) = 0;

private:
    detect_factory(const detect_factory&) = delete;
    detect_factory& operator=(const detect_factory&) = delete;
};

class sophgo_detect_factory : public detect_factory {
public:
    sophgo_detect_factory();
    ~sophgo_detect_factory() override;

    std::shared_ptr<detect> getInstance(std::string, yoloType, int) override;

};

