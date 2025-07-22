#pragma

#include <memory>

#include "segment.h"

class segment_factory {
public:
    segment_factory() = default;
    virtual ~segment_factory() = default;

    virtual std::shared_ptr<segment> getInstance(std::string, yoloType, int) = 0;
    
private:
    segment_factory(const segment_factory&) = delete;
    segment_factory& operator=(const segment_factory&) = delete;
};

class sophgo_segment_factory : public segment_factory {
public:
    sophgo_segment_factory() = default;
    ~sophgo_segment_factory() = default;

    std::shared_ptr<segment> getInstance(std::string, yoloType, int) override;
};


class trt_segment_factory : public segment_factory {
public:
    trt_segment_factory() = default;
    ~trt_segment_factory() = default;

    std::shared_ptr<segment> getInstance(std::string, yoloType, int) override;
};