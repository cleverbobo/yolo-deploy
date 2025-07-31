#pragma once

#include <memory>

#include "pose.h"

class pose_factory {
public:
    pose_factory() = default;
    virtual ~pose_factory() = default;

    virtual std::shared_ptr<pose> getInstance(std::string, yoloType, int) = 0;

private:
    pose_factory(const pose_factory&) = delete;
    pose_factory& operator=(const pose_factory&) = delete;
};

class sophgo_pose_factory : public pose_factory {
public:
    sophgo_pose_factory();
    ~sophgo_pose_factory() override;

    std::shared_ptr<pose> getInstance(std::string, yoloType, int) override;

};

class trt_pose_factory : public pose_factory {
public:
    trt_pose_factory();
    ~trt_pose_factory() override;

    std::shared_ptr<pose> getInstance(std::string, yoloType, int) override;

};

