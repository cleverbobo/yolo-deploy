# 功能规划
- 支持Det, Pose, Seg三种类型算法
- 支持sophgo, tensor_rt, rknn, cpu 
- 基于抽象工厂模式实现每种算法实例
- 支持C++算法，对外提供pybind接口
- 提供模型、数据集下载办法
- 支持多线程[pending，暂不打算支持]
- 使用spdlog进行日志系统管理与调度

# 功能开发时间轴
目前已支持tensorrt,sophgo框架，支持了YOLO全系列detect算法

- 7月完成 segment全系列算法
- 8月完成 pose全系列算法，开发文档，模型以及数据集。
- 9月完成 支持rknn segment、detect、pose算法
- 10月完成 pybind接口开发，以及视频录制。

# 备注
- 目前还在开发阶段，代码不稳定，预计8月份完成发版
- 环境依赖备注：opencv，cnpy