# FAQ

## YOLOv5，YOLOv7的detect输出

问题：为啥这俩模型需要截断为三输出？其他模型则不需要截断？

回答：YOLOv5，YOLOv7是基于anchor的检测算法，所有部署的模型都要先转成onnx，转成onnx的过程中，anchor_size的信息会丢失，容易导致模型的生成的结果存在问题，所有将其截断为三输出。


## YOLOv8及其后续的detect算法的transpose算子问题

问题：为啥YOLOV8的模型后面需要添加transpose算子，将原本的[batch_size, feature_size, bbox_num] 改成[batch_size, bbox_num, feature_size]?

回答：为了保证C++中内存缓存连续命中，提高后处理效率。


