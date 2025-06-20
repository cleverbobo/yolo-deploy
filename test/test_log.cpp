#include "log.h"

int main(int argc ,char **argv) {
    std::string logLevel;
    if (argc > 1) {
        logLevel = argv[1];
    } else {
        logLevel = "info";
    }
    logInit(logLevel);

    // test_log
    YOLO_TRACE("test_case:{}, This is a trace message", 1);
    YOLO_DEBUG("test_case:{}, This is a debug message", 2);
    YOLO_INFO("test_case:{}, This is an info message", 3);
    YOLO_WARN("test_case:{}, This is a warning message", 4);
    YOLO_ERROR("test_case:{}, This is an error message", 5);
    YOLO_CRITICAL("test_case:{}, This is a critical message", 6);
    return 1;
}