#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include <iostream>


inline const char* loggerName = "yoloDeployLogger";

class yoloLogger {

public:
    ~yoloLogger();
    static std::shared_ptr<spdlog::logger> getInstance(std::string level="info",std::string fileLogPath = "");
    void setLevel(std::string level);


private:
    std::shared_ptr<spdlog::logger> m_logger;
    yoloLogger(std::string level, std::string fileLogPath);
    yoloLogger(const yoloLogger&) = delete;
    yoloLogger& operator=(const yoloLogger&) = delete;
};


#define YOLOLOG_STR(integer) #integer
#define YOLOLOG_STR_HELP(integer) YOLOLOG_STR(integer)
#ifdef __FILENAME__
#define YOLOLOG_FILE __FILENAME__
#else
#define YOLOLOG_FILE __FILE__
#endif

#define YOLOLOG_FMT(fmt) "[" YOLOLOG_FILE ":" YOLOLOG_STR_HELP(__LINE__) "] " fmt

template <typename... Args>
inline void trace(const char* fmt, const Args&... args) {
  return yoloLogger::getInstance()->trace(fmt, args...);
}

#define YOLO_TRACE(fmt, ...) ::trace(YOLOLOG_FMT(fmt), ##__VA_ARGS__)

template <typename... Args>
inline void debug(const char* fmt, const Args&... args) {
  return yoloLogger::getInstance()->debug(fmt, args...);
}
#define YOLO_DEBUG(fmt, ...) ::debug(YOLOLOG_FMT(fmt), ##__VA_ARGS__)

template <typename... Args>
inline void info(const char* fmt, const Args&... args) {
  return yoloLogger::getInstance()->info(fmt, args...);
}
#define YOLO_INFO(fmt, ...) ::info(YOLOLOG_FMT(fmt), ##__VA_ARGS__)

template <typename... Args>
inline void warn(const char* fmt, const Args&... args) {
  return yoloLogger::getInstance()->warn(fmt, args...);
}
#define YOLO_WARN(fmt, ...) ::warn(YOLOLOG_FMT(fmt), ##__VA_ARGS__)

template <typename... Args>
inline void error(const char* fmt, const Args&... args) {
  return yoloLogger::getInstance()->error(fmt, args...);
}
#define YOLO_ERROR(fmt, ...) ::error(YOLOLOG_FMT(fmt), ##__VA_ARGS__)

template <typename... Args>
inline void critical(const char* fmt, const Args&... args) {
  return yoloLogger::getInstance()->critical(fmt, args...);
}
#define YOLO_CRITICAL(fmt, ...) ::critical(YOLOLOG_FMT(fmt), ##__VA_ARGS__)


inline void logInit(const std::string& level = "info", const std::string& fileLogPath="") {
    auto logger = yoloLogger::getInstance(level, fileLogPath);
}