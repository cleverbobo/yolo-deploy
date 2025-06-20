#include "log.h"


yoloLogger::yoloLogger(std::string level, std::string fileLogPath) {
    auto logLevel = spdlog::level::from_str(level);
    if (logLevel == spdlog::level::off) {
        std::cout << "Invalid log level: " << level << std::endl;
        std::cout << "Defaulting to info level." << std::endl;
        logLevel = spdlog::level::info;  // 默认设为 info
    }
    std::vector<spdlog::sink_ptr> sinks;

    // console
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(console_sink);

    // file
    if ( !fileLogPath.empty() ) {
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(fileLogPath, 1024 * 1024 * 10, 5);
        sinks.push_back(file_sink);
    }
    
    // logger
    m_logger = std::make_shared<spdlog::logger>(loggerName,sinks.begin(), sinks.end());
    m_logger->set_level(logLevel);
    m_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$]  %v");

    spdlog::register_logger(m_logger);
}

std::shared_ptr<spdlog::logger> yoloLogger::getInstance(std::string level,std::string fileLogPath){
    static std::shared_ptr<yoloLogger> loggerInstance = std::shared_ptr<yoloLogger>(new yoloLogger(level, fileLogPath));
    return loggerInstance->m_logger;
}

void yoloLogger::setLevel(std::string level){
    auto logLevel = spdlog::level::from_str(level);
    if (logLevel == spdlog::level::off) {
        std::cout << "Invalid log level: " << level << std::endl;
    }
    if (m_logger) {
        m_logger->set_level(logLevel);
    } else {
        std::cout << "Logger not initialized." << std::endl;
    }
}

yoloLogger::~yoloLogger() {
    spdlog::shutdown();
}
