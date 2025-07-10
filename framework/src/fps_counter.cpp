#include "fps_counter.h"

fpsCounter::fpsCounter() : m_tempFps(0), m_avgFps(0), m_frameCount(0), m_condCnts(-1) {
    m_startTime = std::chrono::steady_clock::now();
    m_name = "default_fps_counter";
}

fpsCounter::fpsCounter(const std::string& name, const int condCnts, const float condTimes)
    : m_tempFps(0), m_avgFps(0), m_frameCount(0), m_condCnts(condCnts), m_condTimes(condTimes) {
    m_startTime = std::chrono::steady_clock::now();
    m_name = name;
}

fpsCounter::~fpsCounter() {
    // Destructor logic if needed
}

void fpsCounter::updateConfig(const std::string& name, const int condCnts, const float condTimes) {
    m_name = name;
    m_condCnts = condCnts;
    m_condTimes = condTimes;
    m_startTime = std::chrono::steady_clock::now();
}

void fpsCounter::add(int count) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frameCount += count;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(now - m_startTime).count();
    if ( (m_condCnts > 0 && m_frameCount > m_condCnts) ||
         (m_condTimes > 0 && elapsed > m_condTimes) ) {
        // Calculate FPS
        m_tempFps = (m_frameCount * 1000.0f) / elapsed;
        if (m_avgFps == 0) {
            m_avgFps = m_tempFps; // Initialize avgFps on first calculation
        } else {
            m_avgFps = 0.7 * m_avgFps + 0.3 * m_tempFps; 
        }
        

        // reset counters
        m_frameCount = 0;
        m_startTime = now;

        YOLO_INFO("FPS Counter [{}]: Temp FPS: {:.2f}, Avg FPS: {:.2f}",m_name, m_tempFps, m_avgFps);
    }
}

float fpsCounter::getTempFps() {
    return m_tempFps;
}

float fpsCounter::getAvgFps() {
    return m_avgFps;
}

void fpsCounter::summary() {
    std::lock_guard<std::mutex> lock(m_mutex);
    // flush the current statistics
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(now - m_startTime).count();
    
    if (elapsed > 0) {
        m_tempFps = (m_frameCount * 1000.0f) / elapsed;
        if (m_avgFps == 0) {
            m_avgFps = m_tempFps; // Initialize avgFps on first calculation
        } else {
            m_avgFps = 0.7 * m_avgFps + 0.3 * m_tempFps; 
        }
    }

    YOLO_INFO("FPS Counter [{}]: Temp FPS: {:.2f}, Final Avg FPS: {:.2f}", m_name, m_tempFps, m_avgFps);
    m_frameCount = 0;
    m_startTime = now; // Reset start time for next summary
}