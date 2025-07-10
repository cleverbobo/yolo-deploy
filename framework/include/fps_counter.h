#include <chrono>

#include "log.h"
#include "yolo_common.h"

#include <mutex>

class fpsCounter : public NoCopyable {
    public:
        fpsCounter();
        fpsCounter(const std::string& name, const int condCnts = -1, const float condTimes = 10.0f);
        ~fpsCounter();

        void updateConfig(const std::string& name, const int condCnts, const float condTimes = 10.0f);
        void add(int count = 1);
        float getTempFps();
        float getAvgFps();
        void summary();
        
    private:
        float m_tempFps;
        float m_avgFps;
        int m_frameCount;
        std::mutex m_mutex;
        std::string m_name;
        std::chrono::time_point<std::chrono::steady_clock> m_startTime;

        // 最小的统计步数
        int m_condCnts;
        // 最小的统计时长，单位ms
        float m_condTimes = 10.0f;

        
};