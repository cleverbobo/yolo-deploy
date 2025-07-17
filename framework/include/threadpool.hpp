#include <thread>
#include "safe_queue.hpp"

template <typename T>
class ThreadPool {
private:
    std::vector<std::thread> m_threads;
    SafeQueue<std::function<void()>> m_inputQueue;
    SafeUnorderedMap<T> m_outputMap;
    int m_maxThreads;
    int m_maxInputQueueSize;
    int m_maxOutputQueueSize;

    std::mutex m_inputMutex;
    bool m_stop;

    class Worker {
        private:
            ThreadPool<T> *m_pool;
            int m_id;
        public:
            Worker(ThreadPool<T> *pool, int id) : m_pool(pool), m_id(id) {}
            
            void operator()() {
                while(!m_pool->m_stop) {
                    std::function<void()> task;
                    auto ret = m_pool->m_inputQueue.dequeue(task);

                    if (ret == stateType::QUEUE_STOPPED) {
                        YOLO_DEBUG("Worker {} stopped", m_id);
                        return;
                    }

                    YOLO_CHECK(ret == stateType::SUCCESS, "ThreadPool input queue dequeue failed");
                    task();                
                }
            }
    };

    // no copying or moving allowed
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool & operator=(const ThreadPool &) = delete;

public:
    ThreadPool(int maxThreads, int maxInputQueueSize = -1, int maxOutputQueueSize = -1) :
        m_maxThreads(maxThreads),
        m_maxInputQueueSize(maxInputQueueSize),
        m_maxOutputQueueSize(maxOutputQueueSize),
        m_stop(false),
        m_inputQueue(SafeQueue<std::function<void()>>(maxInputQueueSize)),
        m_outputMap(SafeUnorderedMap<T>(maxOutputQueueSize))
        {


        YOLO_CHECK(m_maxThreads > 0, "ThreadPool maxThreads must be greater than 0");
        for (int i = 0; i < m_maxThreads; ++i) {
            m_threads.emplace_back(Worker(this, i));
        }
    }

    void shutdown() {
        m_stop = true;
        m_inputQueue.stop();
        m_outputMap.stop();
        
        for (auto &thread : m_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        YOLO_DEBUG("ThreadPool shutdown complete");
    }

    template <typename Func, typename... Args>
    stateType enqueue(int id, Func &&func, Args &&...args) {

        if (m_stop) {
            return stateType::THREADPOOL_STOPPED;
        }

        auto task = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);

        std::function<void()> wrappedTask = [this, task, id]() {
            auto res = task();
            m_outputMap.insert(id,res);
        };

        auto ret = m_inputQueue.enqueue(wrappedTask);
        return ret;
    }

    stateType dequeue(int id, T &item, bool block = true) {
        if (m_stop) {
            return stateType::THREADPOOL_STOPPED;
        }

        return m_outputMap.get(id, item, block);
    }

    int outputSize() const {
        return m_outputMap.size();
    }


};