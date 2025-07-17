#include <queue>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

#include "yolo_common.h"


template <typename T>
class SafeQueue {
    private:
        std::queue<T> m_queue;
        mutable std::mutex m_mutex;
        std::condition_variable m_empty_lock;
        std::condition_variable m_full_lock;
        int m_max_size; // -1 means no limit
        bool m_stop;
        
        // no copying or moving allowed
        SafeQueue(const SafeQueue &) = delete;
        SafeQueue(SafeQueue &&) = delete;

    public:
        SafeQueue(const int max_size = -1): m_max_size(max_size), m_stop(false) {
        
        };

        ~SafeQueue(){
            stop();
        };

        stateType enqueue(const T & item, const bool block = true) {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_max_size > 0 && m_queue.size() >= m_max_size) {
                if (!block) {
                    return stateType::QUEUE_FULL;
                }
                m_full_lock.wait(lock, [this] { return m_queue.size() < m_max_size || m_stop; });
            }
            if (m_stop) {
                return stateType::QUEUE_STOPPED;
            }
            m_queue.push(item);
            lock.unlock();
            m_empty_lock.notify_one();
            return stateType::SUCCESS;
        }

        stateType dequeue(T & item, const bool block = true) {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_queue.empty()) {
                if (!block) {
                    return stateType::QUEUE_EMPTY;
                }
                m_empty_lock.wait(lock, [this] { return !m_queue.empty() || m_stop; });
            }
            if (m_stop) {
                return stateType::QUEUE_STOPPED;
            }
            item = m_queue.front();
            m_queue.pop();
            lock.unlock();
            m_full_lock.notify_one();
            return stateType::SUCCESS;
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.empty();
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.size();
        }

        void clear() {
            std::lock_guard<std::mutex> lock(m_mutex);
            while (!m_queue.empty()) {
                m_queue.pop();
            }
            m_empty_lock.notify_all();
            m_full_lock.notify_all();
        }

        int max_size() const {
            return m_max_size;
        }

        void set_max_size(int max_size) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_max_size = max_size;
            m_empty_lock.notify_all();
            m_full_lock.notify_all();
        }

        void stop() {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
            m_empty_lock.notify_all();
            m_full_lock.notify_all();
        }
     
};

template <typename T>
class SafeUnorderedMap {
    private:
        std::unordered_map<int, T> m_map;
        mutable std::mutex m_mutex;
        std::condition_variable m_empty_lock;
        std::condition_variable m_full_lock;
        int m_max_size;
        bool m_stop;
        
        // no copying or moving allowed
        SafeUnorderedMap(const SafeUnorderedMap &) = delete;
        SafeUnorderedMap(SafeUnorderedMap &&) = delete;
        
    public:
        SafeUnorderedMap(const int max_size = -1): m_max_size(max_size), m_stop(false) {};
        ~SafeUnorderedMap() {};

        stateType insert(int key, const T & value, const bool block = true) {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_max_size > 0 && m_map.size() >= m_max_size) {
                if (!block) {
                    return stateType::UNORDERMAP_FULL;
                }
                m_full_lock.wait(lock, [this] { return m_map.size() < m_max_size || m_stop; });
            }
            if (m_stop) {
                return stateType::UNORDERMAP_STOPPED;
            }
            m_map[key] = value;
            lock.unlock();
            m_empty_lock.notify_one();
            return stateType::SUCCESS;
        }

        stateType get(int key, T & value, const bool block = true) {
            std::unique_lock<std::mutex> lock(m_mutex);
    
            if (!block) {
                auto it = m_map.find(key);
                if (it == m_map.end()) {
                    return stateType::UNORDERMAP_EMPTY;
                }
                value = std::move(it->second);
                m_map.erase(it);
                m_full_lock.notify_one();
                return stateType::SUCCESS;
            }

            // 阻塞等待直到 key 出现或 m_stop
            m_empty_lock.wait(lock, [this, &key] {
                return m_stop || m_map.find(key) != m_map.end();
            });

            if (m_stop) {
                return stateType::UNORDERMAP_STOPPED;
            }

            auto it = m_map.find(key);
            if (it == m_map.end()) {
                return stateType::UNORDERMAP_EMPTY;  // 理论上不会发生，但保险处理
            }

            value = std::move(it->second);
            m_map.erase(it);
            m_full_lock.notify_one();
            return stateType::SUCCESS;
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_map.empty();
        }
        
        size_t size() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_map.size();
        }

        void clear() {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_map.clear();
            m_empty_lock.notify_all();
            m_full_lock.notify_all();
        }

        int max_size() const {
            return m_max_size;
        }

        void set_max_size(int max_size) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_max_size = max_size;
            m_empty_lock.notify_all();
            m_full_lock.notify_all();
        }
        
        void stop() {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
            m_empty_lock.notify_all();
            m_full_lock.notify_all();
        }

};