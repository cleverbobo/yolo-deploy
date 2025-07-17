#include "threadpool.hpp"


int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int main() {
    auto thread_pool = ThreadPool<int>(4);
    std::string logLevel = "debug";
    logInit(logLevel);
    
    for (int i = 0; i < 10; ++i) {
        if (i % 2 == 0) {
            thread_pool.enqueue(i, add, 1, 1);
        } else {
            thread_pool.enqueue(i, subtract, 1, 1);
        }
    }

    for (int i = 0; i < 10; ++i) {
        int result;
        auto ret = thread_pool.dequeue(i, result, true);
        if (ret == stateType::SUCCESS) {
            std::cout << "idx "<< i << " || Result: " << result << std::endl;
        } else {
            std::cout << "Failed to dequeue result" << std::endl;
        }
    }

    thread_pool.shutdown();
    
    return 0;   
}