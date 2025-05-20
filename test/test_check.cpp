#include "yolo_common.h"

int main() {
    std::cout << "YOLO_CHECK(true, \"1. input is true. nothing will be printed\")" <<std::endl;
    YOLO_CHECK(true, "1. input is true. nothing will be printed");
    
    std::cout << "YOLO_CHECK(stateType::SUCCESS == stateType::SUCCESS, \"2. input is true. nothing will be printed\")" <<std::endl;
    YOLO_CHECK(stateType::SUCCESS == stateType::SUCCESS, "2. input is true. nothing will be printed");

    std::cout << "YOLO_CHECK(stateType::SUCCESS == stateType::ERROR, \"3. input is false. the program will exit!\")" <<std::endl;
    YOLO_CHECK(stateType::SUCCESS == stateType::ERROR, "3. input is false. the program will exit!");
}