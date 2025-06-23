#include <filesystem>

#include "detect_factory.h"
#include "utils.h"
#include "argparse/argparse.hpp"

#include "opencv2/opencv.hpp"

int main(int argc, char** argv) {
    argparse::ArgumentParser detect_parser("detect");
    detect_parser.add_argument("-m", "--model")
        .required()
        .help("Path to your model file")
        .default_value(std::string("./yolov5s_v6.1_3output_f32.engine"));

    detect_parser.add_argument("-i", "--input")
        .required()
        .help("Path to your input file Path. Support jpg, vide0(h264/h265), rtsp/rtmp, video_device")
        .default_value(std::string("./test.jpg"));
    
    detect_parser.add_argument("-d", "--deviceId")
        .help("Device id, default is 0")
        .default_value(0)
        .scan<'d', int>();
    
    detect_parser.add_argument("-o", "--outputDir")
        .help("Path to your output picture directory with dectection result. Default is `./detect_result`")
        .default_value("./detect_result/");
    
    // parse the command line arguments
    detect_parser.parse_args(argc, argv);

    std::string modelPath = detect_parser.get<std::string>("--model");
    isFileExist(modelPath);
    
    std::string inputFile= detect_parser.get<std::string>("--input");
    auto inputType = classifyInput(inputFile);
    if (inputType == INPUTTYPE::UNKNOW) {
        std::cerr << "Input type is not supported!" << std::endl;
        return -1;
    }else if (inputType == INPUTTYPE::JPG_IMAGE || inputType == INPUTTYPE::VIDEO ) {
        isFileExist(inputFile);
    }

    int devId = detect_parser.get<int>("--deviceId");
    std::string outputDir = detect_parser.get<std::string>("--outputDir");

    // init dectect function
    std::shared_ptr<detect_factory> factory = std::make_shared<trt_detect_factory>();
    std::shared_ptr<detect> detect = factory->getInstance(modelPath, yoloType::YOLOV5, 0);



    std::vector<nlohmann::ordered_json> resBoxJsonVec;
    switch (inputType) {
        case INPUTTYPE::JPG_IMAGE: {

            auto img = cv::imread(inputFile);
            auto resBox = detect->process(&img, 1);
            resBoxJsonVec.push_back(box2json(inputFile, resBox[0])) ;

            std::string outputName = inputFile.substr(inputFile.find_last_of("/") + 1);
            drawBox(resBox[0], img, outputName, outputDir);
            break;
        }
            

        case INPUTTYPE::IMAGE_DIR: {
            std::vector<std::string> jpgFiles = getJpgFiles(inputFile);
            int batch_size = detect->getAlgorithmInfo().batch_size;
            int num = jpgFiles.size();
            int calculateTime = (num-1) / batch_size + 1;

            for(int i = 0; i < calculateTime; i++){
                int inputNum = std::min(num - i * batch_size, batch_size);
                std::vector<cv::Mat> images(inputNum);
                for(int j = 0; j < inputNum; j++){
                    images[j] = cv::imread(jpgFiles[i*batch_size+j]);
                }
                auto resBox = detect->process(images.data(), inputNum);
                for(int j = 0; j < inputNum; j++){
                    resBoxJsonVec.push_back(box2json(jpgFiles[i*batch_size+j], resBox[j]));
                    std::string outputName = jpgFiles[i*batch_size+j].substr(jpgFiles[i*batch_size+j].find_last_of("/") + 1);
                    drawBox(resBox[j], images[j], outputName, outputDir);
                }
                images.clear();
            }
            break;

        }
            

        case INPUTTYPE::VIDEO: {
            cv::VideoCapture cap(inputFile);
            if (!cap.isOpened()) {
                std::cerr << "Cannot open video file!" << std::endl;
                return -1;
            }

            auto batch_size = detect->getAlgorithmInfo().batch_size;
            int count = 0;
            int frame_id = 0;
            cv::Mat img;
            std::vector<cv::Mat> imgVec;
            while (auto ret = cap.read(img) || count)
            {
                if(ret) {
                    imgVec.push_back(img);
                    count++;
                    std::cout << "frame_id: " << frame_id << std::endl;
                }
                if (imgVec.size() < batch_size && ret) continue;
                
                
                auto resBox = detect->process(imgVec.data(), count);
                for (int i = 0; i < count; ++i) {
                    resBoxJsonVec.push_back(box2json(inputFile, resBox[i]));
                    // debug
                    std::string outputName = getFileName(inputFile) + "_" + std::to_string(frame_id) + ".jpg";
                    drawBox(resBox[i], imgVec[i], outputName, outputDir);
                    frame_id++;
                }
                count = 0;
                imgVec.clear();
            }
            break;
        }
            
        
        default:
            YOLO_ERROR("Input type is not supported!");
            return -1;
            
    }

    // dumpData
    jsonDump("./detectRes.json",resBoxJsonVec);
    return 0;
}