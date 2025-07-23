#include <filesystem>

#include "detect_factory.h"

#include "ff_decode.h"
#include "utils.h"

#include "argparse/argparse.hpp"

int main(int argc, char** argv) {
    // init logger
    // logInit("info");
    argparse::ArgumentParser detect_parser("detect");
    detect_parser.add_argument("-m", "--model")
        .required()
        .help("Path to your model file")
        .default_value(std::string("./yolov5.bmodel"));
    
    detect_parser.add_argument("-t", "--yoloType")
        .required()
        .help("Type of YOLO model, default is YOLOV5")
        .default_value(std::string("YOLOV5"));

    detect_parser.add_argument("-i", "--input")
        .required()
        .help("Path to your input file Path. Support jpg, vide0(h264/h265), rtsp/rtmp, video_device")
        .default_value(std::string("./test.jpg"));
    
    detect_parser.add_argument("-d", "--deviceId")
        .help("Device id, default is 0")
        .default_value(0)
        .scan<'d', int>();
    
    detect_parser.add_argument("-log", "--logLevel")
        .help("Set log level, default is info")
        .default_value(std::string("info"))
        .choices("debug", "info", "warning", "error", "critical");
    
    detect_parser.add_argument("-o", "--outputDir")
        .help("Path to your output picture directory with dectection result. Default is `./detect_result`")
        .default_value("./detect_result/");
    
    // parse the command line arguments
    detect_parser.parse_args(argc, argv);

    // set log level
    std::string logLevel = detect_parser.get<std::string>("--logLevel");
    logInit(logLevel);

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
    std::shared_ptr<detect_factory> factory = std::make_shared<sophgo_detect_factory>();

    yoloType detectYoloType = enumYoloType(detect_parser.get<std::string>("--yoloType"));
    YOLO_CHECK(detectYoloType != yoloType::UNKNOWN, "Unmatch yolo type: " + detect_parser.get<std::string>("--yoloType"));
    std::shared_ptr<detect> detect = factory->getInstance(modelPath, detectYoloType, 0);

    
    decoder dec(0);
    std::vector<nlohmann::ordered_json> resBoxJsonVec;
    switch (inputType) {
        case INPUTTYPE::JPG_IMAGE: {

            bm_image img = dec.decodeJpg(inputFile);
            auto resBox = detect->process(&img, 1);
            resBoxJsonVec.push_back(box2json(inputFile, resBox[0])) ;

            cv::Mat imgMat;
            cv::bmcv::toMAT(&img, imgMat);
            std::string outputName = inputFile.substr(inputFile.find_last_of("/") + 1);
            drawBox(resBox[0], imgMat, outputName, outputDir);
            bm_image_destroy(img);
            break;
        }
            

        case INPUTTYPE::IMAGE_DIR: {
            std::vector<std::string> jpgFiles = getJpgFiles(inputFile);
            int batch_size = detect->getAlgorithmInfo().batch_size;
            int num = jpgFiles.size();
            int calculateTime = (num-1) / batch_size + 1;

            for(int i = 0; i < calculateTime; i++){
                int inputNum = std::min(num - i * batch_size, batch_size);
                std::vector<bm_image> images(inputNum);
                for(int j = 0; j < inputNum; j++){
                    images[j] = dec.decodeJpg(jpgFiles[i*batch_size+j]);
                }
                auto resBox = detect->process(images.data(), inputNum);
                for(int j = 0; j < inputNum; j++){
                    resBoxJsonVec.push_back(box2json(jpgFiles[i*batch_size+j], resBox[j]));

                    cv::Mat imgMat;
                    cv::bmcv::toMAT(&images[j], imgMat);
                    std::string outputName = jpgFiles[i*batch_size+j].substr(jpgFiles[i*batch_size+j].find_last_of("/") + 1);
                    drawBox(resBox[j], imgMat, outputName, outputDir);
                }
                for (auto& image : images) {
                    auto ret = bm_image_destroy(image);
                }
                images.clear();
            }
            break;

        }
            

        case INPUTTYPE::VIDEO: {
            dec.decodeVideo(inputFile);
            bm_image img;
            auto batch_size = detect->getAlgorithmInfo().batch_size;
            int count = 0;
            int frame_id = 0;
            std::vector<bm_image> imgVec;
            while (auto ret = !dec.decodeVideoGrab(img) || count)
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
                    cv::Mat imgMat;
                    cv::bmcv::toMAT(&imgVec[i], imgMat);
                    // debug
                    std::string outputName = getFileName(inputFile) + "_" + std::to_string(frame_id) + ".jpg";
                    drawBox(resBox[i], imgMat, outputName, outputDir);
                    frame_id++;
                }
                count = 0;
                for (auto& img : imgVec) {
                    auto ret = bm_image_destroy(img);
                }
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