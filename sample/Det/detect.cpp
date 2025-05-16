#include "detect_factory.h"

#include "ff_decode.h"
#include "utils.h"

int main(int argc, char** argv) {
    std::string modelPath = "./yolov5.bmodel";
    std::string img_file= "./test.jpg";

    std::shared_ptr<detect_factory> factory = std::make_shared<sophgo_detect_factory>();
    std::shared_ptr<detect> detect = factory->getInstance(modelPath, yoloType::YOLOV5, 0);


    decoder dec(0);
    bm_image img = dec.decodeJpg(img_file);
    auto resBox = detect->process(&img, 1);

    nlohmann::ordered_json resBoxJson;
    box2json(img_file, resBox[0], resBoxJson);
    jsonDump("./detectRes.json",resBoxJson);

    drawBox(resBox[0], img_file);
}