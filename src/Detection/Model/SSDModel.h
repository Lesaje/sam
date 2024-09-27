#ifndef SSDMODEL_H
#define SSDMODEL_H

#include "../DTO/DetectionDTO.h"
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class SSDModel
{
public:
    explicit SSDModel(DTO::ModelConfig config);
    ~SSDModel();

    std::vector<DTO::DetectionResult> detectObjects(const cv::Mat& image);
    int getClassNumber() const;

private:
    DTO::ModelConfig config;
    std::vector<std::string> classes;
    cv::dnn::Net net;

    void readClassFile();
    void loadModel();
    void setupNetwork();
};
#endif
