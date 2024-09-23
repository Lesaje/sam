#ifndef DETECTION_H
#define DETECTION_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include "Model/SSDModel.h"
#include "Video/Video.h"

class Detection {
public:
    Detection();
    ~Detection();

    void draw();
    cv::Size getWindowSize() const;

private:
    cv::Mat detect(cv::Mat frame);
    float conf_threshold = 0.5;
    float nms_threshold = 0.5;
    std::string video_file = "/home/kuver/Downloads/test_videos/walking.mp4";
    std::unique_ptr<SSDModel> model;
    std::unique_ptr<Video> video;
};

#endif // DETECTION_H