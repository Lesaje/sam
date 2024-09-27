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
    std::string video_file = "/home/kuver/Downloads/test_videos/slow.mp4";
    bool use_webcam = true;
    std::unique_ptr<SSDModel> model;
    std::unique_ptr<Video> video;
};

#endif // DETECTION_H