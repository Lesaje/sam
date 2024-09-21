#include "Detection.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include "Model/SSDModel.h"
#include "Video/Video.h"

Detection::Detection()
{
    this->model = std::make_unique<SSDModel>(conf_threshold, nms_threshold);
    this->video =  std::make_unique<Video>(video_file, model->getClassNumber());
}

Detection::~Detection() = default;

void Detection::draw()
{
    const std::string window_name = "Object Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    cv::resizeWindow(window_name, this->getWindowSize());
    auto start = std::chrono::high_resolution_clock::now();

    while (true) {

        auto f = video->getNextFrame();
        if (f.empty() == true) {
            break;
        }

        cv::Mat result = this->detect(f);
        if (result.empty()) {
            break;
        }

        cv::imshow(window_name, result);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Frame processing time: " << frameTime << " ms" << std::endl;
}

cv::Mat Detection::detect(cv::Mat frame)
{
    if (!frame.empty()) {
        std::vector<int> classIds;
        std::vector<std::string> classNames;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        model->detectObjects(frame, classIds, classNames, confidences, boxes);
        video->drawDetectionResults(frame, classIds, classNames, confidences, boxes);
    }
    return frame;
}

float Detection::getFps() const
{
    return video->getFps();
}

cv::Size Detection::getWindowSize() const
{
    return video->getWindowSize();
}