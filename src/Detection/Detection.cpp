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
    auto frame = 0;

    while (true) {

        auto f = video->getNextFrame();
        if (f.empty() == true) {
            break;
        }


        if (frame % 1 == 0) {
            cv::Mat result = this->detect(f);
            if (result.empty()) {
                break;
            }
            cv::imshow(window_name, result);
        } else {
            cv::imshow(window_name, f);
        }

        if (cv::waitKey(1) >= 0) {
            break;
        }

        frame += 1;
    }

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

cv::Size Detection::getWindowSize() const
{
    return video->getWindowSize();
}