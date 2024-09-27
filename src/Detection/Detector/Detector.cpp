#include "Detector.h"

#include <opencv2/highgui.hpp>
#include "../Model/SSDModel.h"
#include "../Video/Video.h"

Detector::Detector(const std::string& video_source, Video::SourceType source_type)
{
    this->model = std::make_unique<SSDModel>();
    this->video = std::make_unique<Video>(video_source, source_type, model->getClassNumber());
}

Detector::~Detector() = default;

void Detector::process()
{
    cv::Mat frame = video->getNextFrame();
    if (!frame.empty()) {
        processed_frame = detect(frame);
    } else {
        processed_frame = cv::Mat();
    }
}

cv::Mat Detector::detect(cv::Mat frame)
{
    model->detectObjects(frame, class_ids, class_names, confidences, boxes);
    video->drawDetectionResults(frame, class_ids, class_names, confidences, boxes);
    return frame;
}

cv::Mat Detector::getProcessedFrame() const
{
    return processed_frame;
}

std::vector<int> Detector::getClassIds() const
{
    return class_ids;
}

std::vector<std::string> Detector::getClassNames() const
{
    return class_names;
}

std::vector<float> Detector::getConfidences() const
{
    return confidences;
}

std::vector<cv::Rect> Detector::getBoxes() const
{
    return boxes;
}

cv::Size Detector::getWindowSize() const
{
    return video->getWindowSize();
}