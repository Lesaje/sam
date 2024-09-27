#ifndef DETECTION_H
#define DETECTION_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include "../Model/SSDModel.h"
#include "../Video/Video.h"

class Detector {
public:
    Detector(const std::string& video_source, Video::SourceType source_type);
    ~Detector();

    void process();
    cv::Mat getProcessedFrame() const;
    std::vector<int> getClassIds() const;
    std::vector<std::string> getClassNames() const;
    std::vector<float> getConfidences() const;
    std::vector<cv::Rect> getBoxes() const;
    cv::Size getWindowSize() const;

private:
    cv::Mat detect(cv::Mat frame);
    std::unique_ptr<SSDModel> model;
    std::unique_ptr<Video> video;
    cv::Mat processed_frame;
    std::vector<int> class_ids;
    std::vector<std::string> class_names;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
};

#endif // DETECTION_H