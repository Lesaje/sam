#ifndef DETECTION_H
#define DETECTION_H

#include "../DTO/DetectionDTO.h"
#include "../Video/Video.h"
#include "../Model/SSDModel.h"
#include <memory>

class Detector {
public:
    explicit Detector(const DTO::VideoSource& videoSource);
    ~Detector();

    void process();
    [[nodiscard]] DTO::FrameData getProcessedFrame() const;

private:
    DTO::FrameData detect(const cv::Mat& frame);
    std::unique_ptr<SSDModel> model;
    std::unique_ptr<Video> video;
    DTO::FrameData processedFrame;
};


#endif // DETECTION_H