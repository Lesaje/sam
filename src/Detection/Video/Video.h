#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include "../DTO/DetectionDTO.h"
#include <opencv2/core/types.hpp>

class Video {
public:

    explicit Video(const DTO::VideoSource& source);
    ~Video();

    cv::Mat getNextFrame();
    void drawDetectionResults(const DTO::FrameData& frameData);

private:
    DTO::SourceType sourceType;
    std::string sourcePath;
    cv::VideoCapture cap;
    cv::Size windowSize;
};

#endif // VIDEO_H