#include "Video.h"
#include <opencv2/imgproc.hpp>
#include <random>
#include <iomanip>
#include <iostream>

Video::Video(const DTO::VideoSource& source) : sourceType(source.type), sourcePath(source.path)
{
    if (sourceType == DTO::SourceType::FILE)
    {
        cap.open(sourcePath);
        if (!cap.isOpened())
        {
            CV_Error(cv::Error::StsError, "Video file (" + sourcePath + ") cannot open.");
        }
    }
    else if (sourceType == DTO::SourceType::WEBCAM)
    {
        cap.open(0); // Open default camera
        if (!cap.isOpened())
        {
            CV_Error(cv::Error::StsError, "Cannot open webcam.");
        }
    }
}

Video::~Video() = default;

cv::Mat Video::getNextFrame()
{
    cv::Mat frame;
    cap >> frame;
    return frame;
}

void Video::drawDetectionResults(const DTO::FrameData& frameData)
{
    for (const auto& detection : frameData.detections)
    {
        cv::rectangle(frameData.frame, detection.box, cv::Scalar(200, 100, 0), 2);

        std::ostringstream labelSs;
        labelSs << detection.className << ": " << std::fixed << std::setprecision(2) << detection.confidence * 100.0 << "%";
        std::string label = labelSs.str();

        int baseline;
        cv::Size const labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frameData.frame, cv::Point(detection.box.x, detection.box.y - labelSize.height - 10),
                      cv::Point(detection.box.x + labelSize.width, detection.box.y),
                      cv::Scalar(0, 100, 200), cv::FILLED);
        cv::putText(frameData.frame, label, cv::Point(detection.box.x, detection.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}
