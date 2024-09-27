#include "Video.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <iomanip>
#include <iostream>

Video::Video(const std::string& source, SourceType type, int class_num)
    : source_type(type), source_path(source)
{
    initializeCapture();
}

Video::~Video() = default;

void Video::initializeCapture()
{
    if (source_type == SourceType::FILE)
    {
        cap.open(source_path);
        if (!cap.isOpened())
        {
            CV_Error(cv::Error::StsError, "Video file (" + source_path + ") cannot open.");
        }
    }
    else if (source_type == SourceType::WEBCAM)
    {
        cap.open(0); // Open default camera
        if (!cap.isOpened())
        {
            CV_Error(cv::Error::StsError, "Cannot open webcam.");
        }
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    float _fps = (float)cap.get(cv::CAP_PROP_FPS);
    window_size = resizedSize(cv::Size(width, height));

    std::cout << "Video source: " << (source_type == SourceType::FILE ? "File" : "Webcam") << std::endl
              << "- original width = " << width << std::endl
              << "- original height = " << height << std::endl
              << "- fps = " << _fps << std::endl;
}

cv::Mat Video::getNextFrame()
{
    cv::Mat frame;
    cap >> frame;
    return frame;
}

void Video::drawDetectionResults(cv::Mat& image,
                                 const std::vector<int>& classIds,
                                 const std::vector<std::string>& classNames,
                                 const std::vector<float>& confidences,
                                 const std::vector<cv::Rect>& boxes)
{
    for (size_t i = 0; i < classIds.size(); i++)
    {
        cv::rectangle(image, boxes[i], cv::Scalar(200, 100, 0), 2);

        std::ostringstream label_ss;
        label_ss << classNames[i] << ": " << std::fixed << std::setprecision(2) << confidences[i] * 100.0 << "%";
        std::string label = label_ss.str();

        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image, cv::Point(boxes[i].x, boxes[i].y - labelSize.height - 10),
                      cv::Point(boxes[i].x + labelSize.width, boxes[i].y),
                      cv::Scalar(0, 100, 200), cv::FILLED);
        cv::putText(image, label, cv::Point(boxes[i].x, boxes[i].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

cv::Size Video::getWindowSize() const
{
    return window_size;
}

cv::Size Video::resizedSize(cv::Size orig)
{
    int w = 900;
    int h = orig.height * w / orig.width;
    if (h > 1200)
    {
        h = 900;
        w = orig.width * h / orig.height;
    }
    return cv::Size(w, h);
}
