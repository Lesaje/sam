#include "Video.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <iomanip>
#include <iostream>

Video::Video(const std::string& video_file, int class_num) : image_path(video_file), cap(video_file)
{
    setClassColor(class_num);

    if (!cap.isOpened())
    {
        CV_Error(cv::Error::StsError, "Video file (" + image_path + ") cannot open.");
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    float _fps = (float)cap.get(cv::CAP_PROP_FPS);
    window_size = resizedSize(cv::Size(width, height));

    std::cout << "Video file: " << image_path << std::endl
              << "- original width = " << width << std::endl
              << "- original height = " << height << std::endl
              << "- fps = " << _fps << std::endl;
}

Video::~Video() = default;

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
        cv::rectangle(image, boxes[i], class_color[classIds[i]], 2);

        std::ostringstream label_ss;
        label_ss << classNames[i] << ": " << std::fixed << std::setprecision(2) << confidences[i] * 100.0 << "%";
        std::string label = label_ss.str();

        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image, cv::Point(boxes[i].x, boxes[i].y - labelSize.height - 10),
                      cv::Point(boxes[i].x + labelSize.width, boxes[i].y),
                      class_color[classIds[i]], cv::FILLED);
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

void Video::setClassColor(int class_num)
{
    std::mt19937 random_engine(2019);
    std::uniform_int_distribution<int> distribution(0, 255);

    for (int i = 0; i < class_num; ++i)
    {
        cv::Scalar color = cv::Scalar(distribution(random_engine),
                                      distribution(random_engine),
                                      distribution(random_engine));
        class_color.push_back(color);
    }
}