#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

class Video {
public:
    enum class SourceType {
        FILE,
        WEBCAM
    };

    Video(const std::string& source, SourceType type, int class_num);
    ~Video();

    cv::Mat getNextFrame();

    void drawDetectionResults(cv::Mat& frame,
                              const std::vector<int>& classIds,
                              const std::vector<std::string>& classNames,
                              const std::vector<float>& confidences,
                              const std::vector<cv::Rect>& boxes);

    cv::Size getWindowSize() const;

private:
    SourceType source_type;
    std::string source_path;
    cv::VideoCapture cap;
    cv::Size window_size;

    cv::Size resizedSize(cv::Size orig);
    void initializeCapture();
};

#endif // VIDEO_H