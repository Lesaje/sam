#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

class Video {
public:
    Video(const std::string& video_file, int class_num);
    ~Video();

    cv::Mat getNextFrame();
    void drawDetectionResults(cv::Mat& frame,
                              const std::vector<int>& classIds,
                              const std::vector<std::string>& classNames,
                              const std::vector<float>& confidences,
                              const std::vector<cv::Rect>& boxes);

    float getFps() const;
    int getDetectFreq() const;
    cv::Size getWindowSize() const;

private:
    std::string image_path;
    cv::VideoCapture cap;
    float _fps;
    int _detect_freq;
    cv::Size window_size;
    std::vector<cv::Scalar> class_color;

    cv::Size resizedSize(cv::Size orig);
    void setClassColor(int class_num);
};

#endif // VIDEO_H