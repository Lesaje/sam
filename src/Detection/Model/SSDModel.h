#ifndef SSDMODEL_H
#define SSDMODEL_H
#include <opencv2/dnn.hpp>
#include <queue>
#include <thread>

class SSDModel
{
public:
    SSDModel(const std::string& model_path,
             const std::string& class_file_path,
             float conf_threshold = 0.5,
             float nms_threshold = 0.4);
    ~SSDModel();

    void detectObjects(const cv::Mat& image,
                       std::vector<int>& classIds,
                       std::vector<std::string>& classNames,
                       std::vector<float>& confidences,
                       std::vector<cv::Rect>& boxes);

    int getClassNumber() const;

private:
    // Detection thresholds
    float conf_threshold;
    float nms_threshold;

    // Model and class file paths
    std::string model_path;
    std::string class_file_path;

    // Store the list of class names
    std::vector<std::string> classes;

    // DNN model
    cv::dnn::Net net;

    void readClassFile();
    void loadModelFromONNX();
    void loadModelFromTf();
    std::vector<int> detect(const cv::Mat &image,
                            std::vector<int> &classIds,
                            std::vector<float> &confidences,
                            std::vector<cv::Rect> &boxes);

    // New method for setting up network parameters
    void setupNetwork();
};


#endif