#ifndef SSDMODEL_H
#define SSDMODEL_H
#include <opencv2/dnn.hpp>
#include <queue>
#include <thread>

class SSDModel
{
public:
    SSDModel(float conf_threshold = 0.6, float nms_threshold = 0.9);
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
    std::string model_path = "../resources/frozen_inference_graph.pb";
    std::string config = "../resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
    std::string class_file_path = "../resources/object_detection_classes_coco.txt";

    // Store the list of class names
    std::vector<std::string> classes;

    // DNN model
    cv::dnn::Net net;

    void readClassFile();
    void loadModel();
    std::vector<int> detect(const cv::Mat &image,
                            std::vector<int> &classIds,
                            std::vector<float> &confidences,
                            std::vector<cv::Rect> &boxes);

    // New method for setting up network parameters
    void setupNetwork();
};


#endif