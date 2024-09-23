#ifndef SSDMODEL_H
#define SSDMODEL_H
#include <opencv2/dnn.hpp>
#include <queue>
#include <thread>

class SSDModel
{
public:
    SSDModel(float conf_threshold, float nms_threshold);
    ~SSDModel();

    void detectObjects(const cv::Mat& image,
                       std::vector<int>& classIds,
                       std::vector<std::string>& classNames,
                       std::vector<float>& confidences,
                       std::vector<cv::Rect>& boxes);

    int getClassNumber() const;

private:
    // Detection threshold
    float conf_threshold;
    float nms_threshold;

    // SSD MobileNet Model files
    const std::string class_file = "/home/kuver/Documents/SAM/cpp/resources/object_detection_classes_coco.txt";
    const std::string model_file = "/home/kuver/Documents/SAM/cpp/resources/frozen_inference_graph.pb";
    const std::string config_file = "/home/kuver/Documents/SAM/cpp/resources/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";;

    // Store the list of classe name
    std::vector<std::string> classes;
    // DNN model
    cv::dnn::Net net;
    // Information about detected objects

    void readClassFile();
    void loadModel();
    std::vector<int> detect(const cv::Mat &image, std::vector<int> &classIds,
                                    std::vector<float> &confidences,
                                    std::vector<cv::Rect> &boxes);
};


#endif