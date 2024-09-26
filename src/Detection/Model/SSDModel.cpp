#include "SSDModel.h"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

SSDModel::SSDModel(const std::string& model_path,
                   const std::string& class_file_path,
                   float conf_threshold,
                   float nms_threshold)
    : conf_threshold(conf_threshold),
      nms_threshold(nms_threshold),
      model_path(model_path),
      class_file_path(class_file_path)
{
    loadModelFromTf();
    readClassFile();
}

SSDModel::~SSDModel() = default;

void SSDModel::detectObjects(const cv::Mat& image,
                             std::vector<int>& classIds,
                             std::vector<std::string>& classNames,
                             std::vector<float>& confidences,
                             std::vector<cv::Rect>& boxes)
{
    std::vector<int> indices = detect(image, classIds, confidences, boxes);

    std::vector<int> filteredClassIds;
    std::vector<std::string> filteredClassNames;
    std::vector<float> filteredConfidences;
    std::vector<cv::Rect> filteredBoxes;

    for (int index : indices) {
        filteredClassIds.push_back(classIds[index]);
        filteredClassNames.push_back(classes[classIds[index]]);
        filteredConfidences.push_back(confidences[index]);
        filteredBoxes.push_back(boxes[index]);
    }

    classIds = std::move(filteredClassIds);
    classNames = std::move(filteredClassNames);
    confidences = std::move(filteredConfidences);
    boxes = std::move(filteredBoxes);
}

int SSDModel::getClassNumber() const
{
    return classes.size();
}

std::vector<int> SSDModel::detect(const cv::Mat &image,
                                  std::vector<int> &classIds,
                                  std::vector<float> &confidences,
                                  std::vector<cv::Rect> &boxes)
{
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);
    net.setInput(blob);

    //auto start = std::chrono::high_resolution_clock::now();

    cv::Mat output = net.forward();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> duration = end - start;
    //std::cout << "Detection time: " << duration.count() << " ms" << std::endl;

    cv::Mat detections(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    for (int i = 0; i < detections.rows; ++i)
    {
        float confidence = detections.at<float>(i, 2);

        if (confidence > conf_threshold)
        {
            int classId = static_cast<int>(detections.at<float>(i, 1)) - 1;
            int left = static_cast<int>(detections.at<float>(i, 3) * image.cols);
            int top = static_cast<int>(detections.at<float>(i, 4) * image.rows);
            int right = static_cast<int>(detections.at<float>(i, 5) * image.cols);
            int bottom = static_cast<int>(detections.at<float>(i, 6) * image.rows);

            classIds.push_back(classId);
            confidences.push_back(confidence);
            boxes.push_back(cv::Rect(left, top, right - left + 1, bottom - top + 1));
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    return indices;
}

void SSDModel::readClassFile()
{
    std::ifstream ifs(class_file_path);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "Class File not found: " + class_file_path);

    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

void SSDModel::loadModelFromONNX()
{
    std::cout << "Before load" << std::endl;
    net = cv::dnn::readNetFromONNX(model_path);
    std::cout << "After load" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void SSDModel::loadModelFromTf()
{
    net = cv::dnn::readNetFromTensorflow("../resources/frozen_inference_graph.pb",
                                         "../resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt");
    setupNetwork();
}

void SSDModel::setupNetwork()
{
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::string outLayerType = net.getLayer(outLayers[0])->type;
    if (outLayerType != "DetectionOutput")
        CV_Error(cv::Error::StsNotImplemented, "Unexpected output layer type: " + outLayerType);
}
