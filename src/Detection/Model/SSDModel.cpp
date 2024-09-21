#include "SSDModel.h"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

SSDModel::SSDModel(float conf_threshold, float nms_threshold)
    : conf_threshold(conf_threshold), nms_threshold(nms_threshold)
{
    readClassFile();
    loadModel();
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
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(), true, false);
    net.setInput(blob);

    cv::Mat output = net.forward();

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
    std::ifstream ifs(class_file);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "Class File not found.");

    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

void SSDModel::loadModel()
{
    net = cv::dnn::readNetFromTensorflow(model_file,
                                         config_file);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);



    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::string outLayerType = net.getLayer(outLayers[0])->type;
    if (outLayerType != "DetectionOutput")
       CV_Error(cv::Error::StsNotImplemented, "Unexpected output layer type: " + outLayerType);
}