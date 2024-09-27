#include "SSDModel.h"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <utility>

SSDModel::SSDModel(DTO::ModelConfig config) : config(std::move(config))
{
    loadModel();
    readClassFile();
}

SSDModel::~SSDModel() = default;

std::vector<DTO::DetectionResult> SSDModel::detectObjects(const cv::Mat& image)
{
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);
    net.setInput(blob);

    cv::Mat output = net.forward();

    cv::Mat detections(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < detections.rows; ++i)
    {
        float confidence = detections.at<float>(i, 2);

        if (confidence > config.confidenceThreshold)
        {
            int classId = static_cast<int>(detections.at<float>(i, 1)) - 1;
            int left = static_cast<int>(detections.at<float>(i, 3) * image.cols);
            int top = static_cast<int>(detections.at<float>(i, 4) * image.rows);
            int right = static_cast<int>(detections.at<float>(i, 5) * image.cols);
            int bottom = static_cast<int>(detections.at<float>(i, 6) * image.rows);

            classIds.push_back(classId);
            confidences.push_back(confidence);
            boxes.emplace_back(left, top, right - left + 1, bottom - top + 1);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config.confidenceThreshold, config.nmsThreshold, indices);

    std::vector<DTO::DetectionResult> results;
    for (int index : indices) {
        DTO::DetectionResult result;
        result.classId = classIds[index];
        result.className = classes[classIds[index]];
        result.confidence = confidences[index];
        result.box = boxes[index];
        results.push_back(result);
    }

    return results;
}

int SSDModel::getClassNumber() const
{
    return classes.size();
}

void SSDModel::readClassFile()
{
    std::ifstream ifs(config.classFilePath);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "Class File not found: " + config.classFilePath);

    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

void SSDModel::loadModel()
{
    net = cv::dnn::readNetFromTensorflow(config.modelPath, config.configPath);
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
