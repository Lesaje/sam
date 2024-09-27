#include "Detector.h"

Detector::Detector(const DTO::VideoSource& videoSource)
{
    this->model = std::make_unique<SSDModel>(DTO::ModelConfig{
        "../resources/frozen_inference_graph.pb",
        "../resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
        "../resources/object_detection_classes_coco.txt",
        0.6f,
        0.9f
    });
    this->video = std::make_unique<Video>(videoSource);
}

Detector::~Detector() = default;

void Detector::process()
{
    cv::Mat frame = video->getNextFrame();
    if (!frame.empty()) {
        processedFrame = detect(frame);
    } else {
        processedFrame = DTO::FrameData();  //TODO: maybe should change default frame to something different
    }
}

DTO::FrameData Detector::detect(const cv::Mat& frame)
{
    DTO::FrameData frameData;
    frameData.frame = frame;
    frameData.detections = model->detectObjects(frame);
    video->drawDetectionResults(frameData);
    return frameData;
}

DTO::FrameData Detector::getProcessedFrame() const
{
    return processedFrame;
}
