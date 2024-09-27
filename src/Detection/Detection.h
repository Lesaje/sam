#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <opencv2/highgui.hpp>
#include "Detector/Detector.h"

class Detection {
public:
    Detection();
    ~Detection();
    void start();

private:
    std::vector<std::string> video_sources = {
        "/home/kuver/Downloads/test_videos/running.mp4",
        "/home/kuver/Downloads/test_videos/slow.mp4",
        "/home/kuver/Downloads/test_videos/slow.mp4"
    };
    std::vector<Video::SourceType> source_types = {
        Video::SourceType::FILE,
        Video::SourceType::FILE,
        Video::SourceType::FILE
    };

    std::vector<std::unique_ptr<Detector>> detections;
    std::vector<std::thread> threads;
    std::vector<std::string> window_names;
    std::vector<bool> active_streams;
    std::atomic<bool> running;
    std::mutex frame_mutex;

    void processingLoop(int index);
    void displayLoop();
    void stop();

    std::vector<cv::Mat> getProcessedFrames() const;
    std::vector<std::vector<int>> getAllClassIds() const;
    std::vector<std::vector<std::string>> getAllClassNames() const;
    std::vector<std::vector<float>> getAllConfidences() const;
    std::vector<std::vector<cv::Rect>> getAllBoxes() const;
};
