#pragma once
#include "DTO/DetectionDTO.h"
#include "Detector/Detector.h"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

class Detection {
public:
    Detection();
    ~Detection();
    void start();

private:

    std::vector<std::string> videoSources = {
        "/home/kuver/Downloads/test_videos/running.mp4",
        "/home/kuver/Downloads/test_videos/slow.mp4",
        "/home/kuver/Downloads/test_videos/slow.mp4"
    };
    std::vector<DTO::SourceType> sourceTypes = {
        DTO::SourceType::FILE,
        DTO::SourceType::FILE,
        DTO::SourceType::FILE
    };

    std::vector<std::unique_ptr<Detector>> detectors;
    std::vector<std::thread> threads;
    std::vector<std::string> windowNames;
    std::vector<bool> activeStreams;
    std::atomic<bool> running;
    std::mutex frameMutex;

    void processingLoop(int index) const;
    void displayLoop();
    void stop();
};