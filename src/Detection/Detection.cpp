#include "Detection.h"
#include <opencv2/highgui.hpp>
#include <pthread.h>

Detection::Detection() : running(false)
{
    for (size_t i = 0; i < videoSources.size(); ++i) {
        DTO::VideoSource source{videoSources[i], sourceTypes[i]};
        detectors.push_back(std::make_unique<Detector>(source));
        windowNames.push_back("Stream " + std::to_string(i + 1));
        activeStreams.push_back(true);
    }
}

Detection::~Detection()
{
    stop();
}

void Detection::start()
{
    running = true;
    for (size_t i = 0; i < detectors.size(); ++i) {
        threads.emplace_back(&Detection::processingLoop, this, i);
    }

    displayLoop();

    stop();
}

void Detection::stop()
{
    running = false;
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads.clear();

    for (const auto& windowName : windowNames) {
        cv::destroyWindow(windowName);
    }
}

void Detection::displayLoop()
{
    for (const auto& windowName : windowNames) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    }

    while (running) {
        bool allStreamsEnded = true;

        for (size_t i = 0; i < detectors.size(); ++i) {
            if (activeStreams[i]) {
                allStreamsEnded = false;
                DTO::FrameData frameData;
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    frameData = detectors[i]->getProcessedFrame();
                }
                if (!frameData.frame.empty()) {
                    cv::imshow(windowNames[i], frameData.frame);
                }
            }
        }

        if (allStreamsEnded) {
            running = false;
        }

        int key = cv::waitKey(1);
        if (key >= 0) {
            running = false;
        }
    }
}

void Detection::processingLoop(const int index) const
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(index % 3, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (running) {
        detectors[index]->process();
    }
}
