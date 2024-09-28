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
    while (running) {
        bool allStreamsEnded = true;

        for (size_t i = 0; i < detectors.size(); ++i) {
            if (activeStreams[i]) {
                allStreamsEnded = false;
                DTO::FrameData frameData;
                {
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

        cv::waitKey(1);
    }
}

void Detection::processingLoop(const int i) const
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i % 3, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    /*
    DTO::VideoSource source{videoSources[i], sourceTypes[i]};
    auto detector = std::make_unique<Detector>(source);
    std::string window_name = "Stream " + std::to_string(i + 1);
    */

    while (running) {
        DTO::FrameData frame = detectors[i]->process();
    }
}
