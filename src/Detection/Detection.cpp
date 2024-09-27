#include "Detection.h"


Detection::Detection() : running(false)
{
    for (size_t i = 0; i < video_sources.size(); ++i) {
        detections.push_back(std::make_unique<Detector>(video_sources[i], source_types[i]));
        window_names.push_back("Stream " + std::to_string(i + 1));
        active_streams.push_back(true);
    }
}

Detection::~Detection()
{
    stop();
}

void Detection::start()
{
    running = true;
    for (size_t i = 0; i < detections.size(); ++i) {
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

    for (const auto& window_name : window_names) {
        cv::destroyWindow(window_name);
    }
}

void Detection::displayLoop()
{
    for (const auto& window_name : window_names) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    }

    while (running) {
        bool all_streams_ended = true;

        for (size_t i = 0; i < detections.size(); ++i) {
            if (active_streams[i]) {
                all_streams_ended = false;
                cv::Mat frame;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    frame = detections[i]->getProcessedFrame();
                }
                if (!frame.empty()) {
                    cv::imshow(window_names[i], frame);
                }
            }
        }

        if (all_streams_ended) {
            running = false;
        }

        int key = cv::waitKey(1);
        if (key >= 0) {
            running = false;
        }
    }
}

void Detection::processingLoop(int index)
{
    // Set thread affinity to a specific core (0, 1, or 2)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(index % 3, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (running) {
        detections[index]->process();
    }
}

std::vector<cv::Mat> Detection::getProcessedFrames() const
{
    std::vector<cv::Mat> frames;
    for (const auto& detection : detections) {
        frames.push_back(detection->getProcessedFrame());
    }
    return frames;
}

std::vector<std::vector<int>> Detection::getAllClassIds() const
{
    std::vector<std::vector<int>> all_class_ids;
    for (const auto& detection : detections) {
        all_class_ids.push_back(detection->getClassIds());
    }
    return all_class_ids;
}

std::vector<std::vector<std::string>> Detection::getAllClassNames() const
{
    std::vector<std::vector<std::string>> all_class_names;
    for (const auto& detection : detections) {
        all_class_names.push_back(detection->getClassNames());
    }
    return all_class_names;
}

std::vector<std::vector<float>> Detection::getAllConfidences() const
{
    std::vector<std::vector<float>> all_confidences;
    for (const auto& detection : detections) {
        all_confidences.push_back(detection->getConfidences());
    }
    return all_confidences;
}

std::vector<std::vector<cv::Rect>> Detection::getAllBoxes() const
{
    std::vector<std::vector<cv::Rect>> all_boxes;
    for (const auto& detection : detections) {
        all_boxes.push_back(detection->getBoxes());
    }
    return all_boxes;
}