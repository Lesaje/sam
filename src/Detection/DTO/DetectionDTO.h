#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

namespace DTO {

    enum class SourceType {
        FILE,
        WEBCAM
    };

    struct DetectionResult {
        int classId;
        std::string className;
        float confidence;
        cv::Rect box;
    };

    struct FrameData {
        cv::Mat frame;
        std::vector<DetectionResult> detections;
    };

    struct VideoSource {
        std::string path;
        SourceType type;
    };

    struct ModelConfig {
        std::string modelPath;
        std::string configPath;
        std::string classFilePath;
        float confidenceThreshold;
        float nmsThreshold;
    };

} // namespace DTO
