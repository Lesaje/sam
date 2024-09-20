#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#if CV_VERSION_MAJOR < 4
#pragma message( "OpenCV version < 4" )
#endif

#include "./Model/SSDModel.h"
#include "./Video/Video.h"

int main()
{
    float conf_threshold = 0.5;
    float nms_threshold = 0.5;

    std::string img_file = "/home/kuver/Downloads/test_videos/running.mp4";

    const std::string window_name= "Object Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    std::shared_ptr<MessageQueue<cv::Mat>> image_queue(new MessageQueue<cv::Mat>);
    std::shared_ptr<MessageQueue<cv::Mat>> detection_queue(new MessageQueue<cv::Mat>);

    SSDModel ssd_model = SSDModel(conf_threshold, nms_threshold);

    Video input = Video(img_file, ssd_model.getClassNumber());

    input.setImageQueue(image_queue);
    input.setDetectionQueue(detection_queue);
    ssd_model.setDetectionQueue(detection_queue);

    cv::resizeWindow(window_name, input.getWindowSize());

    input.thread_for_read();
    ssd_model.thread_for_detection();

    std::vector<int> classIds;
    std::vector<std::string> classNames;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat current_image;

    const int duration = (int)(1000/input.getFps());
    int count = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    while(cv::waitKey(duration) < 0)
    {

        if(image_queue->getTotal() > 0 && count >= image_queue->getTotal())
        {
            break;
        }

        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
        current_image = image_queue->receive();

        // Execute the detection once per counts specified by getDetectFreq()
        if(count%(input.getDetectFreq()) == 0)
        {
            ssd_model.getNextDetection(classIds, classNames, confidences, boxes);
        }

        // Plot the result and show the image on window
        input.drawResult(current_image, classIds, classNames, confidences, boxes);
        cv::imshow(window_name, current_image);

        ++count;
    }
    std::cout << " --- Object detection finished. Press Enter key to quit.---\n";
    cv::waitKey(0);
    return 0;
}
