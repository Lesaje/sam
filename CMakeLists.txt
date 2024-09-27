cmake_minimum_required(VERSION 3.28)
project(cpp)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

file(GLOB project_SRCS src/*.cpp) #src/*.h

# Add project executable
add_executable(cpp ${project_SRCS}
        src/Detection/Video/Video.h
        src/Detection/Video/Video.cpp
        src/Detection/Model/SSDModel.h
        src/Detection/Model/SSDModel.cpp
        src/Detection/Detector/Detector.h
        src/Detection/Detector/Detector.cpp
        src/Detection/Detection.h
        src/Detection/Detection.cpp
)
target_link_libraries(cpp ${OpenCV_LIBRARIES})


