cmake_minimum_required(VERSION 3.0)
project (AlphaMatting)
find_package(OpenCV REQUIRED)

find_package(gflags REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(glog REQUIRED)

file(GLOB SOURCES "*.cpp")

add_executable(AlphaMatting ${SOURCES})
target_link_libraries(AlphaMatting ${OpenCV_LIBS}
                      ${CV_UTILS_LIBS}
                      gflags fmt glog)
target_include_directories(AlphaMatting PUBLIC
                           ${CV_UTILS_DIRS}
                           ${BASE_DIRS})

