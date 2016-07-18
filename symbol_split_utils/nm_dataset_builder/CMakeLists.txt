# Проект взят из https://github.com/lluisgomez/erfilter_train

cmake_minimum_required(VERSION 2.8)
project(opencv_sandbox)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

find_package(OpenCV 3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

ADD_EXECUTABLE(extract_featuresNM1 extract_featuresNM1.cpp)
ADD_EXECUTABLE(extract_featuresNM2 extract_featuresNM2.cpp)
#ADD_EXECUTABLE(train_classifierNM1 ./train_classifierNM1.cpp)
#ADD_EXECUTABLE(train_classifierNM2 ./train_classifierNM2.cpp)

configure_file(build_csv_datasetNM1.py build_csv_datasetNM1.py COPYONLY)
configure_file(build_csv_datasetNM2.py build_csv_datasetNM2.py COPYONLY)

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(extract_featuresNM1 ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(extract_featuresNM2 ${OpenCV_LIBS})
#  TARGET_LINK_LIBRARIES(train_classifierNM1 ${OpenCV_LIBS})
#  TARGET_LINK_LIBRARIES(train_classifierNM2 ${OpenCV_LIBS})
ENDIF()
