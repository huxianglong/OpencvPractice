cmake_minimum_required(VERSION 3.2)
project(divider)

set(SOURCE_FILES facialLandmarkDetection.cpp)

#set(OpenCV_INCLUDE_PATH "D:\\Library\\opencv\\Mingw_build\\install\\include")
#set(OpenCV_DIR "D:\\Library\\opencv\\Mingw_build\\win-install")
set(OpenCV_DIR "D:\\Library\\opencv\\Build\\install")
FIND_PACKAGE(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
#set(OpenCV_DIR "D:\\Library\\opencv\\build")
#set_property(TARGET MyTarget PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
#set(OpenCV_INCLUDE_PATH "D:\\Library\\opencv\\Mingw_build\\include" )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_PATH}
#        "D:\\Library\\opencv_contrib\\modules\\face\\include"
#        "D:\\Library\\opencv\\include"
        )
#set(OpenCV_LIBS "D:\\Library\\opencv\\Mingw_build\\install\\x86\\vc15\\lib\\opencv_core410.lib" "D:\\Library\\opencv\\Mingw_build\\install\\x86\\vc15\\lib\\opencv_face410.lib")
#set(OpenCV_LIBS "D:\\Library\\opencv\\Mingw_build\\install\\x86\\vc15\\lib")
#set(OpenCV_LIBS "D:\\Library\\opencv\\build\\x64\\vc15\\lib\\opencv_world346d.lib")

#FIND_PACKAGE(OpenCV REQUIRED PATHS "D:\\Library\\opencv\\build\\x64\\vc15\\lib")
#set(OpenCV_FOUND TRUE)
#IF (OpenCV_FOUND)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
#ELSE (OpenCV_FOUND)
#    MESSAGE(FATAL_ERROR "OpenCV library not found")
#ENDIF (OpenCV_FOUND)
