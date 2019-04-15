//
// Created by temporary on 2019/4/14.
//

#ifndef PROJECT_SMOOTH_H
#define PROJECT_SMOOTH_H

#include <opencv2/opencv.hpp>


void LkEstimate(const cv::Mat &prevGrayFrame, const cv::Mat &currGrayFrame,
                const std::vector<cv::Point2f> &prevLandmarks, std::vector<cv::Point2f> &currLandmarks);

void smooth(const std::vector<cv::Mat>& grayFrames,
            const std::vector<std::vector<cv::Point2f>>& landmarks,
            int lb, int ub, int target,
            std::vector<cv::Point2f> &aveLandmark);

#endif //PROJECT_SMOOTH_H
