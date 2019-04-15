//
// Created by temporary on 2019/4/14.
//

#include "smooth.h"
#include <opencv2/face.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

void LkEstimate(const cv::Mat& prevGrayFrame, const cv::Mat& currGrayFrame,
                const std::vector<cv::Point2f> &prevLandmarks, std::vector<cv::Point2f> &currLandmarks) {
    std::vector<uchar> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(prevGrayFrame, currGrayFrame, prevLandmarks, currLandmarks, status, err);
}

void smooth(const vector<cv::Mat>& grayFrames,
            const std::vector<std::vector<cv::Point2f>>& landmarks,
            int lb, int ub, int target,
            std::vector<cv::Point2f>& aveLandmark
        ) {
    // temp to save
    std::vector<cv::Point2f> LkLandmark;
    // target
    const cv::Mat& targetGrayFrame = grayFrames[target];
    aveLandmark = landmarks[target];
    int counter = 1;
    for (int i = lb; i <= ub; ++i) {
        if (i == 0) break;
        if (target + i < 0 || target + i > landmarks.size()) break;
        counter += 1;
        // neighbor
        const cv::Mat& nbGrayFrame = grayFrames[target + i];
        std::vector<cv::Point2f> nbLandmark = landmarks[target + i];
        LkEstimate(nbGrayFrame, targetGrayFrame, nbLandmark, LkLandmark);
        // sum
        for (int j = 0; j < aveLandmark.size(); ++j) {
            aveLandmark[j] += LkLandmark[j];
        }
    }
    for (int j = 0; j < aveLandmark.size(); ++j) {
        aveLandmark[j] /= counter;
    }
}

