//
// Created by Xianglong Hu on 2019/4/14.
//

#include "faceStable.h"
#include "smooth.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

void drawCircle(cv::Mat &im, const vector<Point2f> &landmarks) {
    for(int i = 0; i < landmarks.size(); i++)
    {
        circle(im,landmarks[i],3, Scalar(255, 200,0), FILLED);
    }
}

int main(int argc, char **argv) {
    cv::CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();

    facemark->loadModel("lbfmodel.yaml");

    cv::VideoCapture cap(
            "D:\\Code\\CharacterFaceGen\\FaceLandmarkDetection\\v45_112_Life_Of_Pi_Lying_Actress_tilts_her_head_talking.mp4");

    // Get frame count
    int nFrames = int(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "total frames " << nFrames << endl;

    // Set up output video
    int w = int(cap.get(CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    cv::VideoWriter outVideo("video_out_stable_6.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(w, h));

    cv::namedWindow("Facial Landmark Detection", cv::WINDOW_NORMAL);

    cv::Mat frame, currGray, prevGray;
    std::vector<cv::Rect> faces;

    int nNeighbor = 6;

    std::vector<cv::Mat> frames, grayFrames;
    std::vector<std::vector<cv::Point2f>> landmarks;
    frames.reserve(nNeighbor + 1);
    grayFrames.reserve(nNeighbor + 1);
    landmarks.reserve(nNeighbor + 1);

    int counter = 0, writeCounter = 0;
    bool isAveraged = false;

    while (cap.read(frame)) {
        counter += 1;
        if (counter == 95) {
            cout << "stop" << endl;
        }
        cout << "-----------------------" << endl;
        cout << "Now dealing frame " << counter << endl;
        cout << "Landmark: " << landmarks.size() << endl;
        cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);
        std::vector<std::vector<cv::Point2f>> localLandMark;
        faceDetector.detectMultiScale(currGray, faces);
        bool success = facemark->fit(frame, faces, localLandMark);
        if (success) {
            // add to the window
            frames.push_back(frame.clone());
            grayFrames.push_back(currGray);
            landmarks.push_back(localLandMark[0]);
            // we assume there is only one face
            if (frames.size() == 2 * nNeighbor + 1) {
                // average the frame in the middle
                if (isAveraged) {
                    std::vector<cv::Point2f> aveLandmark;
                    smooth(grayFrames, landmarks, -nNeighbor, nNeighbor, nNeighbor, aveLandmark);
                    cv::Mat targetFrame = frames[nNeighbor];
                    drawCircle(targetFrame, aveLandmark);
                    writeCounter += 1;
                    cout << "Now writing frame " << writeCounter << endl;
                    outVideo.write(targetFrame);
                    cv::imshow("Facial Landmark Detection", targetFrame);
                    if (cv::waitKey(1) == 27)
                        break;
                    // average the first few frames
                } else {
                    std::vector<cv::Point2f> aveLandmark;
                    for (int i = 0; i <= nNeighbor; ++i) {
                        smooth(grayFrames, landmarks, -nNeighbor, nNeighbor, i, aveLandmark);
                        cv::Mat targetFrame = frames[i];
                        drawCircle(targetFrame, aveLandmark);
                        outVideo.write(targetFrame);
                        writeCounter += 1;
                        cout << "Now writing frame " << writeCounter<< endl;
                        cv::imshow("Facial Landmark Detection", targetFrame);
                        if (cv::waitKey(1) == 27)
                            break;
                    }
                    isAveraged = true;
                }
                // remove the first few outdated frames
                frames.erase(frames.begin());
                grayFrames.erase(grayFrames.begin());
                landmarks.erase(landmarks.begin());
            }
        } else {
            if (landmarks.empty()) {
                writeCounter += 1;
                cout << "Now writing frame " << writeCounter << endl;
                outVideo.write(frame);
            } else {
                int start;
                if (isAveraged) {
                    start = nNeighbor + 1;
                }
                std::vector<cv::Point2f> aveLandmark;
                for (int i = start; i <= landmarks.size() - 1; ++i) {
                    smooth(grayFrames, landmarks, -nNeighbor, nNeighbor, i, aveLandmark);
                    cv::Mat targetFrame = frames[i];
                    drawCircle(targetFrame, aveLandmark);
                    outVideo.write(targetFrame);
                    writeCounter += 1;
                    cout << "Now writing frame " << writeCounter << endl;
                    cv::imshow("Facial Landmark Detection", targetFrame);
                    if (cv::waitKey(1) == 27)
                        break;
                }
            }
            // reinit the window
            isAveraged = false;
            frames.clear();
            grayFrames.clear();
            landmarks.clear();
        }
    }

    cap.release();
    outVideo.release();
    return 0;
}
