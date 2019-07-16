//
// Created by temporary on 2019/4/13.
//
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
///
using namespace std;
class PointState
{
public:
    PointState(cv::Point2f point)
            :
            m_point(point),
            m_kalman(4, 2, 0, CV_64F)
    {
        Init();
    }

    void Update(cv::Point2f point)
    {
        cv::Mat measurement(2, 1, CV_64FC1);
        if (point.x < 0 || point.y < 0)
        {
            Predict();
            measurement.at<double>(0) = m_point.x;  //update using prediction
            measurement.at<double>(1) = m_point.y;

            m_isPredicted = true;
        }
        else
        {
            measurement.at<double>(0) = point.x;  //update using measurements
            measurement.at<double>(1) = point.y;

            m_isPredicted = false;
        }

        // Correction
        cv::Mat estimated = m_kalman.correct(measurement);
        m_point.x = static_cast<float>(estimated.at<double>(0));   //update using measurements
        m_point.y = static_cast<float>(estimated.at<double>(1));

        Predict();
    }

    cv::Point2f GetPoint() const
    {
        return m_point;
    }

    bool IsPredicted() const
    {
        return m_isPredicted;
    }

private:
    cv::Point2f m_point;
    cv::KalmanFilter m_kalman;

    double m_deltaTime = 0.2;
    double m_accelNoiseMag = 0.3;

    bool m_isPredicted = false;

    void Init()
    {
        m_kalman.transitionMatrix = (cv::Mat_<double>(4, 4) <<
                                                            1, 0, m_deltaTime, 0,
                0, 1, 0, m_deltaTime,
                0, 0, 1, 0,
                0, 0, 0, 1);

        m_kalman.statePre.at<double>(0) = m_point.x; // x
        m_kalman.statePre.at<double>(1) = m_point.y; // y

        m_kalman.statePre.at<double>(2) = 1; // init velocity x
        m_kalman.statePre.at<double>(3) = 1; // init velocity y

        m_kalman.statePost.at<double>(0) = m_point.x;
        m_kalman.statePost.at<double>(1) = m_point.y;

        cv::setIdentity(m_kalman.measurementMatrix);

        m_kalman.processNoiseCov = (cv::Mat_<double>(4, 4) <<
                                                           pow(m_deltaTime, 4.0) / 4.0, 0, pow(m_deltaTime, 3.0) / 2.0, 0,
                0, pow(m_deltaTime, 4.0) / 4.0, 0, pow(m_deltaTime, 3.0) / 2.0,
                pow(m_deltaTime, 3.0) / 2.0, 0, pow(m_deltaTime, 2.0), 0,
                0, pow(m_deltaTime, 3.0) / 2.0, 0, pow(m_deltaTime, 2.0));


        m_kalman.processNoiseCov *= m_accelNoiseMag;

        cv::setIdentity(m_kalman.measurementNoiseCov, cv::Scalar::all(0.1));

        cv::setIdentity(m_kalman.errorCovPost, cv::Scalar::all(.1));
    }

    cv::Point2f Predict()
    {
        cv::Mat prediction = m_kalman.predict();
        m_point.x = static_cast<float>(prediction.at<double>(0));
        m_point.y = static_cast<float>(prediction.at<double>(1));
        return m_point;
    }
};

///
void TrackPoints(cv::Mat prevFrame, cv::Mat currFrame,
                 const std::vector<cv::Point2f>& currLandmarks,
                 std::vector<PointState>& trackPoints)
{
    // Lucas-Kanade
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::Size winSize(7, 7);

    std::vector<uchar> status(trackPoints.size(), 0);
    std::vector<float> err;
    std::vector<cv::Point2f> newLandmarks;

    std::vector<cv::Point2f> prevLandmarks;
    std::for_each(trackPoints.begin(), trackPoints.end(), [&](const PointState& pts) { prevLandmarks.push_back(pts.GetPoint()); });

    cv::calcOpticalFlowPyrLK(prevFrame, currFrame, prevLandmarks, newLandmarks, status, err, winSize, 3, termcrit, 0, 0.001);

    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            trackPoints[i].Update((newLandmarks[i] + currLandmarks[i]) / 2);
        }
        else
        {
            trackPoints[i].Update(currLandmarks[i]);
        }
    }
}

/// \param argc
/// \param argv
/// \return
using namespace cv;
int main(int argc, char** argv)
{
    cv::CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();

    facemark->loadModel("lbfmodel.yaml");

//    cv::VideoCapture cam(0, cv::CAP_DSHOW);
    cv::VideoCapture cap("D:\\Code\\CharacterFaceGen\\FaceLandmarkDetection\\v45_112_Life_Of_Pi_Lying_Actress_tilts_her_head_talking.mp4");
    // Get frame count
    int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));

    // Get width and height of video stream
    int w = int(cap.get(CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));
    // Get frames per second (fps)
//    double fps = cap.get(CV_CAP_PROP_FPS);
    double fps = cap.get(CAP_PROP_FPS);
    // Set up output video
//    cv::VideoWriter out("video_out.avi", VideoWriter::fourcc('M','J','P','G'), fps, Size(w, h));
    cv::VideoWriter out("video_out.avi", VideoWriter::fourcc('M','J','P','G'), fps, Size(w, h));


    cv::namedWindow("Facial Landmark Detection", cv::WINDOW_NORMAL);

    cv::Mat frame;
    cv::Mat currGray;
    cv::Mat prevGray;

    std::vector<PointState> trackPoints;
    trackPoints.reserve(68);

    while (cap.read(frame))
    {
//        cout << trackPoints.size() << endl;
        std::vector<cv::Rect> faces;
        cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);

        faceDetector.detectMultiScale(currGray, faces, 1.1, 3, cv::CASCADE_FIND_BIGGEST_OBJECT);

        std::vector<std::vector<cv::Point2f>> landmarks;

        bool success = facemark->fit(frame, faces, landmarks);

        if (success)
        {
            if (prevGray.empty())
            {
                trackPoints.clear();

                for (cv::Point2f lp : landmarks[0])
                {
                    trackPoints.emplace_back(lp);
                }
            }
            else
            {
                if (trackPoints.empty())
                {
                    for (cv::Point2f lp : landmarks[0])
                    {
                        trackPoints.emplace_back(lp);
                    }
                }
                else
                {
                    TrackPoints(prevGray, currGray, landmarks[0], trackPoints);
                }
            }

            for (const PointState& tp : trackPoints)
            {
                cv::circle(frame, tp.GetPoint(), 3, tp.IsPredicted() ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), cv::FILLED);
                cv::circle(frame, tp.GetPoint(), 3, cv::Scalar(0, 0, 255), cv::FILLED);
            }

//            for (cv::Point2f lp : landmarks[0])
//            {
//                cv::circle(frame, lp, 2, cv::Scalar(255, 0, 255), cv::FILLED);
//            }
        }

        cv::imshow("Facial Landmark Detection", frame);
        out.write(frame);
        if (cv::waitKey(1) == 27)
            break;

        prevGray = currGray;
    }
    cap.release();
    out.release();
    return 0;
}

