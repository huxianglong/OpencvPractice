#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"


using namespace std;
using namespace cv;
using namespace cv::face;


int main(int argc,char** argv)
{
    // Load Face Detector
    CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("lbfmodel.yaml");

    // Set up webcam for video capture
//    VideoCapture cam(0);
    VideoCapture cap("D:\\Code\\CharacterFaceGen\\FaceLandmarkDetection\\v45_112_Life_Of_Pi_Lying_Actress_tilts_her_head_talking.mp4");

    // Variable to store a video frame and its grayscale
    Mat frame, gray;

//    list<vector<vector<Point2f>>> landmarks;
    // Read a frame
    while(cap.read(frame))
    {

        // Find face
        vector<Rect> faces;
        // Convert frame to grayscale because
        // faceDetector requires grayscale image.
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        faceDetector.detectMultiScale(gray, faces);

        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        vector< vector<Point2f> > local_landmark;

        // Run local_landmark detector
        bool success = facemark->fit(frame,faces,local_landmark);

        if(success)
        {
            // If successful, render the landmarks on the face
            for(int i = 0; i < local_landmark.size(); i++)
            {
//                drawLandmarks(frame, local_landmark[i]);
                cout << local_landmark[i].size() << endl;
            }
        }

        // Display results
//        imshow("Facial Landmark Detection", frame);
        // Exit loop if ESC is pressed
//        if (waitKey(1) == 27) break;

    }
    return 0;
}
