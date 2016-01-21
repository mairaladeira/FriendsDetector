
//#include "opencv2/opencv.hpp"
#include <iostream>
//#include <stdio.h>
#include "preprocess.h"
#include "opencv2/face.hpp"
#include <fstream>
#include <sstream>
#include "opencv2/core.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

/** Function Headers */
vector<Mat> detectAndDisplay( Mat workingImg, Mat originalImg);
static void read_dataset(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
Mat scaleImg(Mat img);
Mat getWorkImage(Mat img);
Mat preprocessImg(Mat faceImg, int faceId);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String eyes_cascade2_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier eyes_cascade2;
String window_name = "Face detection";
vector<Mat> images;

/** @function main */
int main( int argc, char** argv ){
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    if( !eyes_cascade2.load( eyes_cascade2_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    const char* imagename = argc > 1 ? argv[1] : "lena.jpg";
    string database_file = "data/att_database_train.txt";
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    read_dataset(database_file, images, labels, ';');

    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);

    //Mat eigenvectors = model->get<Mat>("eigenvectors");
    //printMatInfo(eigenvectors, "eigenvectors");
    //exit(1);
    Mat img = imread(imagename); // the newer cvLoadImage alternative, MATLAB-style function
    //img = scaleImg(img);
    Mat workImg = getWorkImage(img);
    imshow( "Original Image", workImg );
    vector<Mat> detectedImgs = detectAndDisplay( workImg, img);
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

    // wait for a key
    for ( size_t i = 0; i < detectedImgs.size(); i++ ){
        //ostringstream name;
        //name << "Detected Image: " << i;
        //imshow( name.str(), detectedImgs[i] );
        Mat face_resized;
        cv::resize(detectedImgs[i], face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
        int prediction;
        double confidence;
        model->predict(face_resized, prediction, confidence);
        cout << "predicted class: " << prediction << " with confidence: " << confidence << "\n";
        Mat W = model->getEigenVectors();

    }
    cvWaitKey(0);
    return 0;
}

static void read_dataset(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

Mat scaleImg(Mat img) {
    const int DETECTION_WIDTH = 320;
    Mat smallImg;
    float scale = img.cols/(float) DETECTION_WIDTH;
    if(img.cols > DETECTION_WIDTH) {
        //Shrink the image while keeping the same aspect ratio.
        int scaledHeight = cvRound(img.rows/scale);
        resize(img, smallImg, Size(DETECTION_WIDTH, scaledHeight));
    } else {
        //if image is already small enough
        smallImg = img;
    }
    return smallImg;
}


Mat getWorkImage(Mat img){
    //Transform image to gray scale
    Mat gray;
    if(img.channels() == 3) {
        cvtColor(img, gray, CV_BGR2GRAY);
    } else if(img.channels() == 4) {
        cvtColor(img, gray, CV_BGRA2GRAY);
    } else {
        //if it is already in gray scale
        gray = img;
    }
    //Scale down the image size to avoid big inputs

    Mat equalizedImg;
    equalizeHist(gray, equalizedImg);
    return equalizedImg;
}



/** @function detectAndDisplay */
vector<Mat> detectAndDisplay( Mat workingImg, Mat originalImg ){
    std::vector<Rect> faces;
    vector<Mat> detectedFaces;
    int flags = 0|CASCADE_SCALE_IMAGE;
    Size minFeatureSize(20, 20);
    float searchScaleFactor = 1.1f;
    int minNeighbors = 6;

    face_cascade.detectMultiScale( workingImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = workingImg(faces[i]);
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( originalImg, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        /*Point leftEyePoint;
        Point rightEyePoint;
        Rect leftEye;
        Rect rightEye;
        Mat processedFace = preprocessFace(faceROI, eyes_cascade, eyes_cascade2, true, &leftEye, &rightEye);*/
        std::vector<Rect> eyes;
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, searchScaleFactor, 2, flags, Size(1,1) );

        for ( size_t j = 0; j < eyes.size(); j++ ){
            Point2f eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( originalImg, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }

        Mat finalImg = preprocessImg(faceROI);
        //imshow( "Detected Features", originalImg );
        //imshow( "Processed Image"+i, finalImg );
        detectedFaces.push_back(finalImg);
    }
    return detectedFaces;
    //-- Show what you got
    //imshow( window_name, originalImg );
}
