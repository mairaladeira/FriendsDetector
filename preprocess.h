
#pragma once


#include <stdio.h>
#include <iostream>
#include <vector>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade, CascadeClassifier &eyeCascade2, Point *leftEye, Point *rightEye, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);

Mat preprocessImg(Mat faceImg);

void equalizeLeftAndRightHalves(Mat &faceImg);

Mat preprocessFace(Mat &srcImg, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);