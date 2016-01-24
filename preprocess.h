
#pragma once



// Include OpenCV's C++ Interface
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#define PI 3.14159265



using namespace cv;
using namespace std;

/** @function preprocessImg
 * this function preprocess an image in order to obtain the appropriated images to be used in the model.
 * It performs the following tasks:
 * Histogram equalization of left and right side of the face
 * Bilateral Filter
 * Ellipse mask
 */
Mat preprocessImg(Mat faceImg);

/** @function equalizeLeftAndRightHalves
* Histogram Equalize seperately for the left and right sides of the face.
* This function was obtained from https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition
*/
void equalizeLeftAndRightHalves(Mat &faceImg);

/** @function getWorkImage
 * Get the appropriated image for the haar cascaded classifier.
 * It performs the following tasks:
 * - Conversion to gray scale
 * - Histogram Equalization
 */
Mat getWorkImage(Mat img);

/** @function scaleImg
* Scale and image to 320 width and return the scaled version
*/
Mat scaleImg(Mat img);

/** @function setEyeCoordinates
 * Set the Eyes coordinates using the haar cascade classifier for getting the eyes location in the image
 */
void setEyeCoordinates(int *leftX, int *leftY, int *rightX, int *rightY, Rect face, Mat originalImg, CascadeClassifier eyes_cascade);


/** @function cropFace
* Perform the crop task in the face
*/
Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight, CascadeClassifier face_cascade);

/** @function rotate
 * rotate the face image according to an angle
 */
void rotate(Mat& src, double angle, Mat& dst);


/** @function getCroppedImage
* get the cropped face image using the appropriated rotation for the eyes
*/
Mat getCroppedImage(int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, Rect face, Mat faceROI, CascadeClassifier face_cascade);

