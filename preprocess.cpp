const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.90;         // Controls how tall the face mask is.
bool rotation_def(true);
#include "preprocess.h"     // Easily preprocess face images, for face recognition.


/** @function preprocessImg
 * this function preprocess an image in order to obtain the appropriated images to be used in the model.
 * It performs the following tasks:
 * Histogram equalization of left and right side of the face
 * Bilateral Filter
 * Ellipse mask
 */
Mat preprocessImg(Mat faceImg) {
    equalizeLeftAndRightHalves(faceImg);
    Mat filtered = Mat(faceImg.size(), CV_8U);
    bilateralFilter(faceImg, filtered, 0, 20.0, 2.0);
    Mat mask = Mat(faceImg.size(), CV_8U, Scalar(0)); // Start with an empty mask.
    Point faceCenter = Point( faceImg.cols/2, cvRound(faceImg.rows * FACE_ELLIPSE_CY) );
    Size size = Size( cvRound(faceImg.cols * FACE_ELLIPSE_W), cvRound(faceImg.rows * FACE_ELLIPSE_H) );
    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
    Mat dstImg = Mat(faceImg.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
    filtered.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
    return dstImg;
}


/** @function equalizeLeftAndRightHalves
* Histogram Equalize seperately for the left and right sides of the face.
* This function was obtained from https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition
*/
void equalizeLeftAndRightHalves(Mat &faceImg){

    int w = faceImg.cols;
    int h = faceImg.rows;
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) Equalize the left half and the right half of the face separately.
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {          // Left 25%: just use the left face.
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {   // Mid-left 25%: blend the left face & whole face.
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the whole face as it moves further right along the face.
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {   // Mid-right 25%: blend the right face & whole face.
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the right-side face as it moves further right along the face.
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {                  // Right 25%: just use the right face.
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }// end x loop
    }//end y loop
}

/** @function scaleImg
* Scale and image to 320 width and return the scaled version
*/
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


/** @function getWorkImage
 * Get the appropriated image for the haar cascaded classifier.
 * It performs the following tasks:
 * - Conversion to gray scale
 * - Histogram Equalization
 */
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


/** @function setEyeCoordinates
 * Set the Eyes coordinates using the haar cascade classifier for getting the eyes location in the image
 */
void setEyeCoordinates(int *leftX, int *leftY, int *rightX, int *rightY, Rect face, Mat originalImg, CascadeClassifier eyes_cascade){
    int tlY = face.y;
    if (tlY < 0)tlY = 0;

    int drY = face.y + face.height;
    if (drY > originalImg.rows) drY = originalImg.rows;

    Point tl(face.x, tlY);
    Point dr(face.x + face.width, drY);

    Rect newROI(tl, dr);
    Mat croppedImage_original = originalImg(newROI);
    Mat croppedImageGray;
    cvtColor(croppedImage_original, croppedImageGray, CV_RGB2GRAY);
    std::vector<Rect> eyes;

    eyes_cascade.detectMultiScale(croppedImageGray, eyes, 1.1, 6, CV_HAAR_DO_CANNY_PRUNING, Size(croppedImageGray.size().width*0.2, croppedImageGray.size().height*0.2));

    int eyeLeftX = 0;
    int eyeLeftY = 0;
    int eyeRightX = 0;
    int eyeRightY = 0;
    for (size_t j = 0; j < 2; j++){
            int tlY2 = eyes[j].y + face.y;
            if (tlY2 < 0) tlY2 = 0;
            int drY2 = eyes[j].y + eyes[j].height + face.y;
            if (drY2>originalImg.rows) drY2 = originalImg.rows;
            Point tl2(eyes[j].x + face.x, tlY2);
            Point dr2(eyes[j].x + eyes[j].width + face.x, drY2);
            if (eyeLeftX == 0 && eyeLeftY == 0){
                rectangle(originalImg, tl2, dr2, Scalar(255, 0, 0));
                eyeLeftX = eyes[j].x;
                eyeLeftY = eyes[j].y;
                Rect r1(tl2, dr2);
            }
            else if (eyeRightX == 0 && eyeRightY == 0){
                rectangle(originalImg, tl2, dr2, Scalar(255, 0, 0));
                eyeRightX = eyes[j].x;
                eyeRightY = eyes[j].y;
                Rect r2(tl2, dr2);
            }
        }
    *leftX = eyeLeftX;
    *leftY = eyeLeftY;
    *rightX = eyeRightX;
    *rightY = eyeRightY;
}


/** @function getCroppedImage
* get the cropped face image using the appropriated rotation for the eyes
*/
Mat getCroppedImage(int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, Rect face, Mat faceROI, CascadeClassifier face_cascade) {
    Mat croppedImage;
    if(abs(eyeRightX-eyeLeftX)>50){
        if (!(eyeLeftX == 0 && eyeLeftY == 0)){
            if (eyeLeftX > eyeRightX) croppedImage = cropFace(faceROI, eyeRightX, eyeRightY, eyeLeftX, eyeLeftY, 320, 320, face.x, face.y, face.width, face.height, face_cascade);
            else croppedImage = cropFace(faceROI, eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, 320, 320, face.x, face.y, face.width, face.height, face_cascade);
        }
        else croppedImage = faceROI;

    }
    else croppedImage = faceROI;
    return croppedImage;
}


/** @function rotate
 * rotate the face image according to an angle
 */
void rotate(Mat& src, double angle, Mat& dst){
    int len = max(src.cols, src.rows);
    Point2f pt(len / 2., len / 2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, cv::Size(len, len));
}


/** @function cropFace
* Perform the crop task in the face
*/
Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight, CascadeClassifier face_cascade){
    Mat dstImg;
    Mat crop;
    int eye_directionX = eyeRightX - eyeLeftX;
    int eye_directionY = eyeRightY - eyeLeftY;
    float rotation = atan2((float)eye_directionY, (float)eye_directionX) * 180 / PI;

    if (rotation_def) rotate(srcImg, rotation, dstImg);
    else dstImg = srcImg;

  	std::vector<Rect> faces;
	face_cascade.detectMultiScale(dstImg, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(dstImg.size().width*0.2, dstImg.size().height*0.2));

    //FIXME: when the rotation is wrong
    if(faces.size() == 0)  return srcImg;

    for (size_t i = 0; i < faces.size(); i++){
        int tlY = faces[i].y;
        if (tlY < 0) tlY = 0;

        int drY = faces[i].y + faces[i].height;
        if (drY > dstImg.rows){
            drY = dstImg.rows;
        }
        Point tl(faces[i].x, tlY);
        Point dr(faces[i].x + faces[i].width, drY);

        Rect myROI(tl, dr);
        Mat croppedImage_original = dstImg(myROI);
        Mat croppedImageGray;
        resize(croppedImage_original, crop, Size(width, height), 0, 0, INTER_CUBIC);
        //imshow("ROTATION", crop);
    }

    return crop;
}
