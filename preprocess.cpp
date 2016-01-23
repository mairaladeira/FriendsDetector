const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.90;         // Controls how tall the face mask is.


#include "preprocess.h"     // Easily preprocess face images, for face recognition.

// Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
// or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
// want to search eyes using 2 different cascades. For example, you could use a regular eye detector
// as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
// Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
// Can also store the searched left & right eye regions if desired.
void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point *leftEye, Point *rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
    // For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.
    /*const float EYE_SX = 0.16f;
    const float EYE_SY = 0.26f;
    const float EYE_SW = 0.30f;
    const float EYE_SH = 0.28f;

    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner

    Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));*/
    Rect leftEyeRect, rightEyeRect;

    /*// Return the search windows to the caller, if desired.
    if (searchedLeftEye)
        *searchedLeftEye = Rect(leftX, topY, widthX, heightY);
    if (searchedRightEye)
        *searchedRightEye = Rect(rightX, topY, widthX, heightY);*/

    // Search the left region, then the right region using the 1st eye detector.
    int flags = 0|CASCADE_SCALE_IMAGE;
    std::vector<Rect> eyes;
    eyeCascade1.detectMultiScale( face, eyes, 1.1f, 4, flags, Size(5,5) );
    if (eyes.size() == 2) {
        *searchedLeftEye = eyes[0];
        *searchedRightEye = eyes[1];
        leftEye->x = eyes[0].x;
        leftEye->y = eyes[0].y;
        rightEye->x = eyes[1].x;
        rightEye->y = eyes[1].y;
    }



    if (eyes.size() != 2 && !eyeCascade2.empty()) {
        eyeCascade2.detectMultiScale( face, eyes, 1.1f, 4, flags, Size(5,5) );
        if (eyes.size() > 1) {
            *searchedLeftEye = eyes[0];
            *searchedRightEye = eyes[1];
            leftEye->x = eyes[0].x;
            leftEye->y = eyes[0].y;
            rightEye->x = eyes[1].x;
            rightEye->y = eyes[1].y;
        }
    }

    if (eyes[0].width <= 0) {   // Check if the eye was detected.
        *leftEye = Point(-1, -1);    // Return an invalid point
    }

    if (eyes[1].width <= 0) { // Check if the eye was detected.
        *rightEye = Point(-1, -1);    // Return an invalid point
    }
}

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

// Histogram Equalize seperately for the left and right sides of the face.
void equalizeLeftAndRightHalves(Mat &faceImg){

    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) First, equalize the whole face.
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


// Create a grayscale face image that has a standard size and contrast & brightness.
// "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
// If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
// so that if there is a strong light on one side but not the other, it will still look OK.
// Performs Face Preprocessing as a combination of:
//  - geometrical scaling, rotation and translation using Eye Detection,
//  - smoothing away image noise using a Bilateral Filter,
//  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
//  - removal of background and hair using an Elliptical Mask.
// Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
// If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
// and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
Mat preprocessFace(Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *searchedLeftEye, Rect *searchedRightEye)
{

    // Mark the detected face region and eye search regions as invalid, in case they aren't detected.
    if (searchedLeftEye)
        searchedLeftEye->width = -1;
    if (searchedRightEye)
        searchedRightEye->width = -1;

    // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
    Mat gray;
    if (face.channels() == 3) {
        cvtColor(face, gray, CV_BGR2GRAY);
    }
    else if (face.channels() == 4) {
        cvtColor(face, gray, CV_BGRA2GRAY);
    }
    else {
        // Access the input image directly, since it is already grayscale.
        gray = face;
    }

    // Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible!
    Point leftEye, rightEye;
    detectBothEyes(gray, eyeCascade1, eyeCascade2, &leftEye, &rightEye, searchedLeftEye, searchedRightEye);


    // Check if both eyes were detected.
    if (leftEye.x >= 0 && rightEye.x >= 0) {

        Mat warped = gray;
        if (!doLeftAndRightSeparately) {
            // Do it on the whole face.
            equalizeHist(warped, warped);
        }
        else {
            // Do it seperately for the left and right sides of the face.
            equalizeLeftAndRightHalves(warped);
        }
        //imshow("equalized", warped);

        // Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
        Mat filtered = Mat(warped.size(), CV_8U);
        bilateralFilter(warped, filtered, 0, 20.0, 2.0);
        //imshow("filtered", filtered);

        // Filter out the corners of the face, since we mainly just care about the middle parts.
        // Draw a filled ellipse in the middle of the face-sized image.
        Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Start with an empty mask.
        Point faceCenter = Point( face.cols/2, cvRound(face.rows * FACE_ELLIPSE_CY) );
        Size size = Size( cvRound(face.cols * FACE_ELLIPSE_W), cvRound(face.rows * FACE_ELLIPSE_H) );
        ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
        //imshow("mask", mask);

        // Use the mask, to remove outside pixels.
        Mat dstImg = Mat(face.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
        /*
        namedWindow("filtered");
        imshow("filtered", filtered);
        namedWindow("dstImg");
        imshow("dstImg", dstImg);
        namedWindow("mask");
        imshow("mask", mask);
        */
        // Apply the elliptical mask on the face.
        filtered.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
        //imshow("dstImg", dstImg);

        return dstImg;
    }
    return Mat();
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
