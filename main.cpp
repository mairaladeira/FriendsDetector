#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "preprocess.h"

#include <iostream>
#include <fstream>
#include <sstream>
#define PI 3.14159265
using namespace cv;
using namespace cv::face;
using namespace std;
/** Function Headers */
void detectAndDisplay( Mat workingImg, Mat originalImg, Ptr<BasicFaceRecognizer> model);
static void read_dataset(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
Mat scaleImg(Mat img);
Mat getWorkImage(Mat img);
Mat preprocessImg(Mat faceImg, int faceId);
string getName(int prediction);
int predictFace(Mat face, Ptr<BasicFaceRecognizer> model, int index);
Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight);



/** Global variables */
//bool rotation_def(true);
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String eyes_cascade2_name = "haarcascade_eye_tree_eyeglasses.xml";
//String eyes_right_2splits_name = "haarcascade_righteye_2splits.xml";
//String eyes_left_2splits_name = "haarcascade_lefteye_2splits.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier eyes_cascade2;
//CascadeClassifier eyes_right_2splits;
//CascadeClassifier eyes_left_2splits;
String window_name = "Face detection";
vector<Mat> images;
int im_width;
int im_height;
bool rotation_def(true);

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
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
            Mat img = imread(path, 0);
            int im_width = 320;
            int im_height = 320;
            cv::resize(img, img, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            images.push_back(img);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


string getName(int prediction) {
    switch(prediction) {
        case 0:
            return "Aleksandra";
        case 1:
            return "Gabriela";
        case 2:
            return "Julian";
        case 3:
            return "Kienka";
        case 4:
            return "Maira";
        default:
            return "Unknown";
    }
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat workingImg, Mat originalImg, Ptr<BasicFaceRecognizer> model){
    std::vector<Rect> faces;
    Mat croppedImage;
    int flags = 0|CASCADE_SCALE_IMAGE;
    Size minFeatureSize(20, 20);
    float searchScaleFactor = 1.1f;
    int minNeighbors = 6;
    face_cascade.detectMultiScale(workingImg, faces, searchScaleFactor, 3, CV_HAAR_DO_CANNY_PRUNING, Size(originalImg.size().width*0.2, originalImg.size().height*0.2));
    //face_cascade.detectMultiScale( workingImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = workingImg(faces[i]);
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( originalImg, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        int tlY = faces[i].y;
        //tlY -= (faces[i].height / 3);
        if (tlY < 0){
            tlY = 0;
        }
        int drY = faces[i].y + faces[i].height;
        //drY += +(faces[i].height / 6);
        if (drY > originalImg.rows)
        {
            drY = originalImg.rows;
        }
        Point tl(faces[i].x, tlY);
        Point dr(faces[i].x + faces[i].width, drY);

        Rect newROI(tl, dr);
        Mat croppedImage_original = originalImg(newROI);
        Mat croppedImageGray;
        cvtColor(croppedImage_original, croppedImageGray, CV_RGB2GRAY);
        imshow( "testing cropped face eyes", croppedImageGray );

        std::vector<Rect> eyes;
        //std::vector<Rect> eyes2;
        //std::vector<Rect> eyesR;
        //std::vector<Rect> eyes2R;
        //std::vector<Rect> eyeL;
        //std::vector<Rect> eyes;

        eyes_cascade.detectMultiScale(croppedImageGray, eyes, 1.1, 6, CV_HAAR_DO_CANNY_PRUNING, Size(croppedImageGray.size().width*0.2, croppedImageGray.size().height*0.2));

        int eyeLeftX = 0;
        int eyeLeftY = 0;
        int eyeRightX = 0;
        int eyeRightY = 0;
        for (size_t j = 0; j < 2; j++){
            int tlY2 = eyes[j].y + faces[i].y;
            if (tlY2 < 0){
                tlY2 = 0;
            }
            int drY2 = eyes[j].y + eyes[j].height + faces[i].y;
            if (drY2>originalImg.rows)
            {
                drY2 = originalImg.rows;
            }
            Point tl2(eyes[j].x + faces[i].x, tlY2);
            Point dr2(eyes[j].x + eyes[j].width + faces[i].x, drY2);

            if (eyeLeftX == 0 && eyeLeftY == 0)
            {
                rectangle(originalImg, tl2, dr2, Scalar(255, 0, 0));
                eyeLeftX = eyes[j].x;
                eyeLeftY = eyes[j].y;
                Rect r1(tl2, dr2);

            }
            else if (eyeRightX == 0 && eyeRightY == 0)
            {
                rectangle(originalImg, tl2, dr2, Scalar(255, 0, 0));
                eyeRightX = eyes[j].x;
                eyeRightY = eyes[j].y;
                Rect r2(tl2, dr2);
            }

        }
        cout << "eye left X " << eyeLeftX<< endl;
        cout << "eye left Y " << eyeLeftY<< endl;
        cout << "eye right X " << eyeRightX<< endl;
        cout << "eye left Y " << eyeRightY<< endl;

        if(abs(eyeRightX-eyeLeftX)>50){
            if (!(eyeLeftX == 0 && eyeLeftY == 0)){
                if (eyeLeftX > eyeRightX){
                    croppedImage = cropFace(workingImg, eyeRightX, eyeRightY, eyeLeftX, eyeLeftY, 200, 200, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
                }
                else{
                    croppedImage = cropFace(workingImg, eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, 200, 200, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
                }
            }else{
                croppedImage = faceROI;
            }
        } else{
             croppedImage = faceROI;
            }
        //Mat finalImg = preprocessImg(croppedImage);
        Mat finalImg = preprocessImg(croppedImage);



        int prediction = predictFace(finalImg, model, i);
        Mat eigenvalues = model->getEigenValues();
        Mat eigenvectors = model -> getEigenVectors();
        //Mat eigenvectors = model->get<Mat>("eigenvectors");
        string pred_name = getName(prediction);
        ostringstream box_text;
        box_text << i << " Prediction = " << pred_name;
        // Calculate the position for annotated text (make sure we don't
        // put illegal values in there):
        int pos_x = std::max(faces[i].tl().x - 10, 0);
        int pos_y = std::max(faces[i].tl().y - 10, 0);
        // And now put it into the image:
        putText(originalImg, box_text.str(), Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 2.0);
        //detectedFaces.push_back(finalImg);
    }
    cv::resize(originalImg, originalImg, Size(600, 600), 1.0, 1.0, INTER_CUBIC);
    imshow( "Detected Features", originalImg );
}

int predictFace(Mat face, Ptr<BasicFaceRecognizer> model, int index){
    ostringstream name;
    name << "Detected Image: " << index;
    Mat face_resized;
    cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
    imshow( name.str(), face_resized );
    int prediction = -1;
    double confidence = 0.0;
    model->predict(face_resized, prediction, confidence);
        cout << "Image: " << index << " predicted as class: " << prediction << " with confidence: " << confidence << "\n";
    return prediction;
}

void rotate(Mat& src, double angle, Mat& dst)
{
    int len = max(src.cols, src.rows);
    Point2f pt(len / 2., len / 2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);

    warpAffine(src, dst, r, cv::Size(len, len));
}


Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight){
    Mat dstImg;
    Mat crop;
    int eye_directionX = eyeRightX - eyeLeftX;
    int eye_directionY = eyeRightY - eyeLeftY;
    float rotation = atan2((float)eye_directionY, (float)eye_directionX) * 180 / PI;
    if (rotation_def){
    	rotate(srcImg, rotation, dstImg);
    }
    else {
    	dstImg = srcImg;
    }
  	std::vector<Rect> faces;
	face_cascade.detectMultiScale(dstImg, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(dstImg.size().width*0.2, dstImg.size().height*0.2));

    for (size_t i = 0; i < faces.size(); i++)
    {
        int tlY = faces[i].y;
        //tlY -= (faces[i].height / 3);
        if (tlY < 0){
            tlY = 0;
        }
        int drY = faces[i].y + faces[i].height;
        //drY += +(faces[i].height / 6);
        if (drY > dstImg.rows)
        {
            drY = dstImg.rows;
        }
        Point tl(faces[i].x, tlY);
        Point dr(faces[i].x + faces[i].width, drY);

        Rect myROI(tl, dr);
        Mat croppedImage_original = dstImg(myROI);
        Mat croppedImageGray;
        resize(croppedImage_original, crop, Size(width, height), 0, 0, INTER_CUBIC);
        //face_cascade.detectMultiScale(dstImg, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(dstImg.size().width*0.2, dstImg.size().height*0.2));
        imshow("ROTATION", crop);
    }

    return crop;
}



/** @function main */
int main( int argc, char** argv ){
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    if( !eyes_cascade2.load( eyes_cascade2_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    //if( !eyes_right_2splits.load( eyes_right_2splits_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    //if( !eyes_left_2splits.load( eyes_left_2splits_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    const char* imagename = argc > 1 ? argv[1] : "data/friends_data/aleksandra/1.jpg";
    string database_file = "data/friends_db.txt";
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    read_dataset(database_file, images, labels, ';');

    im_width = images[0].cols;
    im_height = images[0].rows;

    string saveModelPath = "face-rec-model.txt";
    Ptr<BasicFaceRecognizer> model;
    model = createEigenFaceRecognizer(80, 15000);
    model->train(images, labels);
    cout << "Saving the trained model to " << saveModelPath << endl;
    model->save(saveModelPath);
    //model->load(saveModelPath);
    Mat img = imread(imagename);
    Mat workImg = getWorkImage(img);
    detectAndDisplay( workImg, img, model);

    cvWaitKey(0);
    return 0;
}

