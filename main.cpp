#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "preprocess.h"

#include <iostream>
#include <fstream>
#include <sstream>

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
string predictFace(Mat face, Ptr<BasicFaceRecognizer> model, int index);
void getFirstFisherFaces(Mat W, Mat eigenvalues);
double getReconstructedFaceDissimilarity(Mat W, Mat mean, Mat img, int i);
double getDissimilarity(const Mat A, const Mat B);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String eyes_cascade2_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier eyes_cascade2;
String window_name = "Face detection";
vector<Mat> images;
int im_width;
int im_height;

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
        /*std::vector<Rect> eyes;
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, searchScaleFactor, 4, flags, Size(1,1) );

        for ( size_t j = 0; j < 2; j++ ){
            Point2f eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            //circle( originalImg, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }*/

        Mat finalImg = preprocessImg(faceROI);
        string pred_name = predictFace(faceROI, model, i);
        ostringstream box_text;
        box_text << i << " Prediction = " << pred_name;
        int pos_x = std::max(faces[i].tl().x - 10, 0);
        int pos_y = std::max(faces[i].tl().y - 10, 0);
        putText(originalImg, box_text.str(), Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 2.0);
    }
    cv::resize(originalImg, originalImg, Size(600, 600), 1.0, 1.0, INTER_CUBIC);
    imshow( "Detected Features", originalImg );
}

string predictFace(Mat face, Ptr<BasicFaceRecognizer> model, int index){
    ostringstream name;
    name << "Detected Image: " << index;
    Mat face_resized;
    cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
    imshow( name.str(), face_resized );
    int prediction = -1;
    double confidence = 0.0;
    model->predict(face_resized, prediction, confidence);
    cout << "Image: " << index << " predicted as class: " << prediction << " with confidence: " << confidence << "\n";
    //getFirstFisherFaces(model->getEigenVectors(), model->getEigenValues());
    double dissimilarity = getReconstructedFaceDissimilarity(model->getEigenVectors(), model->getMean(), face_resized, index);
    if(dissimilarity >= 0.08)
        prediction = -1;
    string pred_name = getName(prediction);
    return pred_name;
}

void getFirstFisherFaces(Mat W, Mat eigenvalues){
    // Display or save the first, at most 16 Fisherfaces:
    for (int i = 0; i < min(16, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, im_height));
        // Show the image & apply a Bone colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
        // Display or save:
        imshow(format("fisherface_%d", i), cgrayscale);

    }
}

double getDissimilarity(const Mat A, const Mat B) {
    //absdiff(A, B, s1);       // |I1 - I2|
    //s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    //s1 = s1.mul(s1);           // |I1 - I2|^2

    //Scalar s = sum(s1);         // sum elements per channel

    //double sse = s.val[0] + s.val[1] + s.val[2];
    double errorL2 = norm(A, B, CV_L2);
    // Scale the value since L2 is summed across all pixels.
    double dissimilarity = errorL2 / (double)(A.rows * A.cols);
    return dissimilarity;
}

double getReconstructedFaceDissimilarity(Mat W, Mat mean, Mat img, int i) {
    Mat projection = LDA::subspaceProject(W, mean, img.reshape(1,1));
    // Generate the reconstructed face back from the eigenspace.
    Mat reconstructionRow = LDA::subspaceReconstruct(W, mean, projection);
    Mat reconstructionMat = reconstructionRow.reshape(1, im_height);
    // Convert the floating-point pixels to regular 8-bit uchar.
    Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
    reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
    double dissimilarity = getDissimilarity(img, reconstructedFace);
    cout << "dissimilarity: " << dissimilarity << "\n";
    imshow(format("reconstructed_face_%d", i), reconstructedFace);
    return dissimilarity;
}

/** @function main */
int main( int argc, char** argv ){
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    if( !eyes_cascade2.load( eyes_cascade2_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    const char* imagename = argc > 1 ? argv[1] : "data/friends_data/aleksandra/1.jpg";
    string database_file = "data/friends_db.txt";
    // These vectors hold the images and corresponding labels:
    //vector<Mat> images;
    vector<int> labels;
    read_dataset(database_file, images, labels, ';');

    im_width = images[0].cols;
    im_height = images[0].rows;

    string saveModelPath = "face-rec-model.txt";
    Ptr<BasicFaceRecognizer> model;
    model = createEigenFaceRecognizer();
    model->train(images, labels);
    //cout << "Saving the trained model to " << saveModelPath << endl;
    //model->save(saveModelPath);
    //model->load(saveModelPath);
    Mat img = imread(imagename);
    Mat workImg = getWorkImage(img);
    detectAndDisplay( workImg, img, model);

    cvWaitKey(0);
    return 0;
}

