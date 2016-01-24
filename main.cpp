#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "preprocess.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
using namespace cv;
using namespace cv::face;
using namespace std;
/** Function Headers */
/** @function detectAndDisplay */
void detectAndDisplay( Mat workingImg, Mat originalImg, Ptr<BasicFaceRecognizer> model);
/** @function getName */
string getName(int prediction);
/** @function predictFace */
int predictFace(Mat face, Ptr<BasicFaceRecognizer> model, int index);
/** @function getReconstructedFaceDissimilarity */
double getReconstructedFaceDissimilarity(Mat W, Mat mean, Mat img, int i);
/** @function getSimilarity */
double getSimilarity(const Mat A, const Mat B);
/** @function getAverageFace */
void getAverageFace(Mat mean, int index);

/** Global variables */
//bool rotation_def(true);
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String eyes_cascade2_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier eyes_cascade2;
String window_name = "Face detection";
const string model_name = "eigenfaces"; //set to eigenfaces or fisherfaces
vector<Mat> images;
int im_width;
int im_height;
double treshold;
double confidence_treshold;
void display_reconstructions(Mat img, Mat W, Mat mean, int index);
//bool rotation_def(true);

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
            return "Kienka";
        case 3:
            return "Maira";
        default:
            return "Unknown";
    }
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat workingImg, Mat originalImg, Ptr<BasicFaceRecognizer> model){
    std::vector<Rect> faces;
    Mat croppedImage;
    float searchScaleFactor = 1.1f;
    face_cascade.detectMultiScale(workingImg, faces, searchScaleFactor, 3, CV_HAAR_DO_CANNY_PRUNING, Size(20,20));

    for ( size_t i = 0; i < faces.size(); i++ ){
        Mat faceROI = workingImg(faces[i]);
        //imshow(format("Detected Face %d", i), faceROI);
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( originalImg, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        int eyeLeftX = 0;
        int eyeLeftY = 0;
        int eyeRightX = 0;
        int eyeRightY = 0;
        setEyeCoordinates(&eyeLeftX, &eyeLeftY, &eyeRightX, &eyeRightY, faces[i], originalImg, eyes_cascade);

        croppedImage = getCroppedImage(eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, faces[i], faceROI, face_cascade);
        cv::resize(croppedImage, croppedImage, Size(320, 320), 1.0, 1.0, INTER_CUBIC);
        Mat finalImg = preprocessImg(croppedImage);
        imshow(format("Pre processed image %d",i), finalImg);

        int prediction = predictFace(finalImg, model, i);
        string pred_name = getName(prediction);
        ostringstream box_text;
        box_text << i << " Prediction = " << pred_name;
        int pos_x = std::max(faces[i].tl().x - 10, 0);
        int pos_y = std::max(faces[i].tl().y - 10, 0);
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
    int prediction = -1;
    double confidence = 0.0;
    model->predict(face_resized, prediction, confidence);
    Mat mean = model->getMean();
    Mat eigenvectors = model -> getEigenVectors();
    double sim = getReconstructedFaceDissimilarity(eigenvectors, mean, face_resized, index);
    //display_reconstructions(face_resized, eigenvectors, mean, index);
    //imshow(format("Mean: %d", index), norm_0_255(mean.reshape(1, im_width)));
    if(sim >= treshold && confidence > confidence_treshold)
        prediction = -1;
    cout << "Image: " << index << " predicted as class: " << prediction << " with confidence: " << confidence << "\n";
    return prediction;
}

double getReconstructedFaceDissimilarity(Mat W, Mat mean, Mat img, int i) {
    Mat projection = LDA::subspaceProject(W, mean, img.reshape(1,1));
    // Generate the reconstructed face back from the eigenspace.
    Mat reconstructionRow = LDA::subspaceReconstruct(W, mean, projection);
    Mat reconstructionMat = reconstructionRow.reshape(1, im_height);
    // Convert the floating-point pixels to regular 8-bit uchar.
    Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
    reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
    double dissimilarity = getSimilarity(img, reconstructedFace);
    cout << "dissimilarity: " << dissimilarity << endl;
    imshow(format("reconstructed_face_%d", i), reconstructedFace);
    return dissimilarity;
}


void display_reconstructions(Mat img, Mat W, Mat mean, int index){
// Display or save the image reconstruction at some predefined steps:
    for(int num_component = 0; num_component < min(16, W.cols); num_component++) {
        // Slice the Fisherface from the model:
        Mat ev = W.col(num_component);
        Mat projection = LDA::subspaceProject(ev, mean, img.reshape(1,1));
        Mat reconstruction = LDA::subspaceReconstruct(ev, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, img.rows));
        imshow(format("fisherface_reconstruction_%d%d", index, num_component), reconstruction);

    }
}

Mat getReconstructedFace(Mat mean, Mat eigenvectors, Mat face, int i) {
    Mat projection = LDA::subspaceProject(eigenvectors, mean, face.reshape(1,1));
    Mat reconstructionRow = LDA::subspaceReconstruct(eigenvectors, mean, projection);
    //Mat reconstructionMat = reconstructionRow.reshape(1, im_height);
    Mat reconstructedFace = norm_0_255(reconstructionRow.reshape(1, face.rows));
    //reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
    cv::resize(reconstructedFace, reconstructedFace, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
    imshow(format("reconstructed Face %d", i), reconstructedFace);
    return reconstructedFace;
}

// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity(const Mat A, const Mat B){
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        double errorL2 = norm(A, B, CV_L2);
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        cout << "WARNING: Images have a different size in 'getSimilarity()'." << endl;
        return 100000000.0;  // Return a bad value
    }
}


/** @function main */
int main( int argc, char** argv ){
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    if( !eyes_cascade2.load( eyes_cascade2_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    const char* imagename = argc > 1 ? argv[1] : "data/friends_data/gabriela/1.jpg";
    string database_file = "data/friends_db.txt";

    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    read_dataset(database_file, images, labels, ';');

    im_width = images[0].cols;
    im_height = images[0].rows;
    //im_width = 320;
    //im_height = 320;
    string saveModelPath;
    Ptr<BasicFaceRecognizer> model;
    if(model_name == "eigenfaces") {
        saveModelPath = "models/eigenfaces_model.yml";
        model = createEigenFaceRecognizer();
        treshold = 0.08;
        confidence_treshold = 200;
    }
    else if(model_name == "fisherfaces") {
        saveModelPath = "models/fisherfaces_model.yml";
        model = createFisherFaceRecognizer(16);
        treshold = 0.16;
        confidence_treshold = 200;
    }
    else {
        cout << "The model_name must be set to eigenfaces or fisherfaces" << endl;
        exit(1);
    }
    if(access( saveModelPath.c_str(), 0 ) != -1 ) {
       cout << "Loading saved model from: " << saveModelPath << endl;
       model->load(saveModelPath);
    } else {
        cout << "Training the model..." << endl;
        model->train(images, labels);
        cout << "Saving the trained model to: " << saveModelPath << endl;
        model->save(saveModelPath);
    }
    Mat img = imread(imagename);
    Mat workImg = getWorkImage(img);
    detectAndDisplay( workImg, img, model);
    cvWaitKey(0);
    return 0;
}

