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
Mat detectAndDisplay( Mat workingImg, Mat originalImg, Ptr<BasicFaceRecognizer> model, vector<Rect> *faces_vect);
static void read_dataset(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
Mat scaleImg(Mat img);
Mat getWorkImage(Mat img);
Mat preprocessImg(Mat faceImg, int faceId);
string getName(int prediction);
int predictFace(Mat face, Ptr<BasicFaceRecognizer> model, int index);

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

static void read_dataset(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
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
Mat detectAndDisplay( Mat workingImg, Mat originalImg, Ptr<BasicFaceRecognizer> model, vector<Rect> *faces_vect){
    int flags = 0|CASCADE_SCALE_IMAGE;
    Size minFeatureSize(20, 20);
    float searchScaleFactor = 1.1f;
    int minNeighbors = 6;
    std::vector<Rect> faces;

    face_cascade.detectMultiScale( workingImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize );
    *faces_vect = faces;
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = workingImg(faces[i]);
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( originalImg, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        std::vector<Rect> eyes;
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, searchScaleFactor, 4, flags, Size(1,1) );

        for ( size_t j = 0; j < 2; j++ ){
            Point2f eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            //circle( originalImg, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }

        Mat finalImg = preprocessImg(faceROI);

        int prediction = predictFace(faceROI, model, i);
        Mat W = model->getEigenVectors();
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

int main(int argc, const char *argv[]) {
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    if( !eyes_cascade2.load( eyes_cascade2_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    const char* imagename = argc > 1 ? argv[1] : "data/friends_data/aleksandra/1.jpg";
    string database_file = "data/friends_db.txt";
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    try {
        read_dataset(database_file, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << database_file << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    im_width = images[0].cols;
    im_height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::BasicFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:

    Mat img = imread(imagename);
    Mat workImg = getWorkImage(img);
    std::vector<Rect> faces;
    Mat predict_img;
    predict_img = detectAndDisplay( workImg, img, model, &faces);

    //cvWaitKey(0);

    int predictedLabel = model->predict(workImg);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    //string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    //cout << result_message << endl;
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();
    // Display or save:
    if(argc == 2) {
        imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    } else {
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }
    // Display or save the first, at most 16 Fisherfaces:
    for (int i = 0; i < min(16, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Bone colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
        // Display or save:
        if(argc == 2) {
            imshow(format("fisherface_%d", i), cgrayscale);
        } else {
            imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }
    // Display or save the image reconstruction at some predefined steps:
    for(int num_component = 0; num_component < min(16, W.cols); num_component++) {
        // Slice the Fisherface from the model:
        Mat ev = W.col(num_component);
        Mat projection = LDA::subspaceProject(ev, mean, images[0].reshape(1,1));
        Mat reconstruction = LDA::subspaceReconstruct(ev, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        if(argc == 2) {
            imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
        } else {
            imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
        }
    }
    // Display if we are not writing to an output folder:
    if(argc == 2) {
        waitKey(0);
    }
    return 0;
}
