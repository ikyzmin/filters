#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat source;
    Mat destination;
    Mat blurImg;
    Mat boxFilterImg;
    Mat gaussianBlurImg;
    Mat medianBlurImg;
    const float kernelData[] = {-0.1f, 0.2f, -0.1f,
                                0.2f, 3.0f, 0.2f,
                                -0.1f, 0.2f, -0.1f};
    const Mat kernel(3, 3, CV_32FC1, (float *)kernelData);
    source = imread("../cube.jpg",1);
    namedWindow("Cube",WINDOW_AUTOSIZE);
    imshow("Cube",source);
    filter2D(source,destination,-1,kernel);
    namedWindow("Filtered",WINDOW_AUTOSIZE);
    imshow("Filtered",destination);

    blur(source,blurImg,Size(5,5));

    namedWindow("blur",WINDOW_AUTOSIZE);
    imshow("blur",blurImg);

    boxFilter(source,boxFilterImg,-1,Size(5,5));

    namedWindow("box filter",WINDOW_AUTOSIZE);
    imshow("box filter",boxFilterImg);

    GaussianBlur(source,gaussianBlurImg,Size(5,5),10.1f,10.1f);

    namedWindow("Gaussian blur",WINDOW_AUTOSIZE);
    imshow("Gaussian blur",gaussianBlurImg);

    medianBlur(source,medianBlurImg,5);

    namedWindow("Median blur",WINDOW_AUTOSIZE);
    imshow("Median blur",gaussianBlurImg);
    Mat  erodeImg, dilateImg, element,grayscaled,binary;


    cvtColor(source,grayscaled,COLOR_RGB2GRAY,0);
    threshold(grayscaled,binary,100,255,CV_THRESH_BINARY);

    element = Mat();
    erode(binary, erodeImg, element);
    dilate(binary, dilateImg, element);

    namedWindow("Erode",WINDOW_AUTOSIZE);
    imshow("Erode",erodeImg);

    namedWindow("Dilate",WINDOW_AUTOSIZE);
    imshow("Dilate",dilateImg);


    grayscaled.release();
    blurImg.release();
    boxFilterImg.release();
    gaussianBlurImg.release();
    medianBlurImg.release();
    binary.release();
    dilateImg.release();
    erodeImg.release();
    destination.release();
    source.release();
    waitKey(0);
    return 0;
}