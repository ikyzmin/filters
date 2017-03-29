#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

const int kMenuTabs = 12;

const char *menu[] =
        {
                "1 - Apply linear filter",
                "2 - Apply blur(...)",
                "3 - Apply box filter",
                "4 - Apply medianBlur(...)",
                "5 - Apply GaussianBlur(...)",
                "6 - Apply erode(...)",
                "7 - Apply dilate(...)",
                "8 - Apply Sobel(...)",
                "9 - Apply Laplacian(...)",
                "10 - Apply Canny(...)",
                "11 - Apply calcHist(...)",
                "12 - Apply equalizeHist(...)"
        };

const char *winNames[] =
        {
                "Initial image",
                "filter2d",
                "blur",
                "box filter",
                "medianBlur",
                "GaussianBlur",
                "erode",
                "dilate",
                "Sobel",
                "Laplacian",
                "Canny",
                "calcHist",
                "equalizeHist"
        };

const int escCode = 27;

void printMenu() {
    int i = 0;
    printf("Menu items:\n");
    for (i; i < kMenuTabs; i++) {
        printf("\t%s\n", menu[i]);
    }
    printf("\n");
}

void loadImage(Mat &srcImg) {
    do {
        srcImg = imread("../cube.jpg", 1);
    } while (srcImg.data == 0);
    printf("The image was succesfully read\n\n");
}

void chooseMenuTab(int &activeMenuTab, Mat &srcImg) {
    int tabIdx;
    while (true) {
        // print menu items
        printMenu();
        // get menu item identifier to apply operation
        printf("Input item identifier to apply operation: ");
        scanf("%d", &tabIdx);
        loadImage(srcImg);
        if (tabIdx >= 1 && tabIdx < kMenuTabs &&
                   srcImg.data == 0) {
            // read image
            printf("The image should be read to applym operation!\n");
            loadImage(srcImg);
        } else if (tabIdx >= 1 && tabIdx < kMenuTabs) {
            activeMenuTab = tabIdx;
            break;
        }
    }
}

Mat prepareForDilateAndErode(Mat src) {
    Mat grayscaled, binary;
    cvtColor(src, grayscaled, COLOR_RGB2GRAY, 0);
    threshold(grayscaled, binary, 100, 255, CV_THRESH_BINARY);
    return binary;
}

int applyOperation(const Mat &src, const int operationIdx) {
    char key = -1;
    Mat dst;
    Mat prepared;

    const float kernelData[] = {-0.1f, 0.2f, -0.1f,
                                0.2f, 3.0f, 0.2f,
                                -0.1f, 0.2f, -0.1f};
    const Mat kernel(3, 3, CV_32FC1, (float *) kernelData);
    Mat element = Mat();
    switch (operationIdx) {
        case 1:
            filter2D(src, dst, -1, kernel);
            break;

        case 2:
            blur(src, dst, Size(5, 5));
            break;
        case 3:
            boxFilter(src, dst, -1, Size(5, 5));
            break;
        case 4:
            GaussianBlur(src, dst, Size(5, 5), 10.1f, 10.1f);
            break;
        case 5:
            medianBlur(src, dst, 5);
            break;
        case 6:
            prepared = prepareForDilateAndErode(src);
            erode(prepared, dst, element);
            break;
        case 7:
            prepared = prepareForDilateAndErode(src);
            dilate(prepared, dst, element);
            break;

    }
    // show initial image
    namedWindow(winNames[0], 1);
    imshow(winNames[0], src);
    // show processed image
    namedWindow(winNames[operationIdx]);
    imshow(winNames[operationIdx], dst);
    IplImage *img = cvLoadImage("../cube.jpg");
    cvNamedWindow(winNames[operationIdx]);
    cvWaitKey();
    return 0;
}


int main() {

    Mat srcImg; // исходное изображение
    char ans;
    int activeMenuTab = -1;
    do {
        // вызов функции выбора пункта меню
        chooseMenuTab(activeMenuTab, srcImg);
        // применение операций
        applyOperation(srcImg, activeMenuTab);
        // вопрос о необходимости продолжения
        printf("Do you want to continue? ESC - exit\n");
        // ожидание нажатия клавиши
        ans = waitKey();
    } while (ans != escCode);
    destroyAllWindows(); // закрытие всех окон
    srcImg.release(); // освобожение памяти
    return 0;
}