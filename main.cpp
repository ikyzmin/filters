#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

const int kMenuTabs = 13;
const int ddepth = CV_16S;

const char
        *xGradWinName = "Gradient in the direction Ox",
        *yGradWinName = "Gradient in the direction Oy",
        *gradWinName = "Gradient";

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
    Mat laplacianImg, laplacianImgAbs;
    Mat xGrad, yGrad,xGradAbs, yGradAbs, grad,bgrChannels[3];
    double lowThreshold = 70, uppThreshold = 260;
    Mat  bHist, gHist, rHist, histImg;
    float range[] = {0.0f, 256.0f};
    const float* histRange = { range };
    int kBins = 256; // количество бинов гистограммы
    // равномерное распределение интервала по бинам
    bool uniform = true;
    // запрет очищения перед вычислением гистограммы
    bool accumulate = false;
    // размеры для отображения гистограммы
    int histWidth = 512, histHeight = 400;
    // количество пикселей на бин
    int binWidth = cvRound((double)histWidth / kBins);
    int i, kChannels = 3;
    int channels[] = {0};
    Scalar colors[] = {Scalar(255, 0, 0),
                       Scalar(0, 255, 0), Scalar(0, 0, 255)};

    double alpha = 0.5, beta = 0.5;

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
        case 8:
            GaussianBlur(src, dst, Size(3, 3), 0, 0);
            prepared = prepareForDilateAndErode(dst);
            Sobel(prepared, xGrad, ddepth, 1, 0);
            Sobel(prepared, yGrad, ddepth, 0, 1);
            convertScaleAbs(xGrad, xGradAbs);
            convertScaleAbs(yGrad, yGradAbs);
            addWeighted(xGradAbs, alpha, yGradAbs, beta, 0, grad);
            namedWindow(xGradWinName, CV_WINDOW_AUTOSIZE);
            namedWindow(yGradWinName, CV_WINDOW_AUTOSIZE);
            namedWindow(gradWinName, CV_WINDOW_AUTOSIZE);
            imshow(xGradWinName, xGradAbs);
            imshow(yGradWinName, yGradAbs);
            imshow(gradWinName, grad);
            break;
        case 9:
            GaussianBlur(src, dst, Size(3, 3), 0, 0);
            prepared = prepareForDilateAndErode(dst);
            Laplacian(prepared, laplacianImg, ddepth);
            convertScaleAbs(laplacianImg, laplacianImgAbs);
            dst = laplacianImgAbs;
            break;
        case 10:
            blur(src, dst, Size(3,3));
            prepared = prepareForDilateAndErode(dst);
            Canny(prepared, dst, lowThreshold, uppThreshold);
            break;
        case 11:
            // выделение каналов изображения
            split(src, bgrChannels);
            // вычисление гистограммы для каждого канала
            calcHist(&bgrChannels[0], 1, 0, Mat(), bHist, 1, &kBins, &histRange, uniform, accumulate);
            calcHist(&bgrChannels[1], 1, 0, Mat(), gHist, 1, &kBins, &histRange, uniform, accumulate);
            calcHist(&bgrChannels[2], 1, 0, Mat(), rHist, 1, &kBins, &histRange, uniform, accumulate);
            // построение гистограммы
            histImg = Mat(histHeight, histWidth, CV_8UC3,
                          Scalar(0, 0, 0));
            // нормализация гистограмм в соответствии с размерам
            // окна для отображения
            normalize(bHist, bHist, 0, histImg.rows,
                      NORM_MINMAX, -1, Mat());
            normalize(gHist, gHist, 0, histImg.rows,
                      NORM_MINMAX, -1, Mat());
            normalize(rHist, rHist, 0, histImg.rows,
                      NORM_MINMAX, -1, Mat());
            // отрисовка ломаных
            for (i = 1; i < kBins; i++)
            {
                line(histImg, Point(binWidth * (i-1),
                                    histHeight-cvRound(bHist.at<float>(i-1))) ,
                     Point(binWidth * i,
                           histHeight-cvRound(bHist.at<float>(i)) ),
                     colors[0], 2, 8, 0);
                line(histImg, Point(binWidth * (i-1),
                                    histHeight-cvRound(gHist.at<float>(i-1))) ,
                     Point(binWidth * i,
                           histHeight-cvRound(gHist.at<float>(i)) ),
                     colors[1], 2, 8, 0);
                line(histImg, Point(binWidth * (i-1),
                                    histHeight-cvRound(rHist.at<float>(i-1))) ,
                     Point(binWidth * i,
                           histHeight-cvRound(rHist.at<float>(i)) ),
                     colors[2], 2, 8, 0);
            }
            dst = histImg;
            break;
        case 12:
            cvtColor(src, prepared, CV_RGB2GRAY);
            equalizeHist(prepared, dst);
            break;
    }
    if (operationIdx != 8) {
    // show initial image
    namedWindow(winNames[0], 1);
        if (operationIdx==12) {
            imshow(winNames[0], prepared);
        }else{
            imshow(winNames[0], src);
        }
    // show processed image
    namedWindow(winNames[operationIdx]);
    imshow(winNames[operationIdx], dst);
    cvNamedWindow(winNames[operationIdx]);
}
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