#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

// using namespace cv;
int main() {
    cv::Mat image;
    image = cv::imread("dataset/fountain-P11/images/0000.jpg", cv::IMREAD_COLOR);
    assert(image.data);
    cv::namedWindow("print image", cv::WINDOW_AUTOSIZE);
    cv::imshow("print image", image);
    cv::waitKey(0);
    return 0;
}