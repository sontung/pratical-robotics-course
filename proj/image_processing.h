#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>


class image_processing
{
public:
    image_processing();
    static bool check_neighbor(cv::Mat &img, int kernel_size, int x, int y);
    static double find_circle(cv::Mat img);
    static void remove_outliers(cv::Mat &thresholded_img);
    static float analyze_right_hand_cam(cv::Mat im, bool show_im=false);
    static std::vector<cv::Mat> find_square_contours(cv::Mat &im);
    static void remove_points_outside_contours(cv::Mat &thresholded_img, std::vector<cv::Mat> &contours);
};

#endif // IMAGE_PROCESSING_H
