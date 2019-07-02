#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H
#include <Perception/opencv.h> //always include this first!
#include <random>


class image_processing
{
public:
    image_processing();
    static std::string format_string(std::string s, int number);
    static bool check_neighbor(cv::Mat &img, int kernel_size, int x, int y);
    static double find_circle(cv::Mat img);
    static std::vector<cv::Mat> find_circle(cv::Mat img, int ball_color);
    static void remove_outliers(cv::Mat &thresholded_img);
    static float analyze_right_hand_cam(cv::Mat im, cv::Mat &visual_im, int iter_nb);
    static std::vector<cv::Mat> find_square_contours(cv::Mat &im);
    static bool count_balls_for_each_square(cv::Mat &im, int ball_col,
                                            arr &start_sq, arr &dest_sq, arr &dest_location,
                                            int iter_nb);
    static void remove_points_outside_contours(cv::Mat &thresholded_img, std::vector<cv::Mat> &contours);

private:
    static int closet_square(arr pix, std::vector<cv::Mat> sqs);
    static void count_balls_helper(std::vector<cv::Mat> &balls,
                            std::vector<cv::Mat> &balls_on_square,
                            std::vector<int> &square_cen,
                            std::vector<int> &stock,
                            double thresh_distance, unsigned long how_many_squares);
};

#endif // IMAGE_PROCESSING_H
