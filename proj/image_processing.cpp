#include "image_processing.h"

image_processing::image_processing()
{
}

bool image_processing::check_neighbor(cv::Mat &img, int kernel_size, int x, int y) {
    bool res = true;
    int step = 0;
    while (step < kernel_size) {
        cv::Scalar n1 = img.at<uchar>(x+step, y);
        cv::Scalar n2 = img.at<uchar>(x-step, y);
        cv::Scalar n3 = img.at<uchar>(x, y+step);
        cv::Scalar n4 = img.at<uchar>(x, y-step);
        double v1 = n1.val[0];
        double v2 = n2.val[0];
        double v3 = n3.val[0];
        double v4 = n4.val[0];

        if (v1 == 255.0 || v2 == 255.0 || v3 == 255.0 || v4 == 255.0) {
            step++;
        } else {
            res = false;
            break;
        }
    }
    return res;
}

double image_processing::find_circle(cv::Mat img) {
    cv::Mat img2;
    cv::Mat hsv_image2;
    cv::Mat thresholded_img;

    // blur
    for ( int u = 1; u < 31; u = u + 2 ){ cv::GaussianBlur( img, img2, cv::Size( u, u ), 0, 0 ); }
    cv::cvtColor(img2, hsv_image2, cv::COLOR_BGR2HSV);

    // threshold
    cv::inRange(hsv_image2, cv::Scalar(40, 40, 40), cv::Scalar(70, 255, 255), thresholded_img);

    // remove outliers
    int rows = thresholded_img.rows;
    int cols = thresholded_img.cols;
    int i_avg;
    int j_avg;
    int count;
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (check_neighbor(thresholded_img, 10, i, j)) {
                thresholded_img.at<uchar>(i, j) = 255;
                i_avg += i;
                j_avg += j;
                count += 1;
            }
            else thresholded_img.at<uchar>(i, j) = 0;
        }
    }

    std::vector<cv::Mat> contours;
    findContours(thresholded_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(img, contours, -1, color);

    printf(" detected %d objects\n", contours.size());
    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        printf("  object %d: detected %d points in the contour\n", u+1, contours[u].size().height);
        std::cout<< "  area: " <<area<<std::endl;

        i_avg = 0;
        j_avg = 0;
        count = 0;
        for (int v=0;v<contours[u].size().height;v++) {
            cv::Point point = contours[u].at<cv::Point>(v);
            i_avg += point.x;
            j_avg += point.y;
            count ++;
        }
        cv::Point mark = cv::Point(i_avg/count, j_avg/count);
        cv::drawMarker(img, mark, cv::Scalar(0, 0, 0), 16, 3, 8);

    }
    cv::imwrite("right_hand_cam.png", img);
    if (contours.size() > 1) cv::imwrite("right_hand_cam_mult.png", img);
    double area = cv::contourArea(contours[0]);
    return area;
}

void image_processing::remove_outliers(cv::Mat &thresholded_img) {
    int rows = thresholded_img.rows;
    int cols = thresholded_img.cols;
    int i_avg = 0;
    int j_avg = 0;
    int count = 0;
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (check_neighbor(thresholded_img, 10, i, j)) {
                thresholded_img.at<uchar>(i, j) = 255;
                i_avg += i;
                j_avg += j;
                count += 1;
            }
            else thresholded_img.at<uchar>(i, j) = 0;
        }
    }
}

float image_processing::analyze_right_hand_cam(cv::Mat im, cv::Mat &visual_im, bool show_im) {

    // find circle
    cv::Mat im_gray;
    cv::Mat detected_edges;
    cv::Mat dst;
    int lowThreshold = 30;
    cv::cvtColor(im, im_gray, CV_BGR2GRAY);
    cv::blur( im_gray, detected_edges, cv::Size(3,3) );
    cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*3, 3 );
    dst = cv::Scalar::all(0);
    im.copyTo( dst, detected_edges);

    // contours
    cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);

    struct less_than_key
    {
        inline bool operator() (const cv::Mat& struct1, const cv::Mat& struct2)
        {
            cv::Point gripper_center = cv::Point(320, 200);
            cv::Moments m1 = cv::moments(struct1);
            cv::Moments m2 = cv::moments(struct2);

            cv::Point p1 = cv::Point(int(m1.m10 / m1.m00), int(m1.m01 / m1.m00));
            cv::Point p2 = cv::Point(int(m2.m10 / m2.m00), int(m2.m01 / m2.m00));

            double dis1 = pow(p1.x-gripper_center.x, 2) + pow(p1.y-gripper_center.y, 2);
            double dis2 = pow(p2.x-gripper_center.x, 2) + pow(p2.y-gripper_center.y, 2);

            return dis1 < dis2;
        }
    };

    std::vector<cv::Mat> contours;
    std::vector<cv::Mat> interested_contours;
    std::vector<cv::Mat> only_big_contours;
    findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // remove small contours
    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        if (area>10) only_big_contours.push_back(contours[u]);
    }

    // remove square contours
    for (uint u=0;u<only_big_contours.size();u++) {
        double peri = cv::arcLength(only_big_contours[u], true);
        cv::Mat vertices;
        cv::approxPolyDP(only_big_contours[u], vertices, 0.04*peri, true);
        if (vertices.size().height > 5) interested_contours.push_back(only_big_contours[u]);
    }

    std::sort(interested_contours.begin(), interested_contours.end(), less_than_key());

    // draw contours
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(im, contours, -1, color);
    cv::Moments m = cv::moments(interested_contours[0]);
    int cX = int(m.m10 / m.m00);
    int cY = int(m.m01 / m.m00);
    cv::Point p = cv::Point(cX, cY);
    cv::drawMarker(im, p, cv::Scalar(0, 200, 100), 16, 3, 8);

    // draw grippers
    cv::Point mark1 = cv::Point(240, 105);
    cv::Point mark2 = cv::Point(482, 106);
    cv::Point mark3 = cv::Point(320, 200);
    cv::drawMarker(im, mark3, cv::Scalar(0, 100, 100), 16, 3, 8);

    // compute how much devitation from gripper
    printf("ball center is at %d %d\n", cX, cY);
    float slope = (float)(mark1.y-mark2.y)/(mark1.x-mark2.x);
    float distance_to_grippers = (slope*cX-cY+mark2.y-slope*mark2.x)/sqrt(slope*slope+1);
    printf("distance to gippers is %f\n", distance_to_grippers);

    // documenting abnormal situation
    if (abs(distance_to_grippers) <= 60) {
        cv::imwrite("rhc_analyzed_abnormal.png", im);
        cv::imwrite("rhc_edge_abnormal.png", dst);
    }

    cv::imwrite("rhc_analyzed.png", im);
    cv::imwrite("rhc_edge.png", dst);
    visual_im = im.clone();

    if (show_im) {
        cv::imshow("noname", im);
        cv::imshow("edge", dst);
        cv::waitKey(0);
    }
    return distance_to_grippers;

}

std::vector<cv::Mat> image_processing::find_square_contours(cv::Mat &im) {
    // thresholded
    cv::Mat im_blurred;
    cv::Mat hsv_im;
    cv::Mat thresholded_img;
    cv::blur(im, im_blurred, cv::Size(3, 3));
    cv::cvtColor(im_blurred, hsv_im, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_im, cv::Scalar(17, 100, 100), cv::Scalar(37, 255, 255), thresholded_img);//yellow
    remove_outliers(thresholded_img);

    // contour detection
    std::vector<cv::Mat> contours;
    findContours(thresholded_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // remove small contours
    std::vector<cv::Mat> big_contours;
    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        if (area>10) big_contours.push_back(contours[u]);
    }

    // approximate contours
    for (uint u=0;u<big_contours.size();u++) {
        double peri = cv::arcLength(big_contours[u], true);
        cv::Mat vertices;
        double t = 0.01;
        cv::approxPolyDP(big_contours[u], vertices, t*peri, true);

        printf("SQUARE ANALYSIS: %f, %d\n", t, vertices.size().height);
        while (vertices.size().height > 4) {
            t += 0.001;
            cv::approxPolyDP(big_contours[u], vertices, t*peri, true);
        }
        printf("SQUARE ANALYSIS: %f, %d\n", t, vertices.size().height);
        printf("SQUARE ANALYSIS: done\n\n");
        big_contours[u] = vertices;
    }

    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(im, big_contours, -1, color);

    return big_contours;
}

void image_processing::remove_points_outside_contours(cv::Mat &thresholded_img, std::vector<cv::Mat> &contours) {
    int rows = thresholded_img.rows;
    int cols = thresholded_img.cols;
    std::vector<std::vector<int>> points_inside_contours;
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (thresholded_img.at<uchar>(i, j) > 0) {
                for (uint u=0;u<contours.size();u++) {
                    double d = cv::pointPolygonTest(contours[u], cv::Point(j, i), true);
                    if (d >= 0) {
                        std::vector<int> valid = {i, j};
                        points_inside_contours.push_back(valid);
                    }
                }
                thresholded_img.at<uchar>(i, j) = 0;
            }
        }
    }

    for (uint u=0; u<points_inside_contours.size(); u++) {
        int i = points_inside_contours[u][0];
        int j = points_inside_contours[u][1];
        thresholded_img.at<uchar>(i, j) = 255;
    }
}
