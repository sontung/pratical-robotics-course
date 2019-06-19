#include <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>
#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <numeric>
#include <algorithm>

#include "kine.cpp"


bool check_neighbor(cv::Mat &img, int kernel_size, int x, int y) {
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

double find_circle(cv::Mat img) {
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

void remove_outliers(cv::Mat &thresholded_img) {
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

arr compute_world_coord(int cx, int cy, cv::Mat &depth, rai::KinematicWorld &kine_world) {
    // set the intrinsic camera parameters
    rai::Frame *cameraFrame = kine_world.getFrameByName("camera");
    arr Fxypxy = {538.273, 544.277, 307.502, 249.954};
    Fxypxy /= 0.982094;

    arr image_coord = {(float)cx, (float)cy, depth.at<float>(cy, cx)};

    printf("WC: transforming %d %d %f\n", cx, cy, depth.at<float>(cy, cx));

    // camera coordinates
    depthData2point(image_coord, Fxypxy); //transforms the point to camera xyz coordinates

    // world coordinates
    cameraFrame->X.applyOnPoint(image_coord); //transforms into world coordinates

    return image_coord;
}

std::vector<arr> perception(rai::KinematicWorld &kine_world) {
    rai::Frame *cameraFrame = kine_world.getFrameByName("camera");
    std::vector<arr> world_coordinates;

    // grabbing images from robot
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();
    cv::Mat img = CV(_rgb.get()).clone();
    cv::Mat depth_map = CV(_depth.get()).clone();
    cv::imwrite("rgb_head.png", img);
    if(img.rows != depth_map.rows) return world_coordinates;

    // set the intrinsic camera parameters
    arr Fxypxy = {538.273, 544.277, 307.502, 249.954};
    Fxypxy /= 0.982094;

    cv::Mat img2;
    cv::Mat hsv_image2;

    cv::Mat thresholded_img;

    // blur
    for ( int u = 1; u < 31; u = u + 2 ){ cv::GaussianBlur( img, img2, cv::Size( u, u ), 0, 0 ); }
    cv::cvtColor(img2, hsv_image2, cv::COLOR_BGR2HSV);

    // threshold
    cv::inRange(hsv_image2, cv::Scalar(40, 100, 100), cv::Scalar(70, 255, 255), thresholded_img);

    // remove outliers
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

    std::vector<cv::Mat> contours;
    findContours(thresholded_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(img, contours, -1, color);

    printf("detected %d objects\n", contours.size());

    // image coordinates -> world coordinates
    std::vector<float> all_depths;
    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        if (area <= 0) continue;
        i_avg = 0;
        j_avg = 0;
        count = 0;
        printf("  object %d: detected %d points in the contour\n", u+1, contours[u].size().height);
        std::cout<< "  area: " <<area<<std::endl;
        for (int v=0;v<contours[u].size().height;v++) {
            cv::Point point = contours[u].at<cv::Point>(v);
            i_avg += point.x;
            j_avg += point.y;
            count ++;
            float current_depth = depth_map.at<float>(point.y, point.x);
            all_depths.push_back(current_depth);
        }
        cv::Point mark = cv::Point(i_avg/count, j_avg/count);
        cv::drawMarker(img, mark, cv::Scalar(0, 0, 0), 16, 3, 8);

        float avg_depth = std::accumulate(all_depths.begin(), all_depths.end(), 0.0f) / count;
        std::nth_element(all_depths.begin(),
                         all_depths.begin() + all_depths.size()/2,
                         all_depths.end());
        float median_depth = all_depths[all_depths.size()/2];

        arr image_coord = {(float)i_avg/count, (float)j_avg/count, depth_map.at<float>(j_avg/count, i_avg/count)};
        cout<<"center depth "<<image_coord<<endl;
        image_coord = {(float)i_avg/count, (float)j_avg/count, avg_depth};
        cout<<"average depth "<<image_coord<<endl;
        image_coord = {(float)i_avg/count, (float)j_avg/count, median_depth};
        cout<<"median depth "<<image_coord<<endl;


        // camera coordinates
        depthData2point(image_coord, Fxypxy); //transforms the point to camera xyz coordinates

        // world coordinates
        cameraFrame->X.applyOnPoint(image_coord); //transforms into world coordinates
        image_coord.append((double)i_avg/count);
        image_coord.append((double)j_avg/count);

        world_coordinates.push_back(image_coord);
    }


    cv::imwrite("rgb_detected.png", img);
    printf("\n");
    return world_coordinates;
}

void print_with_color(std::string text, int color=32, bool nl_included=true) {
    std::ostringstream stringStream;
    if (nl_included) stringStream << "\033[0;"<<color<<"m"<<text<<"\033[0m\n";
    else stringStream << "\033[0;"<<color<<"m"<<text<<"\033[0m";
    cout<<stringStream.str();
}

float analyze_right_hand_cam(cv::Mat im, bool show_im=false) {

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
        if (area>0) only_big_contours.push_back(contours[u]);
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

    if (show_im) {
        cv::imshow("noname", im);
        cv::imshow("edge", dst);
        cv::waitKey(0);
    }
    return distance_to_grippers;

}

arr analyze_scene(rai::KinematicWorld &kine_world,
                  arr ball_pixel_coord,
                  bool show_img=false) {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();
    cv::Mat im = CV(_rgb.get()).clone();
    cv::Mat depth_map = CV(_depth.get()).clone();

    // thresholded
    cv::Mat im_blurred;
    cv::Mat hsv_im;
    cv::Mat thresholded_img;
    cv::blur(im, im_blurred, cv::Size(3, 3));
    cv::cvtColor(im_blurred, hsv_im, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_im, cv::Scalar(40, 100, 100), cv::Scalar(70, 255, 255), thresholded_img);//green
    cv::inRange(hsv_im, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255), thresholded_img);//yellow
    remove_outliers(thresholded_img);

    // contour detection
    std::vector<cv::Mat> contours;
    findContours(thresholded_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );

    // remove small contours
    std::vector<cv::Mat> big_contours;
    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        if (area>10) big_contours.push_back(contours[u]);
    }

    // approximate contours
    for (uint u=0;u<big_contours.size();u++) {
        double area = cv::contourArea(big_contours[u]);
        double peri = cv::arcLength(big_contours[u], true);
        cv::Mat vertices;
        cv::approxPolyDP(big_contours[u], vertices, 0.04*peri, true);
        printf("SCENE ANALYSIS: area %f has old size=(%d, %d), new size=(%d, %d)\n", area,
               big_contours[u].size().width, big_contours[u].size().height,
               vertices.size().width, vertices.size().height);
        big_contours[u] = vertices;
    }

    // determine target square center
    double biggest_distance = -1.0;
    uint idx = 0;
    double avg_area = 0.0;
    for (uint u=0;u<big_contours.size();u++) {
        double area = cv::contourArea(big_contours[u]);
        avg_area += area;
    }
    avg_area /= big_contours.size();
    for (uint u=0;u<big_contours.size();u++) {
        double area = cv::contourArea(big_contours[u]);
        cv::Moments m = cv::moments(big_contours[u]);
        int cX = int(m.m10 / m.m00);
        int cY = int(m.m01 / m.m00);
        double distance_to_ball_target = pow(cX-ball_pixel_coord(0), 2) + pow(cY-ball_pixel_coord(1), 2);
        if (distance_to_ball_target > biggest_distance && area > avg_area/2.0) {
            biggest_distance = distance_to_ball_target;
            idx = u;
        }
    }

    // determine target square center world coord
    cv::Moments m = cv::moments(big_contours[idx]);
    int cX = int(m.m10 / m.m00);
    int cY = int(m.m01 / m.m00);
    cv::Point p = cv::Point(cX, cY);
    cv::Scalar color_center( rand()&255, rand()&255, rand()&255 );
    cv::drawMarker(im, p, color_center, 16, 3, 8);
    arr wc = compute_world_coord(cX, cY, depth_map, kine_world);

    // mark ball
    p = cv::Point(ball_pixel_coord(0), ball_pixel_coord(1));
    cv::drawMarker(im, p, color, 16, 3, 8);

    drawContours(im, big_contours, -1, color);

    // documenting abnormal behaviours
    if (big_contours.size() > 2) {
        printf("SCENE ANALYSIS: abnormal situation, documenting...\n");
        cv::imwrite("abnormal_scene_analysis.png", im);
    }

    if (show_img) {
        cv::imshow("image", im);
        cv::imshow("blur", im_blurred);
        cv::imshow("threshold", thresholded_img);
        cv::waitKey(0);
    }

    cv::imwrite("rgb_scene.png", im);

    return wc;

}

std::vector<cv::Mat> find_square_contours(cv::Mat &im) {
    // thresholded
    cv::Mat im_blurred;
    cv::Mat hsv_im;
    cv::Mat thresholded_img;
    cv::blur(im, im_blurred, cv::Size(3, 3));
    cv::cvtColor(im_blurred, hsv_im, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_im, cv::Scalar(40, 100, 100), cv::Scalar(70, 255, 255), thresholded_img);//green
    cv::inRange(hsv_im, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255), thresholded_img);//yellow
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
        double t = 0.1;
        cv::approxPolyDP(big_contours[u], vertices, t*peri, true);

        while (vertices.size().height > 4) {
            t -= 0.01;
            cv::approxPolyDP(big_contours[u], vertices, t*peri, true);
        }

        big_contours[u] = vertices;
    }

    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(im, big_contours, -1, color);

    return big_contours;
}

void remove_points_outside_contours(cv::Mat &thresholded_img, std::vector<cv::Mat> &contours) {
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

// analyze to return which ball moves, 0-red, 1-green
int analyze_video(cv::Mat &first_frame) {
    int res = -1;

    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();

    cv::Mat curr_frame;
    cv::Mat delta;
    cv::Mat delta2;

    cv::Mat first_gray;
    cv::cvtColor(first_frame, first_gray, cv::COLOR_BGR2GRAY);
    cv::blur(first_gray, first_gray, cv::Size(3, 3));
    cv::Mat curr_gray;
    cv::Mat curr_hsv;
    cv::Mat green_img;
    cv::Mat red_img;

    std::vector<cv::Mat> square_contours = find_square_contours(first_frame);

    // grab current frame
    curr_frame = CV(_rgb.get()).clone();
    if (curr_frame.total() <= 0) return res;
    cv::blur(curr_frame, curr_frame, cv::Size(3, 3));
    cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_frame, curr_hsv, cv::COLOR_BGR2HSV);

    // detect moving pixels
    cv::absdiff(first_gray, curr_gray, delta);
    cv::threshold(delta, delta2, 25, 255, cv::THRESH_BINARY);
    remove_points_outside_contours(delta2, square_contours);

    std::vector<cv::Mat> contours;
    findContours(delta2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(curr_frame, contours, -1, color);

    // check if moving pixels are balls
    cv::inRange(curr_hsv, cv::Scalar(40, 100, 100), cv::Scalar(70, 255, 255), green_img);
    cv::inRange(curr_hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), red_img);

    // remove moving pixels that is not green or red
    for (uint u=0; u<contours.size(); u++) {
        double area = cv::contourArea(contours[u]);
        if (area > 0) {
            cv::Moments m = cv::moments(contours[u]);
            int cX = int(m.m10 / m.m00);
            int cY = int(m.m01 / m.m00);
            if (green_img.at<uchar>(cY, cX) > 0) {
                res = 1;
            } else if (red_img.at<uchar>(cY, cX) > 0) {
                res = 0;
            }
        }

    }

    cv::imshow("first frame", first_frame);
    cv::imshow("diff from first gray thresholded", delta2);
    cv::imshow("curr frame", curr_frame);
    cv::imshow("green", green_img);
    cv::waitKey(1);

    return res;
}

float grab_right_hand_cam() {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/cameras/right_hand_camera/image", "");

    while (1) {
        byteA img = _rgb.get();
        cv::Mat rgb = cv::Mat(img.d0, img.d1, CV_8UC4, img.p);

        if (rgb.total() > 0) {
            //return find_circle(rgb);
            cv::imwrite("right_hand_cam.png", rgb);
            return analyze_right_hand_cam(rgb);
        }
    }
}

int main(int argc,char **argv){
    bool motion = true;
    bool testing_trivial = true;

    // basic setup
    bool first_time_run = true;
    rai::KinematicWorld C = setup_kinematic_world();
    RobotOperation B(C);
    const arr q_home = C.getJointState();
    arr q_current;
    if (!testing_trivial) rai::initCmdLine(argc,argv);
    cout <<"joint names: " <<B.getJointNames() <<endl;

    while (1) {
        if (!testing_trivial) {
            arr y, J;
            if (first_time_run) {
                cout<<"will send motion. confirm?"<<endl;
                pause_program();
                first_time_run = false;
            }

            // homing
            print_with_color("moving to home position");
            if (motion) B.moveHard(q_home);
            cout<<"q home: "<<q_home<<endl;
            pause_program_auto();

            // make sure gripper is open
            while (1) {
                q_current = B.getJointPositions();
                if (!B.getGripperGrabbed("right")) {
                    q_current(-2) = 0;
                    if (motion) B.moveHard(q_current);
                    break;
                }
                pause_program_auto();
            }
            print_with_color("gripper status: ", 32, false);
            printf("%d\n", B.getGripperOpened("right"));

            // perceiving target point
            print_with_color("doing perception");
            std::vector<arr> res;
            std::vector<arr> targets;
            std::vector<arr> pixel_coords;
            while(1) {
                res = perception(C);
                if (res.size()>0) break;
            }
            for (uint i=0;i<res.size();i++) {
                arr coords = {res[i](0), res[i](1), res[i](2)};
                arr pix = {res[i](3), res[i](4)};
                targets.push_back(coords);
                pixel_coords.push_back(pix);
                cout<<"detected: "<<res[i]<<endl;
            }
            pause_program_auto();

            // ik to target
            print_with_color("doing motion");

            targets[0](2) += 0.15;
            q_current = ik_compute(C, B, targets[0], q_home, motion);
            pause_program_auto();

            // analyzing right hand cam
            print_with_color("calibrating ...");
            float distance_to_grippers = grab_right_hand_cam();
            while (1) {
                if (abs(distance_to_grippers) > 80) {
                    if (distance_to_grippers > 0) targets[0](1) += 0.007;
                    else if (distance_to_grippers < 0) targets[0](1) -= 0.007;
                } else if (abs(distance_to_grippers) < 70) {
                    if (distance_to_grippers > 0) targets[0](1) -= 0.007;
                    else if (distance_to_grippers < 0) targets[0](1) += 0.007;
                }
                q_current = ik_compute(C, B, targets[0], q_home, motion);
                pause_program_auto();
                distance_to_grippers = grab_right_hand_cam();
                if (abs(distance_to_grippers) < 80 && abs(distance_to_grippers) >= 70) break;
            }
            print_with_color("calibrating done.");

            targets[0](2) = 0.81;
            q_current = ik_compute(C, B, targets[0], q_home, motion);
            cout<<"Ready to grab at "<< q_current<<endl;
            pause_program_auto(9);

            // closing finger
            while (1) {
                q_current = B.getJointPositions();
                if (!B.getGripperGrabbed("right")) {
                    q_current(-2) = 1;
                    if (motion) B.moveHard(q_current);
                    break;
                }
                pause_program_auto();
            }

            // go up
            targets[0](2) += 0.15;
            q_current = ik_compute_with_grabbing(C, B, targets[0], q_home, motion);
            pause_program_auto();

            // go to target bin
            cout<<"now go to target bin"<<endl;
            arr bin_target = analyze_scene(C, pixel_coords[0]);
            bin_target(2) += 0.15;
            q_current = ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
            pause_program_auto();
            bin_target(2) = 0.83;
            q_current = ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
            pause_program_auto();

            // release ball
            q_current(-2) = 0;
            if (motion) B.moveHard(q_current);
            pause_program_auto();

            // homing again
            print_with_color("moving to home position");
            if (motion) B.moveHard(q_home);
            pause_program_auto();

        }
        else {
            while (1) {

                // grabbing images from robot
                Var<byteA> _rgb;
                Var<floatA> _depth;
                RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
                _depth.waitForNextRevision();

                cv::Mat first_frame;
                first_frame = CV(_rgb.get()).clone();
                int r;
                r = analyze_video(first_frame);
                while (r == -1) {
                    first_frame = CV(_rgb.get()).clone();
                    if (first_frame.total() > 0) r = analyze_video(first_frame);
                    sleep(1);
                }
                printf("Ball = %d\n", r);

                pause_program_auto();
            }
        }
    }
    return 0;
}

