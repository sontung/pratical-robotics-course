#include <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>
#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>

#include <iostream>
#include <numeric>
#include <algorithm>

#include "kinematics.h"
#include "image_processing.h"


void pause_program() {
    cout << "Press to continue"<<endl;
    std::cin.get();
}

void pause_program_auto(uint time=5) {
    cout<<"Waiting to complete motion"<<endl;
    sleep(time);
}

void print_with_color(std::string text, int color=32, bool nl_included=true) {
    std::ostringstream stringStream;
    if (nl_included) stringStream << "\033[0;"<<color<<"m"<<text<<"\033[0m\n";
    else stringStream << "\033[0;"<<color<<"m"<<text<<"\033[0m";
    cout<<stringStream.str();
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

std::vector<arr> perception(rai::KinematicWorld &kine_world, int ball_color, arr pixel_location) {
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
    if (ball_color==0) cv::inRange(hsv_image2, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), thresholded_img);//red
    else if (ball_color==1) cv::inRange(hsv_image2, cv::Scalar(35, 100, 100), cv::Scalar(55, 255, 255), thresholded_img);//green

    // remove outliers
    int rows = thresholded_img.rows;
    int cols = thresholded_img.cols;
    int i_avg = 0;
    int j_avg = 0;
    int count = 0;
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (image_processing::check_neighbor(thresholded_img, 10, i, j)) {
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

    printf("detected %d balls of color %d\n", contours.size(), ball_color);

    // image coordinates -> world coordinates
    double closest_distance = -1;
    uint idx = 0;
    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        if (area <= 0) continue;
        cv::Moments m = cv::moments(contours[u]);
        int cX = int(m.m10 / m.m00);
        int cY = int(m.m01 / m.m00);
        double distance = pow(cX-pixel_location(0), 2) + pow(cY-pixel_location(1), 2);
        if (distance < closest_distance || closest_distance < 0) {
            closest_distance = distance;
            idx = u;
        }
    }


    std::vector<float> all_depths;
    i_avg = 0;
    j_avg = 0;
    count = 0;
    for (int v=0;v<contours[idx].size().height;v++) {
        cv::Point point = contours[idx].at<cv::Point>(v);
        i_avg += point.x;
        j_avg += point.y;
        count ++;
        float current_depth = depth_map.at<float>(point.y, point.x);
        all_depths.push_back(current_depth);
    }
    cv::Point mark = cv::Point(i_avg/count, j_avg/count);
    cv::drawMarker(img, mark, cv::Scalar(0, 0, 0), 16, 3, 8);

    std::nth_element(all_depths.begin(),
                     all_depths.begin() + all_depths.size()/2,
                     all_depths.end());
    float median_depth = all_depths[all_depths.size()/2];

    arr image_coord = {(float)i_avg/count, (float)j_avg/count, median_depth};
    cout<<"median depth "<<image_coord<<endl;

    // camera coordinates
    depthData2point(image_coord, Fxypxy); //transforms the point to camera xyz coordinates

    // world coordinates
    cameraFrame->X.applyOnPoint(image_coord); //transforms into world coordinates
    image_coord.append((double)i_avg/count);
    image_coord.append((double)j_avg/count);

    world_coordinates.push_back(image_coord);

    cv::imwrite("rgb_detected.png", img);
    printf("\n");
    return world_coordinates;
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
            if (image_processing::check_neighbor(thresholded_img, 10, i, j)) {
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
    image_processing::remove_outliers(thresholded_img);

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

// analyze to return which ball moves, 0-red, 1-green
int analyze_video(cv::Mat &first_frame, cv::Mat &curr_frame, arr &pixel_location) {
    int res = -1;


    cv::Mat delta;
    cv::Mat delta2;

    cv::Mat first_gray;
    cv::cvtColor(first_frame, first_gray, cv::COLOR_BGR2GRAY);
    cv::blur(first_gray, first_gray, cv::Size(3, 3));
    cv::Mat curr_gray;
    cv::Mat curr_hsv;
    cv::Mat green_img;
    cv::Mat red_img;

    std::vector<cv::Mat> square_contours = image_processing::find_square_contours(first_frame);

    // grab current frame
    cv::blur(curr_frame, curr_frame, cv::Size(3, 3));
    cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_frame, curr_hsv, cv::COLOR_BGR2HSV);

    // detect moving pixels
    cv::absdiff(first_gray, curr_gray, delta);
    cv::threshold(delta, delta2, 25, 255, cv::THRESH_BINARY);
    image_processing::remove_points_outside_contours(delta2, square_contours);

    std::vector<cv::Mat> contours;
    std::vector<cv::Mat> big_contours;

    findContours(delta2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    for (uint u=0; u<contours.size(); u++) {
        double area = cv::contourArea(contours[u]);
        if (area > 100) big_contours.push_back(contours[u]);
    }
    drawContours(curr_frame, big_contours, -1, color);

    // check if moving pixels are balls
    cv::inRange(curr_hsv, cv::Scalar(35, 100, 100), cv::Scalar(55, 255, 255), green_img);
    cv::inRange(curr_hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), red_img);

    // fill up holes inside thresholded region
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(red_img, red_img, element);
    cv::dilate(green_img, green_img, element);

    // remove moving pixels that is not green or red
    for (uint u=0; u<big_contours.size(); u++) {
        double area = cv::contourArea(big_contours[u]);
        if (area > 0) {
            cv::Moments m = cv::moments(big_contours[u]);
            int cX = int(m.m10 / m.m00);
            int cY = int(m.m01 / m.m00);
            if (green_img.at<uchar>(cY, cX) > 0) {
                res = 1;
                pixel_location(0) = cX;
                pixel_location(1) = cY;
            } else if (red_img.at<uchar>(cY, cX) > 0) {
                res = 0;
                pixel_location(0) = cX;
                pixel_location(1) = cY;
            }
            cv::Point mark3 = cv::Point(cX, cY);
            cv::drawMarker(curr_frame, mark3, cv::Scalar(0, 100, 100), 16, 3, 8);
        }

    }

    printf("VIDEO ANALYSIS: detect %d\n", res);
    //cv::imshow("first frame", first_frame);
    //cv::imshow("moving", delta2);
    //cv::imshow("curr frame", curr_frame);
    //cv::imshow("green", green_img);
    //cv::imshow("red", red_img);
    //cv::waitKey(0);

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
            return image_processing::analyze_right_hand_cam(rgb);
        }
    }
}

void calibrate_threshold_values() {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();
    while (1) {

        cv::Mat im = CV(_rgb.get()).clone();
        cv::Mat hsv;
        cv::Mat thresholded;
        if (im.total() > 0) {
            cv::cvtColor(im, hsv, cv::COLOR_BGR2HSV);

            cv::inRange(hsv, cv::Scalar(17, 100, 100), cv::Scalar(37, 255, 255), thresholded);//yellow
            cv::inRange(hsv, cv::Scalar(93, 100, 100), cv::Scalar(113, 255, 255), thresholded);//blue
            cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), thresholded);//red
            //cv::inRange(hsv, cv::Scalar(35, 100, 100), cv::Scalar(55, 255, 255), thresholded);//green

            cv::imshow("orig", im);
            cv::imshow("thresholded", thresholded);
            cv::waitKey(1);
        }
    }
}

int main(int argc,char **argv){
    bool motion = true;
    bool testing_trivial = true;

    // basic setup
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();
    bool first_time_run = true;
    rai::KinematicWorld C = kinematics::setup_kinematic_world();
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

            // asking to change balls
            int changed_ball = 1;
            arr pix = {0, 0};
            printf("Ready\n");
            cv::Mat orig_frame = CV(_rgb.get()).clone();
            while (1) {
                orig_frame = CV(_rgb.get()).clone();
                if (orig_frame.total() > 0) break;
            }
            printf("Change balls\n");
            pause_program();
            cv::Mat next_frame = CV(_rgb.get()).clone();
            while (1) {
                next_frame = CV(_rgb.get()).clone();
                if (next_frame.total() > 0) break;
            }
            changed_ball = analyze_video(orig_frame, next_frame, pix);
            printf("Ball target color = %d\n", changed_ball);
            pause_program();

            // perceiving target point
            print_with_color("doing perception");
            std::vector<arr> res;
            std::vector<arr> targets;
            std::vector<arr> pixel_coords;
            while(1) {
                if (pix(0) > 0 || pix(1) > 0) res = perception(C, changed_ball, pix);
                else res = perception(C);
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
            q_current = kinematics::ik_compute(C, B, targets[0], q_home, motion);
            pause_program_auto();

            // analyzing right hand cam
            print_with_color("calibrating ...");
            float distance_to_grippers = grab_right_hand_cam();
            while (1) {
                if (abs(distance_to_grippers) > 70) {
                    if (distance_to_grippers > 0) targets[0](1) += 0.007;
                    else if (distance_to_grippers < 0) targets[0](1) -= 0.007;
                } else if (abs(distance_to_grippers) < 60) {
                    if (distance_to_grippers > 0) targets[0](1) -= 0.007;
                    else if (distance_to_grippers < 0) targets[0](1) += 0.007;
                }
                q_current = kinematics::ik_compute(C, B, targets[0], q_home, motion);
                pause_program_auto();
                distance_to_grippers = grab_right_hand_cam();
                if (abs(distance_to_grippers) < 70 && abs(distance_to_grippers) >= 60) break;
            }
            print_with_color("calibrating done.");

            targets[0](2) = 0.81;
            q_current = kinematics::ik_compute(C, B, targets[0], q_home, motion);
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
            q_current = kinematics::ik_compute_with_grabbing(C, B, targets[0], q_home, motion);
            pause_program_auto();

            // go to target bin
            cout<<"now go to target bin"<<endl;
            arr bin_target = analyze_scene(C, pixel_coords[0]);
            bin_target(2) += 0.15;
            q_current = kinematics::ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
            pause_program_auto();
            bin_target(2) = 0.83;
            q_current = kinematics::ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
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
                int changed_ball = 1;
                arr pix = {0, 0};
                printf("Ready\n");
                cv::Mat orig_frame = CV(_rgb.get()).clone();
                while (1) {
                    orig_frame = CV(_rgb.get()).clone();
                    if (orig_frame.total() > 0) break;
                }
                printf("Change balls\n");
                pause_program();
                cv::Mat next_frame = CV(_rgb.get()).clone();
                while (1) {
                    next_frame = CV(_rgb.get()).clone();
                    if (next_frame.total() > 0) break;
                }
                changed_ball = analyze_video(orig_frame, next_frame, pix);
                printf("Ball target color = %d\n", changed_ball);
            }

        }
    }
    return 0;
}
