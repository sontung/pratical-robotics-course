#include <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>
#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>

#include <iostream>
#include <numeric>
#include <algorithm>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#include "kinematics.h"
#include "image_processing.h"

static cv::Mat RHC;
static cv::Mat FINDING_BALL_TARGET;
static cv::Mat SCENE_ANALYSIS;
static cv::Mat VIDEO_ANALYSIS;
static int ITER = 0;
static int IMG_ID = 0;

std::string format_string(std::string s, int number) {
    char buffer [50];
    int n;
    n=sprintf (buffer, "vis/%s%d.jpg", s.c_str(), number);
    std::string res(buffer);
    return res;
}

void* show_img_thread(void* im) {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    Var<byteA> _rgb2;
    RosCamera cam2(_rgb2, _depth, "sontung", "/cameras/right_hand_camera/image", "");

    while (1) {
        cv::Mat head = CV(_rgb.get()).clone();
        byteA img = _rgb2.get();
        cv::Mat hand = cv::Mat(img.d0, img.d1, CV_8UC4, img.p);
        if (head.total() > 0) cv::imwrite(format_string("video_head/head_live", IMG_ID), head);
        if (hand.total() > 0) {
            cv::resize(hand, hand, cv::Size(640, 480));
            cv::imwrite(format_string("video_hand/hand_live", IMG_ID), hand);
        }

        IMG_ID += 1;
        usleep(50000);
    }
}

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

    arr image_coord = {(float)cx*depth.at<float>(cy, cx),
                       (float)cy*depth.at<float>(cy, cx),
                       depth.at<float>(cy, cx), 1.0f};

    arr Pinv = arr(3,4,
    {0.00180045, 5.51994e-06, -0.569533, -0.0330757,
     -1.82321e-06, -0.00133149, 1.00136, 0.125005,
     5.08217e-05, -0.00117336, -0.439092, 1.55487});

    image_coord = Pinv * image_coord;
    return image_coord;
}

std::vector<arr> perception(rai::KinematicWorld &kine_world, int ball_color, arr pixel_location,
                            cv::Mat &visual_im) {
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
    Fxypxy = {539.637, 540.941, 317.533, 260.024};

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
    cv::Scalar color( 140, 150, 160 );
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

    arr Pinv = arr(3,4,
    {0.00180045, 5.51994e-06, -0.569533, -0.0330757,
     -1.82321e-06, -0.00133149, 1.00136, 0.125005,
     5.08217e-05, -0.00117336, -0.439092, 1.55487});

    arr image_coord = {(float)i_avg/count*median_depth,
                       (float)j_avg/count*median_depth, median_depth, 1.0f};
    cout<<"median depth "<<image_coord<<endl;
    image_coord = Pinv*image_coord;

    arr wc = {image_coord(0), image_coord(1), image_coord(2)};
    wc.append((double)i_avg/count);
    wc.append((double)j_avg/count);



    world_coordinates.push_back(wc);

    cv::imwrite("rgb_detected.png", img);
    visual_im = img.clone();
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
    cv::Scalar color( 140, 150, 160 );
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

arr analyze_scene(rai::KinematicWorld &kine_world, arr dest_pix,
                  arr ball_pixel_coord, cv::Mat &visual_im) {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();
    cv::Mat im;
    cv::Mat depth_map;

    while (1) {
        im = CV(_rgb.get()).clone();
        depth_map = CV(_depth.get()).clone();
        if (im.total() > 0 && depth_map.total() > 0) break;
    }
    cv::Scalar color( 140, 150, 160 );

    cv::Point p = cv::Point(dest_pix(0), dest_pix(1));
    cv::Scalar color_center( 130, 150, 180 );
    cv::drawMarker(im, p, color_center, 16, 3, 8);
    arr wc = compute_world_coord(dest_pix(0), dest_pix(1), depth_map, kine_world);

    // mark ball
    p = cv::Point(ball_pixel_coord(0), ball_pixel_coord(1));
    cv::drawMarker(im, p, color, 16, 3, 8);

    visual_im = im.clone();

    return wc;

}

// analyze to return which ball moves, 0-red, 1-green
int analyze_video(cv::Mat &first_frame, cv::Mat &curr_frame, arr &from, arr &to,
                  cv::Mat &visual_im, int iter_nb) {
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
    delta = delta2.clone();
    image_processing::remove_points_outside_contours(delta2, square_contours);

    std::vector<cv::Mat> contours;
    std::vector<cv::Mat> big_contours;

    findContours(delta2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( 140, 150, 160 );
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
                to(0) = cX;
                to(1) = cY;
            } else if (red_img.at<uchar>(cY, cX) > 0) {
                res = 0;
                to(0) = cX;
                to(1) = cY;
            } else {
                from(0) = cX;
                from(1) = cY;
            }
            cv::Point mark3 = cv::Point(cX, cY);
            cv::drawMarker(curr_frame, mark3, cv::Scalar(0, 100, 100), 16, 3, 8);
        }

    }

    printf("VIDEO ANALYSIS: detect %d\n", res);

    visual_im = first_frame.clone();
    if (res == -1) {
        printf("VIDEO ANALYSIS: documenting failures\n");
        cv::imwrite("firstframe_fail.png", first_frame);
        cv::imwrite("moving_before_denoising_fail.png", delta);
        cv::imwrite("moving_fail.png", delta2);
        cv::imwrite("curr_frame_fail.png", curr_frame);
        cv::imwrite("green_fail.png", green_img);
        cv::imwrite("red_fail.png", red_img);
    }
    cv::imwrite(format_string("curr_frameVA", iter_nb), curr_frame);

    return res;
}

float grab_right_hand_cam(int iter_nb) {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/cameras/right_hand_camera/image", "");

    while (1) {
        byteA img = _rgb.get();
        cv::Mat rgb = cv::Mat(img.d0, img.d1, CV_8UC4, img.p);

        if (rgb.total() > 0) {
            //return find_circle(rgb);
            cv::imwrite("right_hand_cam.png", rgb);
            return image_processing::analyze_right_hand_cam(rgb, RHC, iter_nb);
        }
    }
}

int main(int argc,char **argv){
    bool motion = true;
    bool testing_trivial = false;

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

    int iret1;
    int t0 = 0;
    pthread_t stream_image_threads;
    iret1 = pthread_create( &stream_image_threads, nullptr, show_img_thread, &t0);

    while (1) {
        ITER += 1;
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
            arr pixF = {0, 0};
            arr pixT = {0, 0};
            arr pixBin = {0, 0};

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
            changed_ball = analyze_video(orig_frame, next_frame, pixF, pixT, VIDEO_ANALYSIS, ITER);
            printf("Ball target color = %d\n", changed_ball);
            bool good = image_processing::count_balls_for_each_square(next_frame, changed_ball,
                                                                      pixF, pixT, pixBin, ITER);
            if (!good) continue;
            pause_program();

            // perceiving target point
            print_with_color("doing perception");
            std::vector<arr> res;
            std::vector<arr> targets;
            std::vector<arr> pixel_coords;
            while(1) {
                if (pixF(0) > 0 || pixF(1) > 0) res = perception(C, changed_ball, pixF, FINDING_BALL_TARGET);
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
            float distance_to_grippers = grab_right_hand_cam(ITER);
            int lower_end = 65;
            int upper_end = 75;
            double step_size = 0.005;
            while (1) {
                distance_to_grippers = grab_right_hand_cam(ITER);
                if (abs(distance_to_grippers) < upper_end && abs(distance_to_grippers) >= lower_end) break;

                if (abs(distance_to_grippers) > lower_end) {
                    if (distance_to_grippers > 0) targets[0](1) += step_size;
                    else if (distance_to_grippers < 0) targets[0](1) -= step_size;
                } else if (abs(distance_to_grippers) < upper_end) {
                    if (distance_to_grippers > 0) targets[0](1) -= step_size;
                    else if (distance_to_grippers < 0) targets[0](1) += step_size;
                }
                q_current = kinematics::ik_compute(C, B, targets[0], q_home, motion);
                pause_program_auto();
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

            arr bin_target = analyze_scene(C, pixBin, pixel_coords[0], SCENE_ANALYSIS);
            bin_target(2) += 0.15;
            q_current = kinematics::ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
            pause_program_auto();
            bin_target(2) = 0.79;
            q_current = kinematics::ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
            pause_program_auto();

            // release ball
            q_current(-2) = 0;
            if (motion) B.moveHard(q_current);
            pause_program_auto();

            // go up again
            bin_target(2) += 0.15;
            q_current = kinematics::ik_compute_cheap(C, B, bin_target, q_home, motion);
            pause_program_auto();

            // homing again
            print_with_color("moving to home position");
            if (motion) B.moveHard(q_home);
            pause_program_auto();

        }
        else {
            int changed_ball = 1;
            arr pixF = {0, 0};
            arr pixT = {0, 0};
            arr pixBin = {0, 0};

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
            changed_ball = analyze_video(orig_frame, next_frame, pixF, pixT, VIDEO_ANALYSIS, ITER);
            printf("Ball target color = %d at %f %f\n", changed_ball, pixT(0), pixT(1));

            bool good = image_processing::count_balls_for_each_square(next_frame,
                                                                      changed_ball, pixF,
                                                                      pixT, pixBin, ITER);


        }
    }



    return 0;
}
