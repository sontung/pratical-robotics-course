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

        // hardcoded
        //image_coord(0) += 0.1;
        //image_coord(1) -= 0.01;

        world_coordinates.push_back(image_coord);
    }


    cv::imwrite("rgb_detected.png", img);
    //cv::imshow("rgb", img); //white=2meters
    //cv::imshow("depth", 0.5*depth_map); //white=2meters
    printf("\n");
    return world_coordinates;
}

void print_with_color(std::string text, int color=32) {
    std::ostringstream stringStream;
    stringStream << "\033[0;"<<color<<"m"<<text<<"\033[0m\n";
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
        double area = cv::contourArea(only_big_contours[u]);
        double peri = cv::arcLength(only_big_contours[u], true);
        cv::Mat vertices;
        cv::approxPolyDP(only_big_contours[u], vertices, 0.04*peri, true);
        printf("area %f has old size=(%d, %d), new size=(%d, %d)\n", area,
               only_big_contours[u].size().width, only_big_contours[u].size().height,
               vertices.size().width, vertices.size().height);
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
    //cv::drawMarker(im, mark1, cv::Scalar(0, 100, 100), 16, 3, 8);
    cv::Point mark2 = cv::Point(482, 106);
    cv::Point mark3 = cv::Point(320, 200);
    cv::drawMarker(im, mark3, cv::Scalar(0, 100, 100), 16, 3, 8);
    //cv::line(im, mark1, mark2, cv::Scalar(100, 100, 100));

    // compute how much devitation from gripper
    printf("ball center is at %d %d\n", cX, cY);
    float slope = (float)(mark1.y-mark2.y)/(mark1.x-mark2.x);
    float distance_to_grippers = (slope*cX-cY+mark2.y-slope*mark2.x)/sqrt(slope*slope+1);
    printf("distance to gippers is %f\n", distance_to_grippers);

    cv::imwrite("rhc_analyzed.png", im);
    cv::imwrite("rhc_edge.png", dst);

    if (show_im) {
        cv::imshow("noname", im);
        cv::imshow("edge", dst);
        cv::waitKey(0);
    }
    return distance_to_grippers;

}

arr analyze_scene(rai::KinematicWorld &kine_world, bool show_img=false) {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/camera/rgb/image_raw", "/camera/depth/image_rect");
    _depth.waitForNextRevision();
    cv::Mat im = CV(_rgb.get()).clone();
    cv::Mat depth_map = CV(_depth.get()).clone();

    //cv::Mat im = cv::imread("rgb_head2.png");

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
        printf("area %f has old size=(%d, %d), new size=(%d, %d)\n", area,
               big_contours[u].size().width, big_contours[u].size().height,
               vertices.size().width, vertices.size().height);
        big_contours[u] = vertices;
    }

    // determine empty square center
    double biggest = -1.0;
    uint idx = 0;
    for (uint u=0;u<big_contours.size();u++) {
        double area = cv::contourArea(big_contours[u]);
        if (area > biggest) {
            biggest = area;
            idx = u;
        }
    }

    // determine square center world coord
    cv::Moments m = cv::moments(big_contours[idx]);
    int cX = int(m.m10 / m.m00);
    int cY = int(m.m01 / m.m00);
    cv::Point p = cv::Point(cX, cY);
    cv::Scalar color_center( rand()&255, rand()&255, rand()&255 );
    cv::drawMarker(im, p, color_center, 16, 3, 8);
    arr wc = compute_world_coord(cX, cY, depth_map, kine_world);

    drawContours(im, big_contours, -1, color);

    if (show_img) {
        cv::imshow("image", im);
        cv::imshow("blur", im_blurred);
        cv::imshow("threshold", thresholded_img);
        cv::waitKey(0);
    }

    return wc;

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
    bool testing_trivial = false;

    // basic setup
    rai::KinematicWorld C = setup_kinematic_world();
    RobotOperation B(C);
    cout <<"joint names: " <<B.getJointNames() <<endl;

    if (!testing_trivial) {

        rai::initCmdLine(argc,argv);

        arr y, J;
        C.evalFeature(y, J, FS_vectorZDiff, {"pointer", "obj"});
        cout <<"vector Z diff: " << y <<endl;
        C.evalFeature(y, J, FS_quaternionDiff, {"pointer", "obj"});
        cout <<"quaternion diff: " << y <<endl;

        cout<<"will send motion. confirm?"<<endl;
        pause_program();

        // homing
        print_with_color("moving to home position");
        arr q_home = C.getJointState();
        if (motion) B.moveHard(q_home);
        cout<<"q home: "<<q_home<<endl;
        pause_program_auto();

        // perceiving target point
        print_with_color("doing perception");
        std::vector<arr> targets;
        while(1) {
            targets = perception(C);
            if (targets.size()>0) break;
        }
        for (uint i=0;i<targets.size();i++) cout<<"detected: "<<targets[i]<<endl;
        pause_program_auto();

        // ik to target
        arr q_current;
        print_with_color("doing motion");

        targets[0](2) += 0.15;
        q_current = ik_compute(C, B, targets[0], q_home, motion);
        pause_program_auto();

        // analyzing right hand cam
        float distance_to_grippers = grab_right_hand_cam();
        while (abs(distance_to_grippers) > 80) {
            if (distance_to_grippers > 0) targets[0](1) += 0.01;
            else if (distance_to_grippers < 0) targets[0](1) -= 0.01;
            q_current = ik_compute(C, B, targets[0], q_home, motion);
            pause_program_auto();
            distance_to_grippers = grab_right_hand_cam();
        }

        targets[0](2) = 0.81;
        q_current = ik_compute(C, B, targets[0], q_home, motion);
        cout<<"Ready to grab at "<< q_current<<endl;
        pause_program_auto(9);

        // closing finger
        while (1) {
            q_current = B.getJointPositions();
            cout <<" q:" <<q_current
                <<" gripper right: " <<B.getGripperOpened("right") <<' '<<B.getGripperGrabbed("right")
               <<endl;
            if (!B.getGripperGrabbed("right")) {
                q_current(-2) = 1;
                if (motion) B.moveHard(q_current);
                break;
            }
            pause_program_auto();
        }

        // go up
        targets[0](2) += 0.3;
        q_current = ik_compute_with_grabbing(C, B, targets[0], q_home, motion);
        pause_program_auto();

        // go to target bin
        cout<<"now go to target bin"<<endl;
        arr bin_target = analyze_scene(C);
        bin_target(2) += 0.3;
        q_current = ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
        pause_program_auto();
        bin_target(2) = 0.9;
        q_current = ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
        pause_program_auto();

        // release ball
        q_current(-2) = 0;
        if (motion) B.moveHard(q_current);
        pause_program_auto();

        // homing again
        print_with_color("moving to home position");
        if (motion) B.moveHard(q_home);
        pause_program();

    }
    else {
        while (1) {
            arr res = analyze_scene(C, true);
            break;
            //pause_program();
        }
    }
    return 0;
}

