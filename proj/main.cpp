#include <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>

#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>
#include <RosCom/baxter.h>
#include <Kin/frame.h>
#include <Gui/opengl.h>
#include <Operate/robotOperation.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <numeric>
#include <algorithm>

void pause_program() {
    cout << "Press to continue"<<endl;
    std::cin.get();
}

arr ik_compute(rai::KinematicWorld &kine_world, RobotOperation robot_op,
               arr &target_position, arr q_home, bool sending_motion=true,
               bool verbose=false) {
    rai::Frame *objectFrame = kine_world.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});
    objectFrame->setPosition(target_position);

    double tolerate=0.0001;
    double time=4.0;

    // tracking IK
    arr y, y_verbose, J, Phi, PhiJ;
    arr q, q_best;
    int best_iter;
    double best_error = 100.0;
    arr Wmetric = diag(2., kine_world.getJointStateDimension());

    cout<<"IK: target postion at "<<target_position<<endl;


    int i = 0;
    while(i < 1000) {
        Phi.clear();
        PhiJ.clear();

        //1st task: go to target pos
        kine_world.evalFeature(y, J, FS_position, {"pointer"});
        arr pos_diff = y-target_position;
        //pos_diff(2) *= 1e1; // emphasize on z coord
        Phi.append( (pos_diff) * 1e2);
        PhiJ.append( J * 1e2 );

        //2nd task: joint should stay close to zero
        kine_world.evalFeature(y, J, FS_qItself, {});
        Phi .append( (y-q_home) * 1e0 );
        PhiJ.append( J * 1e0 );

        //3rd task: joint angles
        kine_world.evalFeature(y, J, FS_vectorZDiff, {"pointer", "obj"});
        Phi.append( y * 1e1);
        PhiJ.append( J * 1e1 );

        // IK compute joint updates
        q = kine_world.getJointState();
        q -= 0.05*inverse(~PhiJ*PhiJ + Wmetric) * ~PhiJ * Phi;

        kine_world.setJointState(q);
        kine_world.watch();

        // verbose
        if (verbose) {
            kine_world.evalFeature(y, J, FS_position, {"pointer"});
            cout << "iter " << i+1 << " pos diff = " << y-target_position << endl;
            kine_world.evalFeature(y_verbose, J, FS_position, {"pointer"});
            cout << "     current position="<<y_verbose<<", target position="<<target_position<<endl;
            kine_world.evalFeature(y_verbose, J, FS_quaternion, {"baxterR"});
            cout << "     current quaternion="<<y_verbose<<", target quaternion="<<objectFrame->getQuaternion()<<endl;
            cout << "     abs error=" << sumOfAbs(y-target_position)/3.0<<endl;
            cout << "     phi and phi j sizes="<<Phi.N<<" "<<PhiJ.N<<endl;
        } else kine_world.evalFeature(y, J, FS_position, {"pointer"});

        // save best motion
        double error = sumOfAbs(y-target_position)/3.0;
        if (error < best_error) {
            best_error = error;
            q_best = q;
            best_iter = i;
        }

        // evaluate to terminate early
        if (error < tolerate) break;
        i++;
    }

    printf("IK: done in %d iters with error=%f at iter %d\n", i, best_error, best_iter);
    if (sending_motion) robot_op.move({q_best}, {time});
    kine_world.setJointState(q_best);

    kine_world.evalFeature(y, J, FS_position, {"pointer"});
    cout<<"IK: final postion at "<<y<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"pointer"});
    cout<<"IK: final z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"pointer"});
    cout<<" final quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"obj"});
    cout<<"IK: target z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"obj"});
    cout<<" target quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZDiff, {"pointer", "obj"});
    cout<<"IK: Z vector diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    kine_world.evalFeature(y_verbose, J, FS_quaternionDiff, {"pointer", "obj"});
    cout<<"IK: quaternion diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    return q_best;
}

arr ik_compute_with_grabbing(rai::KinematicWorld &kine_world, RobotOperation robot_op,
                             arr &target_position, arr q_home, bool sending_motion=true) {
    rai::Frame *objectFrame = kine_world.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});
    objectFrame->setPosition(target_position);

    double tolerate=0.0001;
    double time=4.0;

    // tracking IK
    arr y, y_verbose, J, Phi, PhiJ;
    arr q, q_best;
    int best_iter;
    double best_error = 100.0;
    arr Wmetric = diag(2., kine_world.getJointStateDimension());

    cout<<"IK: target postion at "<<target_position<<endl;


    int i = 0;
    while(i < 1000) {
        Phi.clear();
        PhiJ.clear();

        //1st task: go to target pos
        kine_world.evalFeature(y, J, FS_position, {"pointer"});
        arr pos_diff = y-target_position;
        //pos_diff(2) *= 1e1; // emphasize on z coord
        Phi.append( (pos_diff) * 1e2);
        PhiJ.append( J * 1e2 );

        //2nd task: joint should stay close to zero
        kine_world.evalFeature(y, J, FS_qItself, {});
        Phi .append( (y-q_home) * 1e0 );
        PhiJ.append( J * 1e0 );

        //3rd task: joint angles
        kine_world.evalFeature(y, J, FS_vectorZDiff, {"pointer", "obj"});
        Phi.append( y * 1e1);
        PhiJ.append( J * 1e1 );

        // IK compute joint updates
        q = kine_world.getJointState();
        q -= 0.05*inverse(~PhiJ*PhiJ + Wmetric) * ~PhiJ * Phi;

        kine_world.setJointState(q);
        kine_world.watch();

        kine_world.evalFeature(y, J, FS_position, {"pointer"});

        // save best motion
        double error = sumOfAbs(y-target_position)/3.0;
        if (error < best_error) {
            best_error = error;
            q_best = q;
            best_iter = i;
        }

        // evaluate to terminate early
        if (error < tolerate) break;
        i++;
    }

    printf("IK: done in %d iters with error=%f at iter %d\n", i, best_error, best_iter);
    q_best(-2) = 1;
    if (sending_motion) robot_op.move({q_best}, {time});
    kine_world.setJointState(q_best);

    kine_world.evalFeature(y, J, FS_position, {"pointer"});
    cout<<"IK: final postion at "<<y<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"pointer"});
    cout<<"IK: final z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"pointer"});
    cout<<" final quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZ, {"obj"});
    cout<<"IK: target z vector = "<<y_verbose;
    kine_world.evalFeature(y_verbose, J, FS_quaternion, {"obj"});
    cout<<" target quat = "<<y_verbose<<endl;

    kine_world.evalFeature(y_verbose, J, FS_vectorZDiff, {"pointer", "obj"});
    cout<<"IK: Z vector diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    kine_world.evalFeature(y_verbose, J, FS_quaternionDiff, {"pointer", "obj"});
    cout<<"IK: quaternion diff = "<<y_verbose<<" abs error = "<<sumOfAbs(y_verbose)<<endl;

    return q_best;
}

rai::KinematicWorld setup_kinematic_world() {
    rai::KinematicWorld C;
    C.addFile("../rai-robotModels/baxter/baxter_new.g");

    // add a frame for the camera
    rai::Frame *cameraFrame = C.addFrame("camera", "head");
    cameraFrame->Q.setText("d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1)");
    cameraFrame->calc_X_from_parent();
    cameraFrame->setPosition({-0.0472772, 0.226517, 1.79207});
    cameraFrame->setQuaternion({0.969594, 0.24362, -0.00590741, 0.0223832});

    // add a frame for the object
    rai::Frame *objectFrame = C.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});

    // add a frame for the endeff reference
    rai::Frame *pointerFrame = C.addFrame("pointer", "baxterR");
    pointerFrame->setShape(rai::ST_ssBox, {.05, .05, .05, .01});
    pointerFrame->setColor({.8, .1, .1});
    pointerFrame->setRelativePosition({0.,0.,-.05});

    return C;
}

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
        image_coord(1) -= 0.01;

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

double grab_right_hand_cam() {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/cameras/right_hand_camera/image", "");

    while (1) {
        byteA img = _rgb.get();
        cv::Mat rgb = cv::Mat(img.d0, img.d1, CV_8UC4, img.p);

        if (rgb.total() > 0) {
            //return find_circle(rgb);
            cv::imwrite("right_hand_cam.png", rgb);
            return 0.0;
        }
    }
}

void go_down(rai::KinematicWorld &kine_world,
             RobotOperation robot_op,
             arr target_position, arr q_home) {
    arr current_pos, current_J;
    arr dummy = ik_compute(kine_world, robot_op, target_position, q_home, false);
    kine_world.evalFeature(current_pos, current_J, FS_position, {"pointer"});
    cout<<"Initial at: "<<current_pos<<endl;
    int i = 0;


    // start to calibrate
    while (1) {
        target_position(2) -= 0.01;
        cout<<"Go to "<<target_position<<endl;
        dummy = ik_compute(kine_world, robot_op, target_position, q_home, false);
        kine_world.evalFeature(current_pos, current_J, FS_position, {"pointer"});
        cout<<"Now at "<<current_pos<<endl;

        cout<<"analyzing right hand camera"<<endl;
        pause_program();

        // grabbing image from cam right hand
        double area = grab_right_hand_cam();
        if (area < 0 || area > 13000.0) break;
        pause_program();

        i += 1;
    }
}

void analyze_right_hand_cam() {
    cv::Mat im = cv::imread("right_hand_cam2.png");

    // find circle
    cv::Mat im_gray;
    cv::Mat detected_edges;
    cv::Mat dst;
    int lowThreshold = 20;
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
            return (cv::contourArea(struct1) < cv::contourArea(struct2) );
        }
    };

    std::vector<cv::Mat> contours;
    std::vector<cv::Mat> interested_contours;
    findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), less_than_key());

    for (uint u=0;u<contours.size();u++) {
        double area = cv::contourArea(contours[u]);
        printf("%f\n", area);
    }
    interested_contours.push_back(contours[contours.size()-1]);
    interested_contours.push_back(contours[contours.size()-2]);
    interested_contours.push_back(contours[contours.size()-3]);


    // draw contours
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(im, interested_contours, -1, color);
    cv::Moments m = cv::moments(interested_contours[0]);
    int cX = int(m.m10 / m.m00);
    int cY = int(m.m01 / m.m00);
    cv::Point p = cv::Point(cX, cY);
    cv::drawMarker(im, p, cv::Scalar(0, 200, 100), 16, 3, 8);

    // draw grippers
    cv::Point mark1 = cv::Point(240, 105);
    cv::drawMarker(im, mark1, cv::Scalar(0, 100, 100), 16, 3, 8);
    cv::Point mark2 = cv::Point(482, 105);
    cv::drawMarker(im, mark2, cv::Scalar(0, 100, 100), 16, 3, 8);

    cv::line(im, mark1, mark2, cv::Scalar(100, 100, 100));


    cv::imshow("noname", im);
    cv::imshow("edge", dst);
    cv::waitKey(0);
}

int main(int argc,char **argv){
    bool motion = true;
    bool testing_trivial = true;

    if (!testing_trivial) {

        rai::initCmdLine(argc,argv);

        // basic setup
        rai::KinematicWorld C = setup_kinematic_world();
        RobotOperation B(C);
        cout <<"joint names: " <<B.getJointNames() <<endl;

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
        pause_program();

        // perceiving target point
        print_with_color("doing perception");
        std::vector<arr> targets;
        while(1) {
            targets = perception(C);
            if (targets.size()>0) break;
        }
        for (uint i=0;i<targets.size();i++) cout<<"detected: "<<targets[i]<<endl;
        pause_program();

        // ik to target
        arr q_current;
        print_with_color("doing motion");
        //targets[0] = {-0.0564372, 0.662442, 0.905867};

        targets[0](2) += 0.3;
        q_current = ik_compute(C, B, targets[0], q_home, motion);
        pause_program();

        targets[0](2) = 0.81;
        q_current = ik_compute(C, B, targets[0], q_home, motion);
        cout<<"Ready to grab at "<< q_current<<endl;
        pause_program();

        // looping to take right hand cam
        while (1) {
            printf("taking right hand cam\n");
            grab_right_hand_cam();
            pause_program();
        }

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
            pause_program();
        }

        // go up
        targets[0](2) += 0.3;
        q_current = ik_compute_with_grabbing(C, B, targets[0], q_home, motion);
        pause_program();

        // go to target bin
        cout<<"now go to target bin"<<endl;
        arr bin_target = {-0.316387, 0.880409, 0.914508};
        bin_target(2) += 0.3;
        q_current = ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
        pause_program();
        bin_target(2) -= 0.1;
        q_current = ik_compute_with_grabbing(C, B, bin_target, q_home, motion);
        pause_program();

        // release ball
        q_current(-2) = 0;
        if (motion) B.moveHard(q_current);

        pause_program();
    }
    else {
        analyze_right_hand_cam();
    }
    return 0;
}

