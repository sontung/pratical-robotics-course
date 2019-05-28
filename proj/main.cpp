#include <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>

#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>
#include <RosCom/baxter.h>
#include <Kin/frame.h>
#include <Gui/opengl.h>
#include <Operate/robotOperation.h>

#include <iostream>

void pause_program() {
    cout << "Press to continue"<<endl;
    std::cin.get();
}

void ik_compute(rai::KinematicWorld &kine_world, RobotOperation robot_op,
                arr target_position, arr q_home,
                bool verbose=true) {
    rai::Frame *objectFrame = kine_world.addFrame("obj");
    objectFrame->setShape(rai::ST_ssBox, {.1, .1, .1, .02});
    objectFrame->setColor({.8, .8, .1});
    objectFrame->setPosition(target_position);

    // tracking IK
    arr y, y_verbose, J, Phi, PhiJ;
    arr q, q_best;
    float best_error = 100.0f;
    arr Wmetric = diag(2., kine_world.getJointStateDimension());

    //cout<<"staying close to q home: "<<q_home<<endl;
    //pause_program();

    int i = 0;
    while(i < 1000) {
        Phi.clear();
        PhiJ.clear();

        //1st task: go to target pos
        kine_world.evalFeature(y, J, FS_position, {"pointer"});
        //kine_world.evalFeature(y, J, FS_positionDiff, {"pointer", "obj"});
        arr pos_diff = y-target_position;
        pos_diff(2) *= 1e1;
        Phi.append( (pos_diff) * 1e2);
        PhiJ.append( J * 1e2 );

        //2nd task: joint should stay close to zero
        kine_world.evalFeature(y, J, FS_qItself, {});
        Phi .append( (y-q_home) * 1e0 );
        PhiJ.append( J * 1e0 );

        //3rd task: joint angles
        kine_world.evalFeature(y, J, FS_quaternionDiff, {"pointer", "obj"});
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
            cout << "     abs error=" << sumOfAbs(y-target_position)/3.0f<<endl;
            cout << "     phi and phi j sizes="<<Phi.N<<" "<<PhiJ.N<<endl;
        } else kine_world.evalFeature(y, J, FS_position, {"pointer"});

        // save best motion
        float error = sumOfAbs(y-target_position)/3.0f;
        if (error < best_error) {
            best_error = error;
            q_best = q;
        }

        // evaluate to terminate early
        if (error < 0.00001) break;
        i++;
    }

    robot_op.move({q_best}, {2});
    robot_op.sync(kine_world);

}

rai::KinematicWorld setup_kinematic_world() {
    rai::KinematicWorld C;
    C.addFile("../rai-robotModels/baxter/baxter_new.g");

    // add a frame for the camera
    rai::Frame *cameraFrame = C.addFrame("camera", "head");
    cameraFrame->Q.setText("d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1)");
    cameraFrame->calc_X_from_parent();

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
        int v1 = n1.val[0];
        int v2 = n2.val[0];
        int v3 = n3.val[0];
        int v4 = n4.val[0];

        if (v1 == 255 || v2 == 255 ||
                v3 == 255 || v4 == 255) {
            step++;
        } else {
            res = false;
            break;
        }
    }
    return res;
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
    double f = 1./tan(0.5*60.8*RAI_PI/180.);
    f *= 320.;
    arr Fxypxy = {f, f, 320., 240.};

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
    for (uint u=0;u<contours.size();u++) {
        float area = cv::contourArea(contours[u]);
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
        }
        cv::Point mark = cv::Point(i_avg/count, j_avg/count);
        cv::drawMarker(img, mark, cv::Scalar(0, 0, 0), 16, 3, 8);
        arr image_coord = {(float)i_avg/count, (float)j_avg/count, depth_map.at<float>(j_avg/count, i_avg/count)};

        // camera coordinates
        depthData2point(image_coord, Fxypxy); //transforms the point to camera xyz coordinates

        // world coordinates
        cameraFrame->X.applyOnPoint(image_coord); //transforms into world coordinates

        world_coordinates.push_back(image_coord);
    }


    cv::imwrite("rgb_detected.png", img);
    //cv::imshow("rgb", img); //white=2meters
    //cv::imshow("depth", 0.5*depth_map); //white=2meters
    cv::waitKey(10);
    printf("\n");
    return world_coordinates;
}

void print_with_color(std::string text, int color=32) {
    std::ostringstream stringStream;
    stringStream << "\033[0;"<<color<<"m"<<text<<"\033[0m\n";
    cout<<stringStream.str();
}

void calibrate_to_target(rai::KinematicWorld &kine_world,
                         RobotOperation robot_op,
                         arr target_position, arr q_home) {
    // raise the arm a bit above
    arr current_pos, current_J;
    float depth = target_position(2);
    target_position(2) += 0.2;
    cout<<"Initial at: "<<target_position<<endl;
    ik_compute(kine_world, robot_op, target_position, q_home, false);


    // start to calibrate
    while (1) {
        cout<<"Go to: "<<target_position<<endl;
        target_position(2) -= 0.01;
        ik_compute(kine_world, robot_op, target_position, q_home, false);
        kine_world.evalFeature(current_pos, current_J, FS_position, {"pointer"});
        cout<<"Now at: "<<current_pos<<endl;

        // grabbing image from cam left hand

        pause_program();
        //if (target_position(2) > depth) break;
    }
}

void test_cam() {
    Var<byteA> _rgb;
    Var<floatA> _depth;
    RosCamera cam(_rgb, _depth, "sontung", "/cameras/right_hand_camera/image", "");

    while (1) {
        byteA img = _rgb.get();
        cv::Mat rgb = cv::Mat(img.d0, img.d1, CV_8UC4, img.p);
        // bgra2rgba

        if (rgb.total() > 0) {
            cout<<"showing right hand camera"<<endl;
            cv::imshow( "right hand camera", rgb);
            cv::waitKey(0);
        }
    }
}

int main(int argc,char **argv){
    rai::initCmdLine(argc,argv);
    bool sending_motion = false;

    if (sending_motion) {
        cout<<"will send motion. confirm?"<<endl;
        pause_program();

        // basic setup
        rai::KinematicWorld C = setup_kinematic_world();
        RobotOperation B(C);

        // homing
        print_with_color("moving to home position");
        arr q_home = C.getJointState();
        B.moveHard(q_home);
        //B.sync(C);
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

        // calibrate to target point
        //print_with_color("doing calibration");
        //calibrate_to_target(C, B, targets[0], q_home);

        // ik to target
        print_with_color("doing motion");
        targets[0](2) = 0.821475;
        ik_compute(C, B, targets[0], q_home);
        pause_program();
    } else {
        test_cam();
    }

    return 0;
}
