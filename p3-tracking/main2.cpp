nclude <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>
#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>
#include <Kin/frame.h>
#include <Gui/opengl.h>
#include <RosCom/baxter.h>
#include <Operate/robotOperation.h>


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

        //printf("%d %d %d %d\n", v1, v2, v3, v4);


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

std::vector<std::vector<float>> test_perception(cv::Mat &img, cv::Mat &depth_map) {
    //cv::Mat img = cv::imread("rgb_const.png");
    //cv::Mat depth_map = cv::imread("depth_const.png");
    cv::Mat img2;
    cv::Mat hsv_image;
    cv::Mat hsv_image2;
    std::vector<std::vector<float>> res;

    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat lower_red_hue_range;

    // blur
    for ( int u = 1; u < 31; u = u + 2 ){ cv::GaussianBlur( img, img2, cv::Size( u, u ), 0, 0 ); }
    cv::cvtColor(img2, hsv_image2, cv::COLOR_BGR2HSV);

    // threshold
    cv::inRange(hsv_image2, cv::Scalar(40, 100, 100), cv::Scalar(70, 255, 255), lower_red_hue_range);

    // remove outliers
    int rows = lower_red_hue_range.rows;
    int cols = lower_red_hue_range.cols;
    int i_avg = 0;
    int j_avg = 0;
    int count = 0;
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (check_neighbor(lower_red_hue_range, 10, i, j)) {
                lower_red_hue_range.at<uchar>(i, j) = 255;
                i_avg += i;
                j_avg += j;
                count += 1;
                //printf("%d %d\n", i, j);
            }
            else lower_red_hue_range.at<uchar>(i, j) = 0;
        }
    }

    std::vector<cv::Mat> contours;
    findContours(lower_red_hue_range, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(img, contours, -1, color);
    cv::imwrite("contours.png", img);

    cv::imwrite("threshold.png", lower_red_hue_range);

    std::cout<<"RGB map type "<<img.type()<<std::endl;
    std::cout<<"Depth map type "<<depth_map.type()<<std::endl;

    printf("detected %d objects\n", contours.size());

    for (int u=0;u<contours.size();u++) {
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
        std::vector<float> dummy = {(float)i_avg/count, (float)j_avg/count, depth_map.at<float>(j_avg/count, i_avg/count)};
        res.push_back(dummy);
        printf("  x = %d, y = %d, z = %f\n", i_avg/count, j_avg/count, depth_map.at<float>(j_avg/count, i_avg/count));
        //printf("  inside vector: %f %f %f\n", dummy[0], dummy[1], dummy[2]);
    }
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            cv::Scalar d = depth_map.at<char>(i, j);
            //std::cout<<d<<std::endl;
            //printf("depth: %f\n", depth_map.at<float>(i, j));
        }
    }

    //cv::imshow("rgb", img); //white=2meters
    //cv::imshow("depth", 0.5*depth_map); //white=2meters
    cv::waitKey(1);
    printf("\n");
    return res;
}

arr ik_algo(rai::KinematicWorld &kine_world, arr q_home) {
    arr pos_diff;
    arr quat_diff;
    arr y;
    arr Phi, PhiJ, J;
    arr y_verbose; // for printing messages

    arr q_start = kine_world.getJointState();;
    q_home = kine_world.getJointState();


    arr W;
    uint n = kine_world.getJointStateDimension();
    double w = rai::getParameter("w",1e-1);
    printf("q dim: %d %d, joint state dim: %d\n", q_home.d0, q_home.d1, n);

    W.setDiag(w,n);  //W is equal the Id_n matrix times scalar w
    W = diag(2., kine_world.getJointStateDimension());

    //robot.move({q_start}, {5});
    kine_world.setJointState(q_start);
    arr object_pos;
    kine_world.evalFeature(object_pos, J, FS_position, {"object"});
    arr object_quat;
    kine_world.evalFeature(object_quat, J, FS_quaternion, {"object"});

    for (int i=0; i<1000; i++) {
        kine_world.getJointState(q_start);

        //Phi.clear();
        //PhiJ.clear();

        // position IK
        kine_world.evalFeature(pos_diff, J, FS_positionDiff, {"pointer", "object"});
        Phi.append(pos_diff * 1e2);
        PhiJ.append( J * 1e2 );

        // quaternion IK
        //kine_world.evalFeature(quat_diff, J, FS_quaternionDiff, {"baxterR", "object"});
        //Phi.append(quat_diff * 1e0);
        //PhiJ.append( J * 1e0 );

        // null motion IK
        kine_world.evalFeature(y, J, FS_qItself, {});
        Phi.append( (y-q_home) * 1e0 );
        PhiJ.append( J * 1e0 );

        arr q = kine_world.getJointState();
        q -= 0.05*inverse(~PhiJ*PhiJ + W)*~PhiJ*Phi;
        kine_world.setJointState(q);

        cout << "iter " << i+1 << " pos diff = " << pos_diff << endl;
        kine_world.evalFeature(y_verbose, J, FS_position, {"baxterR"});
        cout << "     current position="<<y_verbose<<", target position="<<object_pos<<endl;
        kine_world.evalFeature(y_verbose, J, FS_quaternion, {"baxterR"});
        cout << "     current quaternion="<<y_verbose<<", target quaternion="<<object_quat<<endl;
        cout << "     squared error=" << sumOfAbs(pos_diff)<<endl;

        if (sumOfAbs(pos_diff)<0.001) break;
        rai::wait(0.1);

    }
    //robot.move({q_start}, {5});
    //robot.wait();
    kine_world.setJointState(q_start);

    return q_start;
}

void get_objects_into_configuration(){
    printf("doing perception\n");
    Var<byteA> _rgb;
    Var<floatA> _depth;

    RosCamera cam(_rgb, _depth, "cameraRosNode", "/camera/rgb/image_raw", "/camera/depth_registered/image_raw");

    double f = 1./tan(0.5*60.8*RAI_PI/180.);
    f *= 320.;
    arr Fxypxy = {f, f, 320., 240.}; //intrinsic camera parameters

    Depth2PointCloud d2p(_depth, Fxypxy);

    BaxterInterface B(true);

    rai::KinematicWorld C;
    C.addFile("../../rai-robotModels/baxter/baxter_new.g");
    //C.addFile("model.g");
    //C.addFile("../../rai-robotModels/baxter/baxter.g");

    rai::Frame *pcl = C.addFrame("pcl", "head");
    pcl->Q.setText("d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1)");
    pcl->calc_X_from_parent();
    std::vector<std::vector<float>> interests;
    arr q_real;
    for(uint i=0;i<10;i++){
        _rgb.waitForNextRevision();

        q_real = B.get_q();
//        if(q_real.N==C.getJointStateDimension())
//            C.setJointState(q_real);

        if(d2p.points.get()->N>0){
            C.gl().dataLock.lock(RAI_HERE);
            pcl->setPointCloud(d2p.points.get(), _rgb.get());
            C.gl().dataLock.unlock();
            int key = C.watch(false);
            if(key=='q') break;
        }

        { //display
            cv::Mat rgb = CV(_rgb.get());
            cv::Mat depth = CV(_depth.get());

            if(rgb.total()>0 && depth.total()>0){
                interests = test_perception(rgb, depth);
                cv::waitKey(1);
            }
        }

        for (int ind=0; ind<interests.size(); ind++) {
            //how to convert image to 3D coordinates:
            float x_pixel_coordinate=interests[ind][0];
            float y_pixel_coordinate=interests[ind][1];
            float depth_from_depthcam=interests[ind][2];
            arr pt = { x_pixel_coordinate, y_pixel_coordinate, depth_from_depthcam };
            //std::cout<<"cam coords: "<<pt<<std::endl;
            depthData2point(pt, Fxypxy); //transforms the point to camera xyz coordinates
            pcl->X.applyOnPoint(pt); //transforms into world coordinates
            std::cout<<"world coords: "<<pt<<std::endl;
            //C.addObject("object", rai::ST_capsule, {.01, .01}, {1., 1., 0.}, -1., 0, pt);
        }

    }
    printf("doing motion\n");
    rai::Frame *obj = C.addFrame("object");
    obj->setShape(rai::ST_capsule, {.01, .01});
    obj->setPosition({0.568592, 0.0576726, 1.17724});

    arr final = ik_algo(C, q_real);
    C.setJointState(final);
    rai::wait();

}


int main(int argc,char **argv){
    rai::initCmdLine(argc,argv);

    get_objects_into_configuration();

    return 0;
}


