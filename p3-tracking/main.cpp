nclude <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>
#include <RosCom/roscom.h>
#include <RosCom/rosCamera.h>
#include <Kin/frame.h>
#include <Gui/opengl.h>
#include <RosCom/baxter.h>
#include <iostream>


void get_objects_into_configuration(){
  Var<byteA> _rgb;
  Var<floatA> _depth;

  RosCamera cam(_rgb, _depth, "cameraRosNode", "/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw");

  Depth2PointCloud d2p(_depth, 600.f, 600.f, 320.f, 240.f);

  BaxterInterface B(true);

  rai::KinematicWorld C;
  C.addFile("model.g");
  C.setJointState(B.get_q());

  rai::Frame *pcl = C.addFrame("pcl", "camera");
  for(uint i=0;i<10000;i++){
    _rgb.waitForNextRevision();

    if(d2p.points.get()->N>0){
      C.gl().dataLock.lock(RAI_HERE);
      pcl->setPointCloud(d2p.points.get());
      C.gl().dataLock.unlock();
      C.watch(false);
    }

    {
      cv::Mat rgb = CV(_rgb.get());
      cv::Mat depth = CV(_depth.get());

      if(rgb.total()>0 && depth.total()>0){
        cv::imshow("rgb", rgb); //white=2meters
        cv::imshow("depth", 0.5*depth); //white=2meters
        cv::waitKey(1);
      }
    }
  }

  rai::wait();
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

void test_perception(cv::Mat &img, cv::Mat &depth_map) {
    //cv::Mat img = cv::imread("rgb_const.png");
    //cv::Mat depth_map = cv::imread("depth_const.png");
    cv::Mat img2;
    cv::Mat hsv_image;
    cv::Mat hsv_image2;

    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat lower_red_hue_range;

    // blur
    for ( int u = 1; u < 31; u = u + 2 ){ cv::GaussianBlur( img, img2, cv::Size( u, u ), 0, 0 ); }
    cv::cvtColor(img2, hsv_image2, cv::COLOR_BGR2HSV);

    // threshold
    cv::inRange(hsv_image2, cv::Scalar(40, 40, 40), cv::Scalar(70, 255, 255), lower_red_hue_range);

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
        cv::drawMarker(depth_map, mark, cv::Scalar(255, 0, 0), 16, 3, 8);

        printf("  x = %d, y = %d, z = %f\n", i_avg/count, j_avg/count, depth_map.at<float>(i_avg/count, j_avg/count));
    }
    //cv::imwrite("res2.png", lower_red_hue_range);
    //cv::imwrite("res_final.png", img);

    for (int i=0;i<rows;i++) {
      for (int j=0;j<cols;j++) {
          cv::Scalar d = depth_map.at<char>(i, j);
          //std::cout<<d<<std::endl;
          //printf("depth: %f\n", depth_map.at<float>(i, j));
      }
    }

    //cv::imshow("threshold binary", lower_red_hue_range);
    cv::imshow("rgb", img); //white=2meters
    cv::imshow("depth", 0.5*depth_map); //white=2meters
    cv::waitKey(1);
    printf("\n");
}

void minimal_use(){

  Var<byteA> _rgb;
  Var<floatA> _depth;

#if 1 //using ros
  RosCamera cam(_rgb, _depth, "cameraRosNode", "/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw");
#else //using a webcam
  OpencvCamera cam(_rgb);
#endif

  //looping images through opencv
  for(uint i=0;i<100;i++){
    _rgb.waitForNextRevision();
    {
      cv::Mat rgb = CV(_rgb.get());
      cv::Mat depth = CV(_depth.get());

      if(rgb.total()>0 && depth.total()>0){
        test_perception(rgb, depth);
        //cv::waitKey(1);
        //printf("depth pixel 0, 0 = %f\n", depth.at<float>(0, 0));
        //cv::imwrite("depth.png", depth);
        //cv::imwrite("rgb.png", rgb);
      }
    }
  }
}


int main(int argc,char **argv){
  rai::initCmdLine(argc,argv);
  //test_perception();
  minimal_use();
//    get_objects_into_configuration();

  return 0;
}
