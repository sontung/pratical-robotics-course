#include <Perception/opencv.h> //always include this first!
#include <Perception/opencvCamera.h>
#include <Perception/depth2PointCloud.h>
#include <RosCom/roscom.h>
#include <Kin/frame.h>

void minimal_use(){

  Var<byteA> rgb;
  Var<floatA> depth;

#if 1 //using ros
  RosCom ROS;
  printf("using robots cam\n");
  SubscriberConv<sensor_msgs::Image, byteA, &conv_image2byteA> subRgb(rgb, "/camera/rgb/image_rect_color");
//  SubscriberConv<sensor_msgs::Image, floatA, &conv_imageu162floatA> subDepth(depth, "/camera/depth_registered/image_raw");
#else //using a webcam
  OpencvCamera cam(rgb);
#endif

  //looping images through opencv
  for(uint i=0;i<100;i++){
    cv::Mat img = CV(rgb.get());
    if(img.total()>0){
      cv::imshow("RGB", img);
      cv::imwrite("test.png", img);
      cv::waitKey(1);
    }
    rai::wait(.1);
  }
}

void test_perception() {
    cv::Mat img = cv::imread("good.png");
    cv::Mat hsv_image;
    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    // Threshold the HSV image, keep only the red pixels
    cv::Mat lower_red_hue_range;
    cv::inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
    //cv::imshow("RGB", hsv_image);
    //cv::waitKey(1);
    std::vector<cv::Mat> contours;
    findContours(lower_red_hue_range, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::imwrite("res2.png", lower_red_hue_range);

    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    drawContours(img, contours, -1, color);
    cv::imwrite("res_final.png", img);
}

void test_bg_subtraction() {
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor;
    cv::Ptr<cv::BackgroundSubtractor> blurred_bg_subtractor;
    bg_subtractor = cv::createBackgroundSubtractorMOG2();
    blurred_bg_subtractor = cv::createBackgroundSubtractorMOG2();

    Var<byteA> rgb;
    Var<floatA> depth;
    cv::Mat fgMask;
    cv::Mat blurred_fgMask;

    #if 0 //using ros
        RosCom ROS;
        printf("using robots cam\n");
        SubscriberConv<sensor_msgs::Image, byteA, &conv_image2byteA> subRgb(rgb, "/camera/rgb/image_rect_color");
    #else //using a webcam
        OpencvCamera cam(rgb);
    #endif

    //looping images through opencv
    while(1) {
        cv::Mat img = CV(rgb.get());
        cv::Mat blurred_img;
        if(img.total()>0){
            // fitting bg model
            bg_subtractor->apply(img, fgMask);

            //fitting contours
            std::vector<cv::Mat> contours;
            findContours(fgMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            //drawing contours
            cv::Scalar color( rand()&255, rand()&255, rand()&255 );
            drawContours(img, contours, -1, color);

            cv::imshow("RGB", img);
            cv::imshow("FG Mask", fgMask);

            // different pipeline with blurring
            for ( int u = 1; u < 31; u = u + 2 ){ cv::GaussianBlur( img, blurred_img, cv::Size( u, u ), 0, 0 ); }
            blurred_bg_subtractor->apply(blurred_img, blurred_fgMask);
            std::vector<cv::Mat> contours2;
            findContours(blurred_fgMask, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            drawContours(blurred_img, contours2, -1, color);

            cv::imshow("Blurred RGB", blurred_img);
            cv::imshow("Blurred FG Mask", blurred_fgMask);
            cv::waitKey(1);


        }
        rai::wait(.1);
    }

}

int main(int argc,char **argv){
  rai::initCmdLine(argc,argv);

  //minimal_use();
  //test_perception();
  test_bg_subtraction();
  return 0;
}
