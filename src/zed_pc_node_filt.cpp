#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "zed_opencv_wrapper/videocapture.hpp"

// OpenCV includes
#include <opencv2/opencv.hpp>

// Sample includes
#include "zed_opencv_wrapper/calibration.hpp"
#include "zed_opencv_wrapper/stopwatch.hpp"
#include "zed_opencv_wrapper/stereo.hpp"
#include "zed_opencv_wrapper/ocv_display.hpp"
// <---- Includes

#include "sensor_msgs/msg/point_cloud2.hpp"

#define USE_OCV_TAPI // Comment to use "normal" cv::Mat instead of CV::UMat
#define USE_HALF_SIZE_DISP // Comment to compute depth matching on full image frames

#include <chrono>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_field.hpp"

#include "pcl_conversions/pcl_conversions.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/ximgproc/disparity_filter.hpp>

using namespace std::chrono;
using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class ZedOpenCaptureNode : public rclcpp::Node
{
  public:
    ZedOpenCaptureNode()
    : Node("zed_open_capture_node"), count_(0)
    {
    std::cout << "start init";
      publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("zed_pointcloud", 10);
      timer_ = this->create_wall_timer(
      333ms, std::bind(&ZedOpenCaptureNode::timer_callback, this));
      
          
    sl_oc::VERBOSITY verbose = sl_oc::VERBOSITY::INFO;

    // ----> Set Video parameters
#ifdef EMBEDDED_ARM
    params.res = sl_oc::video::RESOLUTION::VGA;
#else
    params.res = sl_oc::video::RESOLUTION::HD720;
#endif
    params.fps = sl_oc::video::FPS::FPS_30;
    params.verbose = verbose;
    // <---- Set Video parameters

    cap = new sl_oc::video::VideoCapture(params);
    // ----> Create Video Capture
    if( !cap->initializeVideo(-1) )
    {
        std::cerr << "Cannot open camera video capture" << std::endl;
        std::cerr << "See verbosity level for more details." << std::endl;

    }
    int sn = cap->getSerialNumber();
    std::cout << "Connected to camera sn: " << sn << std::endl;
    // <---- Create Video Capture

    // ----> Retrieve calibration file from Stereolabs server
    std::string calibration_file;
    // ZED Calibration
    unsigned int serial_number = sn;
    // Download camera calibration file
    if( !sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file) )
    {
        std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
    }
    std::cout << "Calibration file found. Loading..." << std::endl;

    // ----> Frame size
    int w,h;
    cap->getFrameSize(w,h);
    // <---- Frame size

    // ----> Initialize calibration
    sl_oc::tools::initCalibration(calibration_file, cv::Size(w/2,h), map_left_x, map_left_y, map_right_x, map_right_y,
                                  cameraMatrix_left, cameraMatrix_right, &baseline);

    fx = cameraMatrix_left.at<double>(0,0);
    fy = cameraMatrix_left.at<double>(1,1);
    cx = cameraMatrix_left.at<double>(0,2);
    cy = cameraMatrix_left.at<double>(1,2);

    std::cout << " Camera Matrix L: \n" << cameraMatrix_left << std::endl << std::endl;
    std::cout << " Camera Matrix R: \n" << cameraMatrix_right << std::endl << std::endl;


    map_left_x_gpu = map_left_x.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    map_left_y_gpu = map_left_y.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    map_right_x_gpu = map_right_x.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    map_right_y_gpu = map_right_y.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    //Note: you can use the tool 'zed_open_capture_depth_tune_stereo' to tune the parameters and save them to YAML
    if(!stereoPar.load())
    {
        stereoPar.save(); // Save default parameters.
    }

    left_matcher->setMinDisparity(stereoPar.minDisparity);
    left_matcher->setNumDisparities(stereoPar.numDisparities);
    left_matcher->setBlockSize(stereoPar.blockSize);
    left_matcher->setP1(stereoPar.P1);
    left_matcher->setP2(stereoPar.P2);
    left_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    left_matcher->setMode(stereoPar.mode);
    left_matcher->setPreFilterCap(stereoPar.preFilterCap);
    left_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    left_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    left_matcher->setSpeckleRange(stereoPar.speckleRange);

    // right_matcher->setMinDisparity(stereoPar.minDisparity);
    // right_matcher->setNumDisparities(stereoPar.numDisparities);
    // right_matcher->setBlockSize(stereoPar.blockSize);
    // right_matcher->setP1(stereoPar.P1);
    // right_matcher->setP2(stereoPar.P2);
    // right_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    // right_matcher->setMode(stereoPar.mode);
    // right_matcher->setPreFilterCap(stereoPar.preFilterCap);
    // right_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    // right_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    // right_matcher->setSpeckleRange(stereoPar.speckleRange);

    stereoPar.print();
    }

  private:
    void timer_callback()
    {
      auto start = high_resolution_clock::now();
      float mult = 1;
    #ifdef USE_OCV_TAPI
    cv::UMat frameYUV;  // Full frame side-by-side in YUV 4:2:2 format
    cv::UMat frameBGR(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full frame side-by-side in BGR format
    cv::UMat left_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left unrectified image
    cv::UMat right_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right unrectified image
    cv::UMat left_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left rectified image
    cv::UMat right_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right rectified image
    cv::UMat left_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left image for the stereo matcher
    cv::UMat right_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right image for the stereo matcher
    cv::UMat left_disp_half(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Half sized disparity map
    cv::UMat right_disp_half(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Half sized disparity map
    cv::UMat left_disp(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full output disparity
    cv::UMat left_disp_float(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Final disparity map in float32
    cv::UMat left_disp_image(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Normalized and color remapped disparity map to be displayed
    cv::UMat left_depth_map(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Depth map in float32
#else
    cv::Mat frameBGR, left_raw, left_rect, right_raw, right_rect, frameYUV, left_for_matcher, right_for_matcher, left_disp_half,left_disp,left_disp_float, left_disp_vis;
#endif

      uint64_t last_ts=0; // Used to check new frame arrival
      // Get a new frame from camera
      const sl_oc::video::Frame frame = cap->getLastFrame();

      // ----> If the frame is valid we can convert, rectify and display it
      if(frame.data!=nullptr && frame.timestamp!=last_ts)
      {
          last_ts = frame.timestamp;

          // ----> Conversion from YUV 4:2:2 to BGR for visualization
#ifdef USE_OCV_TAPI
          cv::Mat frameYUV_cpu = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data );
          frameYUV = frameYUV_cpu.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_HOST_MEMORY);
#else
          frameYUV = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data );
#endif
          cv::cvtColor(frameYUV,frameBGR,cv::COLOR_YUV2BGR_YUYV);
          // <---- Conversion from YUV 4:2:2 to BGR for visualization

          // ----> Extract left and right images from side-by-side
          left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows));
          right_raw = frameBGR(cv::Rect(frameBGR.cols / 2, 0, frameBGR.cols / 2, frameBGR.rows));
          // <---- Extract left and right images from side-by-side

          // ----> Apply rectification
          sl_oc::tools::StopWatch remap_clock;
#ifdef USE_OCV_TAPI
          cv::remap(left_raw, left_rect, map_left_x_gpu, map_left_y_gpu, cv::INTER_AREA );
          cv::remap(right_raw, right_rect, map_right_x_gpu, map_right_y_gpu, cv::INTER_AREA );
#else
          cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_AREA );
          cv::remap(right_raw, right_rect, map_right_x, map_right_y, cv::INTER_AREA );
#endif
          double remap_elapsed = remap_clock.toc();
          std::stringstream remapElabInfo;
          remapElabInfo << "Rectif. processing: " << remap_elapsed << " sec - Freq: " << 1./remap_elapsed;
          // <---- Apply rectification

          // ----> Stereo matching
          sl_oc::tools::StopWatch stereo_clock;
          double resize_fact = 1.0;
#ifdef USE_HALF_SIZE_DISP
          resize_fact = 0.5 * mult;
          // Resize the original images to improve performances
          cv::resize(left_rect,  left_for_matcher,  cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
          cv::resize(right_rect, right_for_matcher, cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
#else
          left_for_matcher = left_rect; // No data copy
          right_for_matcher = right_rect; // No data copy
#endif
          auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
          auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

          // Apply stereo matching
          left_matcher->compute(left_for_matcher, right_for_matcher, left_disp_half);
          right_matcher->compute(right_for_matcher, left_for_matcher, right_disp_half);

          float lambda = 8000;
          float sigma = 1.4;
          wls_filter->setLambda(lambda);
          wls_filter->setSigmaColor(sigma);
          cv::Mat filtered_disp;
          // wls_filter->filter(left_disp_half, left_rect, left_disp_half, right_disp_half);

          auto stop = high_resolution_clock::now();
          auto duration = duration_cast<microseconds>(stop - start);

          RCLCPP_INFO(this->get_logger(), "time taken is %ld", duration.count());

          left_disp_half.convertTo(left_disp_float,CV_32FC1);
          cv::multiply(left_disp_float,1./16.,left_disp_float); // Last 4 bits of SGBM disparity are decimal

#ifdef USE_HALF_SIZE_DISP
          cv::multiply(left_disp_float,2.,left_disp_float); // Last 4 bits of SGBM disparity are decimal
          cv::UMat tmp = left_disp_float; // Required for OpenCV 3.2
          cv::resize(tmp, left_disp_float, cv::Size(), 1./resize_fact, 1./resize_fact, cv::INTER_AREA);
#else
          left_disp = left_disp_float;
#endif


          double elapsed = stereo_clock.toc();
          std::stringstream stereoElabInfo;
          stereoElabInfo << "Stereo processing: " << elapsed << " sec - Freq: " << 1./elapsed;
          // <---- Stereo matching


          // ----> Show disparity image
          cv::add(left_disp_float,-static_cast<double>(stereoPar.minDisparity-1),left_disp_float); // Minimum disparity offset correction
          cv::multiply(left_disp_float,1./stereoPar.numDisparities,left_disp_image,255., CV_8UC1 ); // Normalization and rescaling

          cv::applyColorMap(left_disp_image,left_disp_image,cv::COLORMAP_JET); // COLORMAP_INFERNO is better, but it's only available starting from OpenCV v4.1.0
          
          // sl_oc::tools::showImage("Disparity", left_disp_image, params.res,true, stereoElabInfo.str());

          // ----> Extract Depth map
          // The DISPARITY MAP can be now transformed in DEPTH MAP using the formula
          // depth = (f * B) / disparity
          // where 'f' is the camera focal, 'B' is the camera baseline, 'disparity' is the pixel disparity

          double num = static_cast<double>(fx*baseline);
          cv::divide(num,left_disp_float,left_depth_map);

          float central_depth = left_depth_map.getMat(cv::ACCESS_READ).at<float>(left_depth_map.rows/2, left_depth_map.cols/2 );
          std::cout << "Depth of the central pixel: " << central_depth << " mm" << std::endl;
          // <---- Extract Depth map

          // ----> Create Point Cloud
          sl_oc::tools::StopWatch pc_clock;
          size_t buf_size = static_cast<size_t>(left_depth_map.cols * left_depth_map.rows);
          // std::vector<cv::Vec3d> buffer( buf_size, cv::Vec3f::all( std::numeric_limits<float>::quiet_NaN() ) );
          cv::Mat depth_map_cpu = left_depth_map.getMat(cv::ACCESS_READ);
          float* depth_vec = (float*)(&(depth_map_cpu.data[0]));

// #pragma omp parallel for
          pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
          pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
          auto filtered_cloud_ptr = filtered_cloud.makeShared();

          for(size_t idx=0; idx<buf_size; idx++)
          {
            pcl::PointXYZ point;
            size_t r = idx/left_depth_map.cols;
            size_t c = idx%left_depth_map.cols;
            double depth = static_cast<double>(depth_vec[idx]);
            //std::cout << depth << " ";
            if(!isinf(depth) && depth >=0 && depth > stereoPar.minDepth_mm/mult && depth < stereoPar.maxDepth_mm/mult)
            {
              point.y = -((c-cx)*depth/fx)/1000 * mult;
              point.z = -((r-cy)*depth/fy)/1000 * mult;
              point.x = depth/1000 * mult;
              pcl_cloud.push_back(point);
            }
          }
          auto cloud_ptr = pcl_cloud.makeShared();
          pcl::VoxelGrid<pcl::PointXYZ> sor;
          sor.setInputCloud(cloud_ptr);
          float leaf_size = 0.03;
          sor.setLeafSize (leaf_size, leaf_size, leaf_size);
          sor.filter (*filtered_cloud_ptr);

          sensor_msgs::msg::PointCloud2 ros_cloud;
          pcl::toROSMsg(*filtered_cloud_ptr, ros_cloud);
          ros_cloud.header.frame_id = "zed_cloud";
          ros_cloud.header.stamp = this->get_clock()->now();
          publisher_->publish(ros_cloud);

        //   double pc_elapsed = stereo_clock.toc();
          // std::stringstream pcElabInfo;
//            pcElabInfo << "Point cloud processing: " << pc_elapsed << " sec - Freq: " << 1./pc_elapsed;
          //std::cout << pcElabInfo.str() << std::endl;
        }
    }

    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;
    cv::Mat cameraMatrix_left, cameraMatrix_right;
    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(stereoPar.minDisparity,stereoPar.numDisparities,stereoPar.blockSize);
    cv::Ptr<cv::StereoSGBM> right_matcher = cv::StereoSGBM::create(stereoPar.minDisparity,stereoPar.numDisparities,stereoPar.blockSize);

#ifdef USE_OCV_TAPI
    cv::UMat map_left_x_gpu;
    cv::UMat map_left_y_gpu;
    cv::UMat map_right_x_gpu;
    cv::UMat map_right_y_gpu;
#endif

    double baseline=0;
    double fx;
    double fy;
    double cx;
    double cy;
    sl_oc::video::VideoParams params;
    sl_oc::video::VideoCapture *cap;
    sl_oc::tools::StereoSgbmPar stereoPar;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
  std::cout << "test";
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ZedOpenCaptureNode>());

  rclcpp::shutdown();
  return 0;
}