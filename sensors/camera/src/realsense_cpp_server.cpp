#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>

#include <iostream>
#include <sstream>
#include <unistd.h>

#include "ros/ros.h"
#include "ros/console.h"
#include "std_msgs/String.h"
#include "std_msgs/Header.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/PointField.h"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "fh_sensors_camera/capture.h"
#include "fh_sensors_camera/depth.h"
#include "fh_sensors_camera/intrinsics.h"
// intrinsics.srv is not used
#include "fh_sensors_camera/pointcloud.h"
#include "fh_sensors_camera/rgb.h"
#include "fh_sensors_camera/uvSrv.h"

typedef std::tuple<uint8_t, uint8_t, uint8_t> RGB_tuple;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;


#define LOG(msg) (std::cout << "[" + baseService + "] " + msg << std::endl);

class RealsenseServer{
  public:
    std::string baseService;
    bool capture;
    std::string camName;

    std::vector<float> cloudColor;
    std::vector<float> uv;
    cv::Mat colorImage;
    cv::Mat depthImage;
    PointCloud::Ptr pclPointCloud;

    std::vector<float> cloudColorStatic;
    std::vector<float> uvStatic;
    cv::Mat colorImageStatic;
    cv::Mat depthImageStatic;
    PointCloud::Ptr pclPointCloudStatic;


    rs2::config cfg;
    rs2::pipeline pipe;

    rs2::points pclPoints;
    rs2::hole_filling_filter hole_filter;
    rs2::decimation_filter dec_filter;
    rs2::threshold_filter thr_filter;

    //std::vector<int> u, v;

    //Declare ROS variables and functions
    ros::NodeHandle n;
    ros::ServiceServer serviceCapture;
    ros::ServiceServer serviceSendDepth;
    ros::ServiceServer serviceSendRGB;
    ros::ServiceServer serviceUVStatic;
    ros::ServiceServer servicePointCloudStatic;
    ros::Publisher pubPointCloudGeometryStatic;
    ros::Publisher pubRGB;

    //Declare services functions
    bool serviceCaptureStatic(fh_sensors_camera::capture::Request& req, fh_sensors_camera::capture::Response& res){
      LOG("Updating statics.");
      if (req.capture.data){
        updateStatics();
        res.success.data = true;
      } else {
        LOG("Client contacted the server but did not asked to capture new statics");
        res.success.data = false;
      }
      return true;
    }

    bool serviceSendDepthImageStatic(fh_sensors_camera::depth::Request& req, fh_sensors_camera::depth::Response& res){
      sensor_msgs::ImagePtr imgDepthMsg = cv_bridge::CvImage(std_msgs::Header(), "mono16", depthImageStatic).toImageMsg();
      res.img = *imgDepthMsg;
      return true;
    }

    bool serviceSendRGBImageStatic(fh_sensors_camera::rgb::Request& req, fh_sensors_camera::rgb::Response& res){

      sensor_msgs::ImagePtr imgColorMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", colorImageStatic).toImageMsg();
      res.img = *imgColorMsg;
      return true;
    }

    bool serviceGetUVStatic(fh_sensors_camera::uvSrv::Request& req, fh_sensors_camera::uvSrv::Response& res){
      std_msgs::MultiArrayDimension uvDim1;
      uvDim1.label = "length";
      uvDim1.size = uvStatic.size();
      uvDim1.stride = uvStatic.size()/2;

      std_msgs::MultiArrayDimension uvDim2;
      uvDim2.label = "pair";
      uvDim2.size = 2;
      uvDim2.stride = 2;

      std_msgs::MultiArrayLayout uvLayout;
      uvLayout.dim.push_back(uvDim1);
      uvLayout.dim.push_back(uvDim2);
      res.uv.layout= uvLayout;
      res.uv.data = uvStatic;

      return true;
    }

    bool serviceGetPointCloudStatic(fh_sensors_camera::pointcloud::Request& req, fh_sensors_camera::pointcloud::Response& res){

      sensor_msgs::PointCloud2 msgPointCloud2;

      pcl::PCLPointCloud2 pclPointCloud2;
      pcl::toPCLPointCloud2(*pclPointCloudStatic, pclPointCloud2);
      pcl_conversions::fromPCL(pclPointCloud2, msgPointCloud2);
      msgPointCloud2.header.frame_id = "ptu_camera_color_optical_frame";
      msgPointCloud2.height = 1;
      msgPointCloud2.width = pclPoints.size();

      res.pc = msgPointCloud2;
      res.color.data = cloudColorStatic;
      return true;
    }



    void initializeRealsense();
    void update();
    void updateStatics();
    RGB_tuple getTexcolor(rs2::video_frame texture, rs2::texture_coordinate texcoords);
    void publishPointcloud();
    void publishRGB();
    std::vector<float> getUVCoordinates(const rs2::texture_coordinate* uvRS,
                        rs2::points points, int height, int width);
    std::vector<float> getCloudColor(const rs2::texture_coordinate* uvRS,
                        rs2::points points, rs2::video_frame colorFrame);
    PointCloud::Ptr points_to_pcl(const rs2::points& points, rs2::depth_frame depth);

    // Declare constructor
    RealsenseServer(){

        ros::NodeHandle nh;

        int camWidth = 640;
        int camHeight = 480;

        nh.getParam(ros::this_node::getName() + "/cam_base_name", camName);
        baseService = "/sensors/";
        baseService += camName;

        //Initialize ROS
        serviceCapture = n.advertiseService(baseService + "/capture", &RealsenseServer::serviceCaptureStatic, this);
        serviceSendDepth = n.advertiseService(baseService + "/depth", &RealsenseServer::serviceSendDepthImageStatic, this);
        serviceSendRGB = n.advertiseService(baseService + "/rgb", &RealsenseServer::serviceSendRGBImageStatic, this);
        serviceUVStatic = n.advertiseService(baseService + "/pointcloud/static/uv", &RealsenseServer::serviceGetUVStatic, this);
        servicePointCloudStatic = n.advertiseService(baseService + "/pointcloud/static", &RealsenseServer::serviceGetPointCloudStatic, this);

        pubPointCloudGeometryStatic = n.advertise<sensor_msgs::PointCloud2>(baseService + "/pointcloudGeometry/static", 1);
        pubRGB = n.advertise<sensor_msgs::Image>(baseService + "/rgb", 1);
        //Configure Realsense streams
        cfg.enable_stream(RS2_STREAM_COLOR, camWidth, camHeight, RS2_FORMAT_BGR8, 6);
        cfg.enable_stream(RS2_STREAM_DEPTH, camWidth, camHeight, RS2_FORMAT_Z16, 6);

        hole_filter = rs2::hole_filling_filter(2);

        thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.1f);
        thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, 1.0f);
    }
};

void RealsenseServer::initializeRealsense(){
    //SOURCE - https://github.com/IntelRealSense/librealsense/issues/5052æ
    LOG("initializing");
    pipe.start(cfg);

    //DROP STARTUP FRAMES
    for(int i = 0; i < 50; i++){
      pipe.wait_for_frames();
    }
    LOG("Ready");

}

std::vector<float> RealsenseServer::getUVCoordinates(const rs2::texture_coordinate* uvRS,
                    rs2::points points, int height, int width){
  /* Orders texture coordinates (UV) for camera service */

  std::vector<float> uvCoordinates;
  uvCoordinates.reserve(points.size()*2);

  for (size_t i = 0; i < points.size(); i++) {
    int u = std::min(std::max(int(uvRS[i].u*width + .5f), 0), width - 1);
    int v = std::min(std::max(int(uvRS[i].v*height + .5f), 0), height - 1);
    uvCoordinates.push_back(v);
    uvCoordinates.push_back(u);
  }

  return uvCoordinates;
}

std::vector<float> RealsenseServer::getCloudColor(const rs2::texture_coordinate* uvRS,
                    rs2::points points, rs2::video_frame colorFrame){
  /* Orders color information from point cloud for camera service */

  std::vector<float> pclColor;
  pclColor.reserve(points.size()*3);

  for (size_t i = 0; i < points.size(); i++) {

    RGB_tuple current_color = getTexcolor(colorFrame, uvRS[i]);
    pclColor.push_back(std::get<0>(current_color));
    pclColor.push_back(std::get<1>(current_color));
    pclColor.push_back(std::get<2>(current_color));

  }
  return pclColor;
}


void RealsenseServer::update(){
    ros::Time time_now = ros::Time::now();

    rs2::frameset frames = pipe.wait_for_frames();

    if (frames){
      // ALIGN THE STREAMS
      rs2::align align(RS2_STREAM_COLOR);

      rs2::frameset alignedFrames = align.process(frames);
      rs2::frame processedDepthFrame = hole_filter.process(dec_filter.process(thr_filter.process(alignedFrames.get_depth_frame())));
      rs2::depth_frame depthFrame(processedDepthFrame);
      rs2::video_frame colorFrame = alignedFrames.get_color_frame();

      colorImage = cv::Mat(cv::Size(colorFrame.get_width(), colorFrame.get_height()),
                      CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
      depthImage = cv::Mat(cv::Size(depthFrame.get_width(), depthFrame.get_height()),
                      CV_16U, (void*)depthFrame.get_data(), cv::Mat::AUTO_STEP);

      //Generate point cloud message, point cloud colours array, and UV array
      rs2::pointcloud pcl;
      pcl.map_to(colorFrame);
      pclPoints = pcl.calculate(depthFrame);

      const rs2::texture_coordinate* uvRealSense = pclPoints.get_texture_coordinates();

      std::vector<float> uvTemp = getUVCoordinates(uvRealSense, pclPoints, colorFrame.get_height(), colorFrame.get_width());
      if(*max_element(std::begin(uvTemp), std::end(uvTemp))){
        uv = uvTemp;
      }
      cloudColor = getCloudColor(uvRealSense, pclPoints, colorFrame);

      pclPointCloud = points_to_pcl(pclPoints, depthFrame);
      // SOURCE - http://pointclouds.org/documentation/tutorials/resampling.html#moving-least-squares
    }
}

void RealsenseServer::updateStatics(){

  depthImageStatic = depthImage;
  colorImageStatic = colorImage;
  uvStatic = uv;
  cloudColorStatic = cloudColor;
  pclPointCloudStatic = pclPointCloud;

}

RGB_tuple RealsenseServer::getTexcolor(rs2::video_frame texture, rs2::texture_coordinate texcoords) {
  //SOURCE - https://github.com/Resays/xyz_rgb_realsense/blob/master/xyz_rgb_realsense.cp
  //SOURCE 2, line 278 - http://docs.ros.org/en/kinetic/api/librealsense2/html/rs__export_8hpp_source.html
	const int w = texture.get_width(), h = texture.get_height();
	int x = std::min(std::max(int(texcoords.u*w + .5f), 0), w - 1);
	int y = std::min(std::max(int(texcoords.v*h + .5f), 0), h - 1);
	int idx = x * texture.get_bytes_per_pixel() + y * texture.get_stride_in_bytes();
	const auto texture_data = reinterpret_cast<const uint8_t*>(texture.get_data());
	return std::tuple<uint8_t, uint8_t, uint8_t>( texture_data[idx], texture_data[idx + 1], texture_data[idx + 2] );

}

void RealsenseServer::publishRGB(){

  sensor_msgs::ImagePtr imgColorMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", colorImage).toImageMsg();
  pubRGB.publish(imgColorMsg);

}

void RealsenseServer::publishPointcloud(){

  sensor_msgs::PointCloud2 msgPointCloud2;

  pcl::PCLPointCloud2 pclPointCloud2;
  pcl::toPCLPointCloud2(*pclPointCloud, pclPointCloud2);
  pcl_conversions::fromPCL(pclPointCloud2, msgPointCloud2);
  msgPointCloud2.header.frame_id = "ptu_camera_color_optical_frame";
  msgPointCloud2.height = 1;
  msgPointCloud2.width = pclPoints.size();

  pubPointCloudGeometryStatic.publish(msgPointCloud2);

}

PointCloud::Ptr RealsenseServer::points_to_pcl(const rs2::points& points, rs2::depth_frame depth){
  //SOURCE - https://github.com/IntelRealSense/librealsense/blob/master/wrappers/pcl/pcl/rs-pcl.cpp
  PointCloud::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //rs2::depth_frame depth(processed_depth_frame);
  cloud->width = depth.get_width();
  cloud->height = depth.get_height();
  cloud->is_dense = false;
  cloud->points.resize(points.size());
  auto ptr = points.get_vertices();
  for (auto& p : cloud->points) {
      p.x = ptr->x;
      p.y = ptr->y;
      p.z = ptr->z;
      ptr++;
  }
  return cloud;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "realsense_service_cpp_node");

  RealsenseServer camera;
  camera.initializeRealsense();

  ros::Rate loop_rate(30);
  while(ros::ok()){


    camera.update();
    //camera.publishPointcloud();
    //camera.publishRGB();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
