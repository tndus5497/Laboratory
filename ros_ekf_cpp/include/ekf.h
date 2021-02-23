#include <iostream>
#include <ros/ros.h>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <tf/tf.h>
#include <tf2/LinearMath/Quaternion.h>

#include <sensor_msgs/PointCloud2.h>
#include <velodyne_pointcloud/point_types.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <nav_msgs/Odometry.h>

#include <boost/format.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class ros_ekf
{
public:
  ros_ekf();
  ~ros_ekf();

private:
  ros::NodeHandle nh;
  ros::Subscriber sub;
  ros::Publisher pub;
  
  struct State
  {
	float x;
	float y;
	float z;
	float a;
	float w;
	float h;
	float l;
	float dx;
	float dy;
	float dz;
	float da;
  };  



