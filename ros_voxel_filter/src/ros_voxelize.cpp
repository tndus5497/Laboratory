#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <set>

#include <pcl/io/pcd_io.h>
#include <boost/format.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>


using namespace std;

ros::Publisher pub;

void voxelization(const sensor_msgs::PointCloud2ConstPtr& scan)
{
    // Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>); 
    pcl::fromROSMsg(*scan,*cloud1);
    
    (*cloud).width = (*cloud1).size();
    (*cloud).height = 1;
    (*cloud).is_dense = true;
    (*cloud).points.resize((*cloud).width*(*cloud).height);
    
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
    vg.setInputCloud (cloud1);
    vg.setLeafSize (2.0f, 2.0f, 2.0f); //original = 2.0, driving = 1.5 
    vg.filter (*cloud);
 
    pcl::PointIndices::Ptr inliers1(new pcl::PointIndices);
    pcl::ExtractIndices<pcl::PointXYZI> extract1;

    for(int j=0; j<(*cloud).size(); j++)
    {
        //if(cloud1->points[j].intensity!=0)  
        //{   
         inliers1->indices.push_back(j);
        //}   
    }

    extract1.setInputCloud(cloud);
    extract1.setIndices(inliers1);
    extract1.setNegative(false);
    extract1.filter(*cloud);

    pcl::PCLPointCloud2 cloud_p;
    pcl::toPCLPointCloud2(*cloud,  cloud_p);


    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(cloud_p, output);

    output.header.frame_id = "base_link"; // ndt matching -> base_link
    pub.publish(output);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "voxelization");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/points", 100, voxelization);
  pub = nh.advertise<sensor_msgs::PointCloud2> ("after_downsampling", 100);
 
  ros::spin();
}

