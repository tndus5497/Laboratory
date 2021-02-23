cmake_minimum_required(VERSION 2.8.3)
project(ros_voxel_filter)   #####

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
  velodyne_pointcloud
  geometry_msgs
  tf
)

catkin_package(CATKIN_DEPENDS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
  velodyne_pointcloud
  geometry_msgs
  tf
)

###########
## Build ##
###########

include_directories(
  Eigen
  ${catkin_INCLUDE_DIRS}
)


SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

link_directories(${PCL_LIBRARY_DIRS})

#Ground Plane Fitter
find_package(PCL 1.7.2 REQUIRED)
add_definitions(${PCL_DEFINITIONS})



#Scan Line Run 
add_executable(ros_voxel_filter src/ros_voxelize.cpp   #####
)

target_include_directories(ros_voxel_filter PRIVATE   ######
	${PCL_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

target_link_libraries(ros_voxel_filter   ######
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)


