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

struct Est_ctrv
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


static State state;
static Est_ctrv est_ctrv;

static Eigen::MatrixXd I = Eigen::MatrixXd::Identity(11,11);
static Eigen::MatrixXd H = Eigen::MatrixXd(7, 11);
static Eigen::MatrixXd P = Eigen::MatrixXd(11, 11);
static Eigen::MatrixXd Q = Eigen::MatrixXd(11, 11);
static Eigen::MatrixXd R = Eigen::MatrixXd(7, 7);
static Eigen::MatrixXd JA = Eigen::MatrixXd(11, 11);

static Eigen::MatrixXd measure = Eigen::MatrixXd(7, 1);
static Eigen::MatrixXd odom = Eigen::MatrixXd(3, 1);
static Eigen::MatrixXd estimated_state = Eigen::MatrixXd(7, 1);

static jsk_recognition_msgs::BoundingBoxArray Input_bbox;
static jsk_recognition_msgs::BoundingBoxArray estimated_bbox;
static jsk_recognition_msgs::BoundingBoxArray updated_bbox;

static jsk_recognition_msgs::BoundingBoxArray arr_estimated_bbox_push;
static jsk_recognition_msgs::BoundingBoxArray arr_updated_bbox_push;

static nav_msgs::Odometry Input_odom;

float dt = 0.09; // CenterPoint Output : 11Hz

bool Init_mat = false;  // Check Initialization of EKF matrix (false = init)
bool Init_stat = false; // Check Initialization of EKF state (false = init)

ros::Publisher pub_estimated_bbox;
ros::Publisher pub_updated_bbox;


void CTRV_model(struct State p)
{
	float px, py, pz, pa;

	float x_ = p.x;
	float y_ = p.y;
	float z_ = p.z;
	float a_ = p.a;
	float dx_ = p.dx;
	float dy_ = p.dy;
	float dz_ = p.dz;
	float da_ = p.da;
	
	if (fabs(da_) > 0.01)
	{
		px = x_ + sqrt(dx_*dx_ + dy_*dy_) / da_ * (sin(a_ + da_ * dt) - sin(a_));
		py = y_ + sqrt(dx_*dx_ + dy_*dy_) / da_ * (cos(a_) - cos(a_ + da_ * dt));
		pz = z_ + dz_ * dt;
		pa = a_ + da_ * dt;
	}
	else
	{
		px = x_ + dx_ * dt;
		py = y_ + dy_ * dt;
		pz = z_ + dz_ * dt;
		pa = a_ + da_ * dt;
	}

	pa = atan2(sin(pa), cos(pa));
	
	state.x = px;
	state.y = py;
	state.z = pz;
	state.a = pa;
	state.dx = dx_;
	state.dy = dy_;
	state.dz = dz_;
	state.da = da_;
}

/////////////////// To avoid affecting the state of EKF ///////////////////
void CTRV_model_for_est(struct Est_ctrv c)
{
	float px, py, pz, pa;

	float x_ = c.x;
	float y_ = c.y;
	float z_ = c.z;
	float a_ = c.a;
	float dx_ = c.dx;
	float dy_ = c.dy;
	float dz_ = c.dz;
	float da_ = c.da;


	if (fabs(da_) > 0.01)
	{
		px = x_ + sqrt(dx_*dx_ + dy_*dy_) / da_ * (sin(a_ + da_ * dt) - sin(a_));
		py = y_ + sqrt(dx_*dx_ + dy_*dy_) / da_ * (cos(a_) - cos(a_ + da_ * dt));
		pz = z_ + dz_ * dt;
		pa = a_ + da_ * dt;
	}
	else
	{
		px = x_ + dx_ * dt;
		py = y_ + dy_ * dt;
		pz = z_ + dz_ * dt;
		pa = a_ + da_ * dt;
	}

	pa = atan2(sin(pa), cos(pa));
		
	est_ctrv.x = px;
	est_ctrv.y = py;
	est_ctrv.z = pz;
	est_ctrv.a = pa;
	est_ctrv.dx = dx_;
	est_ctrv.dy = dy_;
	est_ctrv.dz = dz_;
	est_ctrv.da = da_;
}

Eigen::MatrixXd Jacobian_A(struct State p)
{
	float a_j, b_j, c_j, d_j, e_j, f_j, g_j, h_j;
	
	float x = p.x;
	float y = p.y;
	float z = p.z;
	float a = p.a;
	float dx = p.dx;
	float dy = p.dy;
	float dz = p.dz;
	float da = p.da;
	
	if (fabs(da) > 0.01)
	{
		a_j = sqrt(dx*dx + dy * dy)*(-cos(a) + cos(dt * da + a)) / da;
		b_j = dx * (-sin(a) + sin(dt*da + a)) / (da * sqrt(dx*dx + dy*dy));
		c_j = dy * (-sin(a) + sin(dt*da + a)) / (da * sqrt(dx*dx + dy*dy));
		d_j = dt * sqrt(dx*dx + dy * dy)*cos(dt*da + a) / da - sqrt(dx*dx + dy * dy)*(-sin(a) + sin(dt*da + a)) / da * da;
		e_j = sqrt(dx*dx + dy * dy)*(-sin(a) + sin(dt*da + a)) / da;
		f_j = dx * (cos(a) - cos(dt*da + a)) / (da * sqrt(dx*dx + dy * dy));
		g_j = dy * (cos(a) - cos(dt*da + a)) / (da * sqrt(dx*dx + dy * dy));
		h_j = dt * sqrt(dx*dx + dy * dy)*sin(dt*da + a) / da - sqrt(dx*dx + dy * dy)*(cos(a) - cos(dt*da + a)) / da * da;

		JA.row(0) << 1, 0, 0, a_j, 0, 0, 0, b_j, c_j, 0, d_j;
		JA.row(1) << 0, 1, 0, e_j, 0, 0, 0, f_j, g_j, 0, h_j;
		JA.row(2) << 0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0;
		JA.row(3) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt;
		JA.row(4) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
		JA.row(5) << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
		JA.row(6) << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
		JA.row(7) << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
		JA.row(8) << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
		JA.row(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
		JA.row(10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
	}
	else
	{
		JA.row(0) << 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0;
		JA.row(1) << 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0;
		JA.row(2) << 0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0;
		JA.row(3) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt;
		JA.row(4) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
		JA.row(5) << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
		JA.row(6) << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
		JA.row(7) << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
		JA.row(8) << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
		JA.row(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
		JA.row(10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
	}

	return JA;

}

void predict(struct State p)
{
	Eigen::MatrixXd P_prev = Eigen::MatrixXd(11, 11);
	
	CTRV_model(p);
	JA = Jacobian_A(p);

	P_prev = JA * P;
	
	P = P_prev * JA.transpose() + Q;

}

void update(struct State p, Eigen::MatrixXd z)
{ 	
	Eigen::MatrixXd x = Eigen::MatrixXd(11, 1);
	x(0, 0) = p.x;
	x(1, 0) = p.y;
	x(2, 0) = p.z;
	x(3, 0) = p.a;
	x(4, 0) = p.w;
	x(5, 0) = p.h;
	x(6, 0) = p.l;
	x(7, 0) = p.dx;
	x(8, 0) = p.dy;
	x(9, 0) = p.dz;
	x(10, 0) = p.da;
	
	Eigen::MatrixXd Hx = Eigen::MatrixXd(7,1);
	Hx(0, 0) = p.x;
	Hx(1, 0) = p.y;
	Hx(2, 0) = p.z;
	Hx(3, 0) = p.a;
	Hx(4, 0) = p.w;
	Hx(5, 0) = p.h;
	Hx(6, 0) = p.l;
	

	Eigen::MatrixXd PHT = Eigen::MatrixXd(11,7);
	Eigen::MatrixXd S = Eigen::MatrixXd(7, 7);
	Eigen::MatrixXd inv_S = Eigen::MatrixXd(7, 7);
	Eigen::MatrixXd K = Eigen::MatrixXd(11, 7);
	Eigen::MatrixXd Y = Eigen::MatrixXd(7, 1);
	Eigen::MatrixXd I_KH = Eigen::MatrixXd(11, 11);
	Eigen::MatrixXd P_prev = Eigen::MatrixXd(11, 11);
	Eigen::MatrixXd K_prev = Eigen::MatrixXd(11, 7);

	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(11,11);

	PHT = P * H.transpose();
	//std::cout << "PHT" << PHT << std::endl;

	//std::cout << "H" << H << std::endl;
	//std::cout << "R" << R << std::endl;

	S = H * PHT + R;
	//std::cout << "S" << S << std::endl;

	inv_S = S.inverse();
	//std::cout << "inv_S" << inv_S << std::endl;

	K = PHT * inv_S;
	//std::cout << "K " << K << std::endl;

	Y = z - Hx;
	//std::cout << "Y " << Y << std::endl;

	x = x + K * Y;
	//std::cout << "x " << x << std::endl;

	I_KH = I - K * H; 
	//std::cout << "I_KH " << I_KH << std::endl;

	P_prev = I_KH * P;
	//std::cout << "P_prev " << P_prev << std::endl;

	K_prev = K * R;
	//std::cout << "K_prev " << K_prev << std::endl;

	P = P_prev * I_KH.transpose() + K_prev * K.transpose();
	//std::cout << "P : before " << P << std::endl;

	state.x = x(0, 0);
	state.y = x(1, 0);
	state.z = x(2, 0);
	state.a = x(3, 0);
	state.w = x(4, 0);
	state.h = x(5, 0);
	state.l = x(6, 0);
	state.dx = x(7, 0);
	state.dy = x(8, 0);
	state.dz = x(9, 0);
	state.da = x(10, 0);
	
	state.a = atan2(sin(state.a), cos(state.a)); // Angle normalization
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& scan)
{
	Input_odom.header.frame_id = "/lidar_top";
	Input_odom.pose = scan->pose;
	
	odom(0,0) = Input_odom.pose.pose.position.x;
	odom(1,0) = Input_odom.pose.pose.position.y;
	odom(2,0) = Input_odom.pose.pose.position.z;
}


void ekf_process(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& scan)
{
	if (Init_mat == false)
	{
		/////////////////// P,Q,R Matrix of AB3DMOT ///////////////////
		H.row(0) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		H.row(1) << 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		H.row(2) << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
		H.row(3) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
		H.row(4) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
		H.row(5) << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
		H.row(6) << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;

		P.row(0) << 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		P.row(1) << 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		P.row(2) << 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0;
		P.row(3) << 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0;
		P.row(4) << 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0;
		P.row(5) << 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0;
		P.row(6) << 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0;
		P.row(7) << 0, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0;
		P.row(8) << 0, 0, 0, 0, 0, 0, 0, 0, 10000, 0, 0;
		P.row(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 10000, 0;
		P.row(10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10000;

		Q.row(0) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		Q.row(1) << 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		Q.row(2) << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
		Q.row(3) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
		Q.row(4) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
		Q.row(5) << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
		Q.row(6) << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
		Q.row(7) << 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0;
		Q.row(8) << 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0;
		Q.row(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0;
		Q.row(10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01;

		R.row(0) << 1, 0, 0, 0, 0, 0, 0;
		R.row(1) << 0, 1, 0, 0, 0, 0, 0;
		R.row(2) << 0, 0, 1, 0, 0, 0, 0;
		R.row(3) << 0, 0, 0, 1, 0, 0, 0;
		R.row(4) << 0, 0, 0, 0, 1, 0, 0;
		R.row(5) << 0, 0, 0, 0, 0, 1, 0;
		R.row(6) << 0, 0, 0, 0, 0, 0, 1;
	
		Init_mat = true;
	}

	arr_estimated_bbox_push.boxes.clear(); // remove staked data
	arr_updated_bbox_push.boxes.clear(); // remove staked data
	
	
	/////////////////// Initialize bbox array  ///////////////////
	Input_bbox.header.frame_id = "/lidar_top";	
	Input_bbox.boxes = scan->boxes;
	
	estimated_bbox.header = Input_bbox.header;
	estimated_bbox.boxes = scan->boxes;
	
	updated_bbox.header = Input_bbox.header;
	updated_bbox.boxes = scan->boxes;
			
	arr_estimated_bbox_push.header = Input_bbox.header;
	arr_updated_bbox_push.header = Input_bbox.header;	
	
	double roll, pitch, yaw;
	
	for(int idx=0 ; idx <Input_bbox.boxes.size(); idx++)
	{	
		if(idx < 1)
		{
		
		Input_bbox.boxes[idx].pose.position.x = odom(0,0) + Input_bbox.boxes[idx].pose.position.x;
		Input_bbox.boxes[idx].pose.position.y = odom(1,0) + Input_bbox.boxes[idx].pose.position.y;
		Input_bbox.boxes[idx].pose.position.z = odom(2,0) + Input_bbox.boxes[idx].pose.position.z;
		
		tf::Quaternion q(
        Input_bbox.boxes[idx].pose.orientation.x,
        Input_bbox.boxes[idx].pose.orientation.y,
        Input_bbox.boxes[idx].pose.orientation.z,
        Input_bbox.boxes[idx].pose.orientation.w);
    	tf::Matrix3x3 m(q);
    	   	
    	m.getRPY(roll, pitch, yaw); // quaternion to euler RPY
		
		if (Init_stat == false) // Initial State of EKF
		{			
			state.x = Input_bbox.boxes[idx].pose.position.x;
			state.y = Input_bbox.boxes[idx].pose.position.y;
			state.z = Input_bbox.boxes[idx].pose.position.z;
			state.a = yaw;
			state.w = Input_bbox.boxes[idx].dimensions.x;
			state.h = Input_bbox.boxes[idx].dimensions.z;
			state.l = Input_bbox.boxes[idx].dimensions.y;
			state.dx = 0;
			state.dy = 0;
			state.dz = 0;
			state.da = 0;
			
			Init_stat = true;
			
		}
			
		/////////////////// Measurement Update ///////////////////		
		measure(0,0) = Input_bbox.boxes[idx].pose.position.x;
		measure(1,0) = Input_bbox.boxes[idx].pose.position.y;
		measure(2,0) = Input_bbox.boxes[idx].pose.position.z;
		measure(3,0) = yaw;
		measure(4,0) = Input_bbox.boxes[idx].dimensions.x;
		measure(5,0) = Input_bbox.boxes[idx].dimensions.y;
		measure(6,0) = Input_bbox.boxes[idx].dimensions.z;
								
		/////////////////// To avoid affecting state of EKF  ///////////////////
		est_ctrv.x = state.x; 
		est_ctrv.y = state.y; 
		est_ctrv.z = state.z; 
		est_ctrv.a = state.a; 
		est_ctrv.w = state.w; 
		est_ctrv.h = state.h; 
		est_ctrv.l = state.l; 
		est_ctrv.dx = state.dx; 
		est_ctrv.dy = state.dy; 
		est_ctrv.dz = state.dz; 
		est_ctrv.da = state.da; 
		
		tf::Quaternion q_predict;		
		
		/////////////////// Estimate n frame after position  ///////////////////
		for (int i = 0; i < 11; i++) // Predict 11 frame after position(1sec)
		{
			CTRV_model_for_est(est_ctrv);
					
			q_predict.setRPY( 0, 0, est_ctrv.a); //Euler RPY to quaternion	
			
			estimated_state(0,0) = est_ctrv.x; 
			estimated_state(1,0) = est_ctrv.y;
			estimated_state(2,0) = est_ctrv.z;
			estimated_state(3,0) = q_predict[0];
			estimated_state(4,0) = q_predict[1];
			estimated_state(5,0) = q_predict[2];
			estimated_state(6,0) = q_predict[3];
			
		}		
		
		/////////////////// EKF process ///////////////////
		predict(state);		
		update(state,measure);
													
		tf::Quaternion q_update;
		q_update.setRPY( 0, 0, state.a );
		
		/////////////////// Publish Estimated Position ///////////////////		
		estimated_bbox.boxes[idx].pose.position.x = estimated_state(0,0) - odom(0,0); // Global coordinate to local coordinate (To visualization)
		estimated_bbox.boxes[idx].pose.position.y = estimated_state(1,0) - odom(1,0);
		estimated_bbox.boxes[idx].pose.position.z = estimated_state(2,0) - odom(2,0);
		estimated_bbox.boxes[idx].pose.orientation.x = estimated_state(3,0);
		estimated_bbox.boxes[idx].pose.orientation.y = estimated_state(4,0);
		estimated_bbox.boxes[idx].pose.orientation.z = estimated_state(5,0);
		estimated_bbox.boxes[idx].pose.orientation.w = estimated_state(6,0);
		estimated_bbox.boxes[idx].dimensions.x = state.w;
		estimated_bbox.boxes[idx].dimensions.y = state.h;
		estimated_bbox.boxes[idx].dimensions.z = state.l;
		estimated_bbox.boxes[idx].value = Input_bbox.boxes[idx].value;
		estimated_bbox.boxes[idx].label = Input_bbox.boxes[idx].label+1; // To make different color with GT
		
		/////////////////// Publish Updated Position ///////////////////	
		updated_bbox.boxes[idx].pose.position.x = state.x - odom(0,0); // Global coordinate to local coordinate (To visualization)
		updated_bbox.boxes[idx].pose.position.y = state.y - odom(1,0);
		updated_bbox.boxes[idx].pose.position.z = state.z - odom(2,0);
		updated_bbox.boxes[idx].pose.orientation.x = q_update[0];
		updated_bbox.boxes[idx].pose.orientation.y = q_update[1];
		updated_bbox.boxes[idx].pose.orientation.z = q_update[2];
		updated_bbox.boxes[idx].pose.orientation.w = q_update[3];
		updated_bbox.boxes[idx].dimensions.x = state.w;
		updated_bbox.boxes[idx].dimensions.y = state.h;
		updated_bbox.boxes[idx].dimensions.z = state.l;
		updated_bbox.boxes[idx].value = Input_bbox.boxes[idx].value;
		updated_bbox.boxes[idx].label = Input_bbox.boxes[idx].label+2; // To make different color with GT
		
		arr_estimated_bbox_push.boxes.push_back(estimated_bbox.boxes[idx]);
		arr_updated_bbox_push.boxes.push_back(updated_bbox.boxes[idx]);
						
		pub_estimated_bbox.publish(arr_estimated_bbox_push);
		pub_updated_bbox.publish(arr_updated_bbox_push);		
		}
	}
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "ekf");
  ros::NodeHandle nh;
  ros::Subscriber sub_bbox, sub_odom;  

  sub_bbox = nh.subscribe<jsk_recognition_msgs::BoundingBoxArray> ("/pp_boxes", 100, ekf_process);
  sub_odom = nh.subscribe<nav_msgs::Odometry> ("/odom", 100, odom_callback);
  
  pub_estimated_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray> ("Estimated_boxes", 100);
  pub_updated_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray> ("updated_boxes", 100);
 
  ros::spin();
}

