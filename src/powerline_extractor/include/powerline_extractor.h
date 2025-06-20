#ifndef POWERLINE_EXTRACTOR_H
#define POWERLINE_EXTRACTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <pcl_ros/transforms.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>

#include "point_cloud_preprocessor.h"
#include "power_line_coarse_extractor_s.h"




#include "power_line_filter.h"

#include "power_line_fine_extraction.h"

#include "obstacle_analyzer.h"

// 前向声明粗提取器
class PowerlineCoarseExtractor;

class PowerlineExtractor {
public:
    PowerlineExtractor(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~PowerlineExtractor();

private:
    // 回调函数
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    
    // 参数检查和初始化

    // 在现有私有函数中添加
    void initializeAccumulateCloud();    // 初始化预处理管道

    void loadParameters();
    void initializePublishers();
    void initializeSubscribers();
    void initializeCoarseExtractor();

    void initializeCoarseFilter();
    void initializeFineExtractor();
    
    // 点云变换
    bool transformPointCloud(const sensor_msgs::PointCloud2::ConstPtr& input_msg,
                           sensor_msgs::PointCloud2& transformed_msg);
    
    // 聚类处理
    void clusterPowerlines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud);
    
    // 发布点云
    void publishPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr& original_cloud,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& preprocessed_cloud,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud,
                          const std_msgs::Header& header);

private:
    // ROS 节点句柄
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 订阅器和发布器

    // 在现有发布器后添加
    ros::Publisher preprocessor_cloud_pub_;    //预处理后的点云
    ros::Publisher extractor_s_cloud_pub_;    //粗提取_s后的点云

    ros::Publisher fine_extractor_cloud_pub_;      // octree累积点云发布器（用于调试）
    ros::Publisher coarse_filter_cloud_pub_;  
    ros::Subscriber point_cloud_sub_;
    ros::Publisher original_cloud_pub_;
    ros::Publisher preprocessed_cloud_pub_;
    ros::Publisher powerline_cloud_pub_;
    ros::Publisher clustered_powerline_cloud_pub_;


    ros::Publisher obb_marker_pub;

    ros::Publisher powerlines_distance_cloud_pub_;
    // 注意：移除了non_ground_cloud_pub_，因为不再进行地面点移除
    
    // TF变换
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;


    //点云数据预处理
    std::unique_ptr<PointCloudPreprocessor> preprocessor_;  //实例化类
    pcl::PointCloud<pcl::PointXYZI>::Ptr preprocessor__output_cloud_;  //输出预处理后的点云
    // pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZI>& octree_;  //输出的octree     octree = preprocessor.getOctree();

    //粗提取_s
    std::unique_ptr<PowerLineExtractor> extractor_s_; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractor_s__output_cloud_;

    //距离可视化
    std::unique_ptr<ObstacleAnalyzer> analyzer_; 
    std::vector<OrientedBoundingBox> obbs_;













    


    //粗步过滤
    std::unique_ptr<PowerLineFilter> coarse_filter_;

    //精提取器
    std::unique_ptr<PowerLineFineExtractor> fine_extractor_;

    
    
    // 参数
    std::string lidar_topic_;
    std::string lidar_frame_;
    std::string target_frame_;
    
    // 预处理参数（传递给粗提取器）
    double voxel_size_;
    double min_range_;
    double max_range_;
    double min_height_;
    double max_height_;
    
    // PCA参数（传递给粗提取器）
    double pca_radius_;
    double linearity_threshold_;
    
    // 聚类参数
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
    // 离群点移除参数（传递给粗提取器）
    int outlier_mean_k_;
    double outlier_std_thresh_;
    
    // 点云对象


    pcl::PointCloud<pcl::PointXYZI>::Ptr original_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr preprocessed_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud_;  // 保留用于兼容性，但不使用
    pcl::PointCloud<pcl::PointXYZI>::Ptr powerline_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr clustered_powerline_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pc_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr fine_extract_cloud_;
    
    // 处理标志
    bool first_cloud_received_;
    ros::Time last_process_time_;
    double process_frequency_;
};

#endif // POWERLINE_EXTRACTOR_H