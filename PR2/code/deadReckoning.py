# import * from pr2_utils
import matplotlib.pyplot as plt
import numpy as np
import transforms3d
from scipy import signal
from tqdm import tqdm, trange

import pr2_utils
from utils import *

dataset = 21
INIT_R = np.eye(3)
INIT_P = np.array([0,0,0]).astype('float')

def populateMap(lidar_ranges,LIDAR_ANGLES,lidar_ind,robot_R,robot_p,MAP):
    lidar_range_thresh,lidar_angles_thresh = thresh_lidar(lidar_ranges[...,lidar_ind],LIDAR_ANGLES)
    xs0,ys0 = lidar_polar2cart(lidar_range_thresh,lidar_angles_thresh)
    xs0 = xs0+0.29833 #Conversion from lidar to IMU frame
    xs0,ys0,_ = ((robot_R@np.stack((xs0,ys0,np.ones_like(xs0)))).T+robot_p).T
    xis,yis = lidar_cart2cell(xs0,ys0,MAP)

    bresenham_coords = np.array([[0,0]])
    robot_p_cell = lidar_cart2cell(robot_p[0],robot_p[1],MAP)
    for xi_i,yi_i in zip(xis,yis):
        bresenham_coords = np.vstack((bresenham_coords,pr2_utils.bresenham2D(robot_p_cell[0],robot_p_cell[1],xi_i,yi_i).T)).astype(int)
        # bresenham_coords.append((pr2_utils.bresenham2D(robot_p[0],robot_p[1],xs_i,ys_i).T))
    # bresenham_coords = np.unique(bresenham_coords,axis=0)
    bresenham_good_ind = clean_indices(bresenham_coords.T[0],bresenham_coords.T[1],MAP)
    MAP['map'][bresenham_coords.T[0][bresenham_good_ind],bresenham_coords.T[1][bresenham_good_ind]]-=np.log(4)

    indGood = clean_indices(xis,yis,MAP)
    MAP['map'][xis[indGood],yis[indGood]] += 2*np.log(4)



if __name__ == "__main__":
    with np.load("../data/Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

    with np.load("../data/Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans 0.5 30

    with np.load("../data/Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    with np.load("../data/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    LIDAR_ANGLES = np.arange(lidar_angle_min,lidar_angle_max+(lidar_angle_increment*0.01),lidar_angle_increment)
    cum_encoder = np.cumsum(encoder_counts,axis=1)
    r_avg_whl, l_avg_whl = (cum_encoder[0]+cum_encoder[2])*(0.0022/2), (cum_encoder[1]+cum_encoder[3])*(0.0022/2)
    l_velocity = np.gradient(r_avg_whl,encoder_stamps,edge_order=1)
    r_velocity = np.gradient(l_avg_whl,encoder_stamps,edge_order=1)
    velocity_encoder = (l_velocity+r_velocity)/2
    omega = imu_angular_velocity[2]


    robot_R = INIT_R; robot_p = INIT_P; robot_th = 0;
    MAP = generate_MAP()
    trajectory = []
    pose_th_his = []

    for encoder_ind,(t,dt) in tqdm(enumerate(zip(encoder_stamps[1:],encoder_stamps[1:]-encoder_stamps[:-1])),total=encoder_stamps.shape[0]):
        lidar_ind = np.argmin(np.abs(lidar_stamsp-t))
        imu_ind = np.argmin(np.abs(imu_stamps-t))
        
        dir_vec = np.array([np.cos(robot_th),np.sin(robot_th),0])
        # vs = np.expand_dims(np.random.normal(velocity_imu[encoder_ind],SIGMA_V,N),-1)*dir_vecs
        robot_p=robot_p+(velocity_encoder[encoder_ind]*dt*dir_vec)
        # print(dt)
        trajectory+=[list(robot_p)]
        
        # ws = np.random.normal(omega[imu_ind],SIGMA_W,N)
        robot_th = robot_th+(omega[imu_ind]*dt)
        pose_th_his+=[robot_th]
        
        robot_R = rotationMatrix(robot_th)
        
        populateMap(lidar_ranges,LIDAR_ANGLES,lidar_ind,robot_R,robot_p,MAP)
    trajectory = np.array(trajectory)

    fig1 = plt.figure()
    plt.plot(pose_th_his)
    plt.title(f'Theta Dataset {dataset}')
    fig1.savefig(f'./images/Theta_{dataset}.png')

    
    fig2 = plt.figure()
    plt.imshow(np.vectorize(sigmoid)(MAP['map'])>0.2);
    plt.plot(lidar_cart2cell(trajectory[...,1],trajectory[...,0],MAP)[0],lidar_cart2cell(trajectory[...,1],trajectory[...,0],MAP)[1])
    plt.title(f'Dead Reckoning Occupancy grid map Dataset {dataset}')
    fig2.savefig(f'./images/DeadReconing_{dataset}.png')





