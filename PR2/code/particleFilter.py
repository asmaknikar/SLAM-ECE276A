import matplotlib.pyplot as plt
import numpy as np
import transforms3d
from scipy import signal
from tqdm import tqdm, trange
import pickle

import pr2_utils
from deadReckoning import *
from utils import *
import cv2

dataset = 21
N = 20
SIGMA_V = 0.1
SIGMA_W = 0.01
MAP_RESOLUTION = 0.05
INIT_R = np.eye(3)
INIT_P = np.array([0,0,0]).astype('float')
R_cam = np.array([[0.92,-0.2,0.35],[ 0.21,0.98,  0],[-0.34,0.07, 0.94]])
p_cam = np.array([0.18, 0.005, 0.36])
drift = False

disp_path = "../data/dataRGBD/Disparity20/"
rgb_path = "../data/dataRGBD/RGB20/"




def get_correlation(Y,MAP,correlation_grid_size_factor=2):
    
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    
    x_range = np.arange(-correlation_grid_size_factor*MAP['res'],(correlation_grid_size_factor*MAP['res'])+MAP['res'],MAP['res'])
    y_range = np.arange(-correlation_grid_size_factor*MAP['res'],(correlation_grid_size_factor*MAP['res'])+MAP['res'],MAP['res'])
    c = pr2_utils.mapCorrelation(sigmoid_map(MAP['map'])>0.5,x_im,y_im,Y,x_range,y_range)
    x_cmax,y_cmax = np.unravel_index(c.argmax(), c.shape)
    # print(c,x_range[x_cmax],y_range[y_cmax],x_cmax,y_cmax)
    
    return np.max(c),np.array([x_range[x_cmax],y_range[y_cmax],0])

def populateMap_particle(Y,robot_p,MAP):
    xs0,ys0 = Y

    
    xis,yis = lidar_cart2cell(xs0,ys0,MAP)

    bresenham_coords = np.array([[0,0]])
    robot_p_cell = lidar_cart2cell(robot_p[0],robot_p[1],MAP)
    for xi_i,yi_i in zip(xis,yis):
        bresenham_coords = np.vstack((bresenham_coords,pr2_utils.bresenham2D(robot_p_cell[0],robot_p_cell[1],xi_i,yi_i).T)).astype(int)
    bresenham_good_ind = clean_indices(bresenham_coords.T[0],bresenham_coords.T[1],MAP)
    MAP['map'][bresenham_coords.T[0][bresenham_good_ind],bresenham_coords.T[1][bresenham_good_ind]]-=np.log(4)

    indGood = clean_indices(xis,yis,MAP)
    MAP['map'][xis[indGood],yis[indGood]] += 2*np.log(4)


def get_texture(disp_ind,rgb_ind):
    imd = cv2.imread(disp_path+f'disparity20_{disp_ind+1}.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(rgb_path+f'rgb20_{rgb_ind+1}.png')[...,::-1] # (480 x 640 x 3)

    # print(imc.shape)

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    
    return np.stack((z[valid],-x[valid],-y[valid])), (imc[rgbv[valid].astype(int),rgbu[valid].astype(int)]/255.0)





if __name__=="__main__":
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




    MAP = generate_MAP()
    robot_p = INIT_P;
    correlation_grid_size_factor = 1
    # robot_p = np.expand_dims(robot_p,0)
    robot_ps = np.tile(robot_p,(N,1))
    robot_th = np.tile(0,(N))
    alphas = np.tile(1/N,(N))
    trajectories = []
    best_trajectory = []
    robot_th_histories = []
    best_robot_th = []
    for encoder_ind,(t,dt) in tqdm(enumerate(zip(encoder_stamps[1:],encoder_stamps[1:]-encoder_stamps[:-1])),total=encoder_stamps.shape[0]):
        lidar_ind = np.argmin(np.abs(lidar_stamsp-t))
        imu_ind = np.argmin(np.abs(imu_stamps-t))
        
        dir_vecs = np.stack([np.cos(robot_th),np.sin(robot_th),np.zeros_like(robot_th)]).T
        vs = np.expand_dims(np.random.normal(velocity_encoder[encoder_ind],SIGMA_V,N),-1)*dir_vecs
        robot_ps=robot_ps+(vs*dt)
        
        ws = np.random.normal(omega[imu_ind],SIGMA_W,N)
        robot_th = robot_th+(ws*dt)
        
        robot_Rs = np.apply_along_axis(lambda x:rotationMatrix(x[0]),1,np.expand_dims(robot_th,-1))
        
        lidar_range_thresh,lidar_angles_thresh = thresh_lidar(lidar_ranges[...,lidar_ind],LIDAR_ANGLES)
        xs0,ys0 = lidar_polar2cart(lidar_range_thresh,lidar_angles_thresh)
        xs0 = xs0+0.29833 #Conversion from lidar to IMU frame
        Ys = (((robot_Rs)@np.stack((xs0,ys0,np.zeros_like(xs0))))+np.expand_dims(robot_ps,-1))[:,:2,:]
        # corrs = np.apply_along_axis(lambda ysi:get_correlation(Ys[ysi[0]],MAP,correlation_grid_size_factor),1,np.expand_dims(np.arange(N),1))
        cs = [];delta_ps = [];
        for ni,Y in enumerate(Ys):
            c, delta_p = get_correlation(Y,MAP,correlation_grid_size_factor)
            # c = 1
            # print(c)
            alphas[ni]*=(c+1e-6)
            cs+=[c]
            delta_ps+=[delta_p]
        if(drift):
            robot_ps[np.array(cs)>0]+=np.array(delta_ps)[np.array(cs)>0]
        alphas = alphas/np.sum(alphas)
        alpha_ind = np.argmax(alphas)
        

        populateMap_particle(Ys[alpha_ind],robot_ps[alpha_ind],MAP)
        # populateMap(lidar_ranges,LIDAR_ANGLES,lidar_ind,robot_Rs[alpha_ind],robot_ps[alpha_ind],MAP)
        
        trajectories+=[robot_ps]
        robot_th_histories += [robot_th]
        best_trajectory+=[robot_ps[alpha_ind]]
        best_robot_th+=[robot_th[alpha_ind]]
        
        n_eff = 1/np.sum(alphas**2)
        if(n_eff<(N/20)):
            print("refactoring")
            robot_ps = robot_ps[np.random.choice(np.arange(N),size=N,p=alphas)]
            robot_th = robot_th[np.random.choice(np.arange(N),size=N,p=alphas)]
            alphas = np.tile(1/N,(N))
    fig1 = plt.figure()
    for i in np.array(trajectories):
        plt.plot(i[:,0],i[:,1])
    plt.title(f'Trajectories for {N} particles')
    fig1.savefig(f'./images/Trajectory_{N}_particles_{dataset}_drift_{drift}.png')

    fig2 = plt.figure()
    for i in np.array(trajectories):
        plt.plot(i[:,0],i[:,1])
    plt.title(f'Theta for {N} particles')
    fig2.savefig(f'./images/Theta_{N}_particles_{dataset}_drift_{drift}.png')

    fig3 = plt.figure()
    plt.imshow(np.vectorize(sigmoid)(MAP['map'])>0.2)
    # plt.plot(lidar_cart2cell(trajectory[...,1],trajectory[...,0],MAP)[0],lidar_cart2cell(trajectory[...,1],trajectory[...,0],MAP)[1])
    plt.title(f'Occupancy grid map Dataset {dataset} N={N}')
    fig3.savefig(f'./images/Map_{N}_particles_{dataset}_drift_{drift}.png')

    with open(f'./images/MAP_d{dataset}_N_{N}.pickle', 'wb') as handle:
        pickle.dump(MAP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    MAP['texture'] = np.zeros(list(MAP['map'].shape)+[3])

    for encoder_ind,(t,dt) in tqdm(enumerate(zip(encoder_stamps[1:],encoder_stamps[1:]-encoder_stamps[:-1])),total=encoder_stamps.shape[0]):
        lidar_ind = np.argmin(np.abs(lidar_stamsp-t))
        imu_ind = np.argmin(np.abs(imu_stamps-t))
        rgb_ind = np.argmin(np.abs(rgb_stamps-t))
        disp_ind = np.argmin(np.abs(disp_stamps-t))
        
        texture_ps,texture_cs = get_texture(disp_ind,rgb_ind)
        texture_ps_body = ((R_cam@texture_ps)+np.expand_dims(p_cam,-1))
        xps,yps,zps = ((rotationMatrix(best_robot_th[encoder_ind])@texture_ps_body)+np.expand_dims(best_trajectory[encoder_ind],-1))
        
        
        on_ground_ind = zps<0.127
        xps = xps[on_ground_ind]
        yps = yps[on_ground_ind]
        texture_cs_g = texture_cs[on_ground_ind]
        
        xps_cell,yps_cell = lidar_cart2cell(xps,yps,MAP)
        clean_ind = clean_indices(xps_cell,yps_cell,MAP)
        
        MAP['texture'][xps_cell[clean_ind],yps_cell[clean_ind]] = texture_cs_g[clean_ind]


    fig4 = plt.figure()
    # MAP['texture'][MAP['map']==0.5]=0
    plt.imshow(MAP['texture']*np.expand_dims(MAP['map']<0.2,-1));
    # plt.plot(lidar_cart2cell(trajectory[...,1],trajectory[...,0],MAP)[0],lidar_cart2cell(trajectory[...,1],trajectory[...,0],MAP)[1])
    plt.title(f'Texture Map Dataset {dataset} N={N}')
    fig4.savefig(f'./images/TextureMap_{N}_particles_{dataset}_drift_{drift}.png')


        




        
