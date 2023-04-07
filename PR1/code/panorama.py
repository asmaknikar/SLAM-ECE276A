import datetime
import pickle
import sys
import time

import jax.numpy as jnp
import numpy as np
import ruptures as rpt
import transforms3d
from jax import grad, jacrev, jit, random, vmap
from jax.config import config
from matplotlib import pyplot as plt
from tqdm import tqdm

IMU_FOLDER = "../data/trainset/imu/"
CAM_FOLDER = "../data/trainset/cam/"
VICON_FOLDER = "../data/trainset/vicon/"

IMU_TEST_FOLDER = "../data/testset/imu/"
CAM_TEST_FOLDER = "../data/testset/cam/"
QT_PICKLE_PREFIX = "qts_2023_02_08T19_45_45_test_"

FIGURES_FOLDER = "./figures/"
PICKLES_FOLDER = "./pickles/"


def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')  # need for python 3
        return d

def getDataForExp(i,camFolder=CAM_FOLDER,viconFolder=VICON_FOLDER):
    camData = read_data(f'{camFolder}cam{i}.p')
    if(viconFolder is None):
        vicondata = None
    else:
        vicondata = read_data(f'{viconFolder}viconRot{i}.p')
    return camData,vicondata


if __name__=="__main__":
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    MODE = "test"
    print(f'We are in {MODE} mode')
    if(MODE=="train"):
        cutData = [1,2,8,9]
    else:
        cutData = [10,11]
    for exp in cutData:
        if(MODE=="train"):
            cam1data,vicon1data = getDataForExp(exp,CAM_FOLDER,VICON_FOLDER)
            cam_ts = np.squeeze(cam1data['ts'],0)
            imu_ts = np.squeeze(vicon1data['ts'],0)
            rots = vicon1data['rots']
        else:
            cam1data,_ = getDataForExp(exp,CAM_TEST_FOLDER,None)
            cam_ts = np.squeeze(cam1data['ts'],0)
            imu_ts = np.squeeze(read_data(f'{IMU_TEST_FOLDER}imuRaw{exp}.p')['ts'],0)
            qts = np.load(f'{PICKLES_FOLDER}{QT_PICKLE_PREFIX}{exp}.npy')
            rots = np.apply_along_axis(lambda x: transforms3d.quaternions.quat2mat(x),-1,qts).transpose(1,2,0)
        latlin = np.linspace(np.pi/6,-np.pi/6,320)
        lonlin = np.linspace(-np.pi/8,np.pi/8,240)
        sph_mesh = np.array(np.meshgrid(latlin,lonlin))
        # rho = np.ones_like(lat)

        sph_coords = (sph_mesh).transpose(1,2,0)
        cart_coords = np.apply_along_axis(lambda x: [np.cos(x[1])*np.cos(x[0]),-np.cos(x[1])*np.sin(x[0]),-np.sin(x[1])],-1,sph_coords)

        canvas = np.ones((960,1280,3),dtype='uint8')*255

        for i in tqdm(range(0,cam1data['cam'].shape[-1])):
            img = cam1data['cam'][...,i]
            nearest_idx = np.argmin(np.abs(imu_ts-cam_ts[i]))
            rot = rots[...,nearest_idx]

            # rot_cart_coords = np.apply_along_axis(lambda x: transforms3d.quaternions.rotate_vector(x,qt),-1,cart_coords)
            # rot_cart_coords = np.apply_along_axis(lambda x: rot@x,-1,cart_coords)
            # rot_sph_coords = np.apply_along_axis(lambda x: asSpherical(x),-1,rot_cart_coords)
            # canvas_coords = np.apply_along_axis(lambda x: toUnrolledCylinder(x),-1,rot_sph_coords)
            # canvas[canvas_coords.reshape((-1,2))[:,1],canvas_coords.reshape((-1,2))[:,0]] = img.reshape((-1,3))

            # rot_cart_coords = np.apply_along_axis(lambda x: transforms3d.quaternions.rotate_vector(x,qt),-1,cart_coords)
            rot_cart_coords = np.apply_along_axis(lambda x: rot@x,-1,cart_coords)
            # rot_cart_coords = (rot@cart_coords.reshape(-1,3).T).reshape(240,320,3)
            rot_sph_coords = np.apply_along_axis(lambda x: [np.arcsin(-x[2]/np.linalg.norm(x)),np.arctan2(x[1],x[0]),np.linalg.norm(x)],-1,rot_cart_coords)
            canvas_coords = np.apply_along_axis(lambda x: np.array([(x[0]+(np.pi/2))*(960/np.pi),(x[1]+(np.pi))*(1280/(2*np.pi))]).astype(int),-1,rot_sph_coords)
            canvas[canvas_coords[...,0],canvas_coords[...,1]] = img

        fig, axes = plt.subplots(1, 1)
        axes.imshow(canvas)
        axes.set_title(f'Dataset{exp}')
        fig.savefig(f'{FIGURES_FOLDER}panorama_{TIMESTAMP}_{MODE}_SET_{exp}.jpg')







