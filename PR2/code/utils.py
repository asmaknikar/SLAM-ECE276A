import matplotlib.pyplot as plt
import numpy as np
import transforms3d
from scipy import signal
from tqdm import tqdm, trange

import pr2_utils

MAP_RESOLUTION = 0.05

def show_lidar(ranges,angles):
    # angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    # ranges = np.load("test_ranges.npy")
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, ranges)
    ax.set_rmax(10)
    ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')
    plt.show()

    
def generate_MAP(map_range=30,map_resolution=MAP_RESOLUTION):
    MAP = {}
    MAP['res']   = map_resolution #meters
    MAP['xmin']  = -map_range  #meters
    MAP['ymin']  = -map_range
    MAP['xmax']  =  map_range
    MAP['ymax']  =  map_range
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    return MAP

def thresh_lidar(ranges,angles):
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    return ranges,angles

def lidar_polar2cart(ranges,angles):
    # xy position in the sensor frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)

    # convert position in the map frame here 
    Y = np.stack((xs0,ys0))
    return Y

def lidar_cart2cell(xs0,ys0,MAP):
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return(np.stack((xis,yis)))


def lidar_polar2cell(ranges,angles,MAP):
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)

    # convert position in the map frame here 
    # Y = np.stack((xs0,ys0))

    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return(np.stack((xis,yis)))

def clean_indices(xis,yis,MAP):
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    return(indGood)

def rotationMatrix(theta):
    return(np.array([[np.cos(theta),-np.sin(theta),0],
                     [np.sin(theta),np.cos(theta),0],
                     [0,0,1]]))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_map(m):
    return np.exp(m) / (1 + np.exp(m))

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)
