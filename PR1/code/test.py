from base64 import encode
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

config.update("jax_debug_nans", True)
IMU_FOLDER = "../data/trainset/imu/"
CAM_FOLDER = "../data/trainset/cam/"
VICON_FOLDER = "../data/trainset/vicon/"

IMU_TEST_FOLDER = "../data/testset/imu/"
CAM_TEST_FOLDER = "../data/testset/cam/"

FIGURES_FOLDER = "./figures/"
PICKLES_FOLDER = "./pickles/"

SENSITIVITY_ACC_MVPG = 300
SENSITIVITY_PRY_MVPDEGPS = 3.33
VREF_MV = 3300
EPS = 1e-6
ALPHA = 0.01


###########################
# BASIC BUILDER FUNCTIONS#
##########################
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

def getDataForExp(i,imuFolder=IMU_FOLDER,viconFolder=VICON_FOLDER):
    imudata = read_data(f'{imuFolder}imuRaw{i}.p')
    if(viconFolder is None):
        vicondata = None
    else:
        vicondata = read_data(f'{viconFolder}viconRot{i}.p')
    return imudata,vicondata


##########################
#  USELESS FUNCITONS
#############################3
def get_changepoints(imu1data):
    min_cuts,max_cuts = [],[]
    for i in tqdm(range(6)):
        algo = rpt.Pelt(model='rbf', min_size=3, jump=10).fit(imu1data['vals'][0])
        my_bkps = algo.predict(pen=3)
        min_cuts.append(min(my_bkps))
        max_cuts.append(max(my_bkps))
    print(min_cuts,max_cuts)
    return(int((min(min_cuts)*0.8)),int(max(max_cuts)+(imu1data['vals'].shape[1]-max(max_cuts))*0.2))

def plotImuData(ub_imu1data,ts):
    fig, axes = plt.subplots(2, 3, figsize=(16,12))
    for i in range(6):
        axes.flatten()[i].plot(ts,ub_imu1data[i])

# ###################################
# QUATERNION SUPORT
#########################################
@jit
def qMult(q,p):
    qs = q[0]
    ps = p[0]
    qv = q[1:]
    pv = p[1:]
    return (jnp.hstack((jnp.array([qs*ps-jnp.dot(pv,qv)]),(qs*pv)+(ps*qv)+jnp.cross(qv,pv))))

@jit
def qLog(q):
    q+=EPS
    qs = q[0]
    qv = q[1:]
    norm = jnp.linalg.norm(q)
    return jnp.hstack((
        jnp.array([jnp.log(norm)]),
        (qv/(jnp.linalg.norm(qv)+EPS))*jnp.arccos(qs/(norm+EPS))
    ))

@jit
def qExp(q):
    qs = q[0]
    qv = q[1:]
    normv = jnp.linalg.norm(qv)
    mult = jnp.exp(qs)
    return mult*jnp.hstack((
        jnp.array([jnp.cos(normv)]),
        (qv/(normv+EPS))*jnp.sin(normv)
    ))

@jit
def qConjugate(q):
    return jnp.hstack((
        q[0:1],
        -q[1:]
    ))

@jit
def qInv(q):
    return(qConjugate(q)/((jnp.linalg.norm(q)**2)+EPS))

@jit
def qUnit(q):
    return(q/((jnp.linalg.norm(q))+EPS))
    
###############################################################
# MOTION MODEL
###############################################################

def motionModelQ_tp1(qt,wt,dt):
    return qMult(qt,qExp(jnp.hstack((jnp.array([0]),(dt*jnp.array(wt))/2))))

def motionModel(ub_imu1data, dts):
    # wts = ub_imu1data[3:].T
    # wts = wts[:,[2,0,1]]
    # wts = ub_imu1data[3:].T
    wts = ub_imu1data[3:].T[:,[1,2,0]]
    
    qt = jnp.array([1,0,0,0])
    qts = [qt]
    qts_euler = [transforms3d.euler.quat2euler(qt)]
    for dt,wt in tqdm(zip(dts,wts[:-1]),total=dts.shape[0]):
        # wt = jnp.array([wt[1],wt[2],wt[0]])
        qt_p1 = motionModelQ_tp1(qt,wt,dt)
        qt = qt_p1
        qts+=[qt_p1]
        qts_euler+=[transforms3d.euler.quat2euler(qt_p1)]
    qts = jnp.vstack(qts)
    qts_euler = jnp.vstack(qts_euler)
    return qts,qts_euler

#######################################################
# COST FUNCTION
#####################################################
def h(qt):
    return qMult(qMult(qInv(qt),jnp.array([0,0,0,1])),qt)

def cost_sec_summation(at,qt):
    # (ub_imu1data[:3,0]*jnp.array([-1,-1,1]))-h(qts[0])[1:]
    return jnp.linalg.norm(at-h(qt)[1:])**2

def cost_fir_summation(qt_p1,qt,dt,wt):
    logTerm = qMult(qInv(qt_p1),motionModelQ_tp1(qt,wt,dt))
    term = qLog(logTerm+EPS)
    return (2*jnp.linalg.norm(term))**2

def cost_fn(qts,dts,ats):
    qt_p1s = qts[1:]
    qts_cf = qts[:-1]
    wts = ub_imu1data[3:][[1,2,0]].T[:-1]

    sum1 = jnp.sum(jnp.apply_along_axis(lambda x:cost_fir_summation(x[:4],x[4:8],x[8],x[9:]),1,
                                 jnp.hstack((qt_p1s,qts_cf,jnp.expand_dims(dts,-1),wts))                             
                                ))
    sum2 = jnp.sum(jnp.apply_along_axis(lambda x:cost_sec_summation(x[:3],x[3:]),1,jnp.hstack((ats,qts))))
    return 0.5*(sum1+sum2)

##########################################################
###
#######################################################
def gradientDescent(qts_init,dts,ats):
    qts = qts_init
    prev_cf_val = cost_fn(qts,dts,ats)

    grad_cost_fn = jacrev(cost_fn)

    pbar = tqdm(range(100))
    cf_vals = []

    pbar = tqdm(range(1000))
    for i in pbar:
        qts = qts-(ALPHA*grad_cost_fn(qts,dts,ats))
        qts = jnp.apply_along_axis(qUnit,1,qts)
        # print(jnp.apply_along_axis(jnp.linalg.norm,1,qts))
        cf_val = cost_fn(qts,dts,ats)
        pbar.set_description(f'Cost fn {cf_val}')

        cf_vals+=[cf_val]
        if(prev_cf_val-cf_val<0.0004*prev_cf_val):
            break
        prev_cf_val = cf_val
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(cf_vals)
    return qts,fig


####################################################




def get_unbiased_scaled_imu(imu1data,cp_l=None,cp_l1=None):
    if(cp_l is None):
        cp_l,cp_h = get_changepoints(imu1data)
    if(cp_l1 is None):
        cp_l1 = 0

    bias = np.mean(imu1data['vals'][:,cp_l1:cp_l],1)
    scalef_acc = (VREF_MV/ SENSITIVITY_ACC_MVPG)/1023
    scalef_rpy = (VREF_MV/ (SENSITIVITY_PRY_MVPDEGPS*(180/np.pi)))/1023
    scalef = np.array([scalef_acc,scalef_acc,scalef_acc,scalef_rpy,scalef_rpy,scalef_rpy])
    ub_imu1data = (imu1data['vals']-np.expand_dims(bias,-1))*np.expand_dims(scalef,-1)

    scaled_imu1data = (imu1data['vals']*np.expand_dims(scalef,-1))
    bias_scaled = np.mean(scaled_imu1data[:,cp_l1:cp_l],1)
    # ub_imu1data = scaled_imu1data-np.expand_dims(bias_scaled,-1)

    ub_imu1data[2]+=1

    print(f'bias {bias}')
    print(f'scale factor {scalef}')
    print(f'scaled bias {bias_scaled}')

    # for i in range(6):
    #     plt.plot(imu1data['ts'].T[:cp_l],imu1data['vals'][i][:cp_l])
    # plt.show()

    return ub_imu1data

def motionModel_verification_plot(vicon1data,ub_imu1data,imu1data,qts_euler=None):
    viconEulers = []
    for i in range(vicon1data['rots'].shape[2]):
        viconEulers+=[transforms3d.euler.mat2euler(vicon1data['rots'][...,i])]
    viconEulers = np.vstack(viconEulers)

    dts = (imu1data['ts'][0][1:]-imu1data['ts'][0][:-1])
    if(qts_euler is None):
        qts_init,qts_euler = motionModel(ub_imu1data,dts)
    # imuEulers = []
    # for qt in np.array(qts_init):
    #     imuEulers+=[transforms3d.euler.quat2euler(qt)]
    # imuEulers = np.vstack(imuEulers)

    fig, axes = plt.subplots(3,1)
    for i in range(3):
        axes.flatten()[i].plot(vicon1data['ts'][0],viconEulers[:,i],label="VICON")
        # axes.flatten()[i].plot(imu1data['ts'][0],imuEulers[:,i])    
        axes.flatten()[i].plot(imu1data['ts'][0],qts_euler[:,i],label='MOTION MODEL')
        axes.flatten()[i].legend()
    return fig

def motionModel_acc_verification_plot(ub_imu1data,ats,imu1data,qts_init=None):
    dts = (imu1data['ts'][0][1:]-imu1data['ts'][0][:-1])

    if(qts_init is None):
        qts_init,qts_euler = motionModel(ub_imu1data,dts)
    fig, axes = plt.subplots(3,1)
    for i in range(3):
        axes.flatten()[i].plot(imu1data['ts'][0],ats[:,i],label="IMU")
        axes.flatten()[i].plot(imu1data['ts'][0],jnp.apply_along_axis(h,1,qts_init)[:,i+1],label="h")
        axes.flatten()[i].legend()
    return fig

def optimised_verification_plot(vicon1data,qts,imu1data):
    viconEulers = []
    for i in range(vicon1data['rots'].shape[2]):
        viconEulers+=[transforms3d.euler.mat2euler(vicon1data['rots'][...,i])]
    viconEulers = np.vstack(viconEulers)

    optImuEulers = []
    for qt in np.array(qts):
        optImuEulers+=[transforms3d.euler.quat2euler(qt)]
    optImuEulers = np.vstack(optImuEulers)

    fig, axes = plt.subplots(3,1)
    for i in range(3):
        axes.flatten()[i].plot(vicon1data['ts'][0],viconEulers[:,i],label="VICON")
        axes.flatten()[i].plot(imu1data['ts'][0],optImuEulers[:,i],label='OPTIMISED MODEL')
        axes.flatten()[i].legend()
    return fig







    
if __name__ == "__main__":
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    MODE = "test"
    print(f'We are in {MODE} mode')
    if(MODE=="train"):
        cutData = {1:632,2:720,3:512,4:512,5:416,6:368,7:416,8:464,9:456}
    else:
        cutData = {10:408,11:360}


    for exp in cutData:
        # exp = 1
        if(MODE=="train"):
            imu1data,vicon1data = getDataForExp(exp,IMU_FOLDER,VICON_FOLDER)
        else:
            imu1data,_ = getDataForExp(exp,IMU_TEST_FOLDER,None)
        ub_imu1data = get_unbiased_scaled_imu(imu1data,cutData[exp])
        
        fig, axes = plt.subplots(2, 3, figsize=(16,12))
        for i in range(6):
            axes.flatten()[i].plot(imu1data['ts'].T,ub_imu1data[i])
        fig.savefig(f'{FIGURES_FOLDER}unbiasedImuData_{TIMESTAMP}_{MODE}_SET_{exp}.jpg')

        dts = (imu1data['ts'][0][1:]-imu1data['ts'][0][:-1])
        ats = ub_imu1data[:3].T*jnp.array([-1,-1,1])

        qts_init,qts_euler = motionModel(ub_imu1data,dts)

        if(MODE=="train"):
            motionModel_verification_plot(vicon1data,ub_imu1data,imu1data,qts_euler).savefig(f'{FIGURES_FOLDER}motionModelVerification_{TIMESTAMP}_SET_{exp}.jpg')
        motionModel_acc_verification_plot(ub_imu1data,ats,imu1data,qts_init).savefig(f'{FIGURES_FOLDER}AccVerification_{TIMESTAMP}_{MODE}_SET_{exp}.jpg')
        # plt.show();time.sleep(0.5)

        qts,fig1 = gradientDescent(qts_init,dts,ats)
        fig1.savefig(f'{FIGURES_FOLDER}GradDesc_costfn_{TIMESTAMP}_{MODE}_SET_{exp}.jpg')

        if(MODE=="train"):
            optimised_verification_plot(vicon1data,qts,imu1data).savefig(f'{FIGURES_FOLDER}OptimisedComparison_{TIMESTAMP}_SET_{exp}.jpg')

        with open(f'{PICKLES_FOLDER}qts_{TIMESTAMP}_{MODE}_{exp}.npy', 'wb') as f:
            np.save(f,np.array(qts))
        print(f'set {exp} done !!!!!!!!')
        


    # plt.show();time.sleep(0.5)








    # cp_l,cp_h = get_changepoints(imu1data)
