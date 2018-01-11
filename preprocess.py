
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
import scipy.io

from rfdata import RFdata
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_usctsim_result(result_path):
    """
    * argument
        - result_path : by USCTSim
        - list_pos : list of positions for preprocessing
        - offset : time offset 
    * return
        - map_speed
        - map_dens
        - preprocessed
    """

    assert os.path.exists(result_path)

    param_path = os.path.join(result_path, 'param.json')
    sensor_path = os.path.join(result_path, 'sensor.mat')
    points_path = os.path.join(result_path, 'mask_points.mat')
    rfdata_path = os.path.join(result_path, 'rfdata.mat')
    medium_path = os.path.join(result_path, 'medium.mat')

    with open(param_path, 'r') as f: param = json.load(f)
    arr_cart_pos = scipy.io.loadmat(sensor_path)["sensor"]["mask"][0][0].T
    mask_points = scipy.io.loadmat(points_path)["mask_points"]
    mat_rfdata = h5py.File(rfdata_path, "r")["rfdata"]
    map_speed = h5py.File(medium_path, "r")["medium"]["sound_speed"]
    map_dens = h5py.File(medium_path, "r")["medium"]["density"]
    
    vmin = np.min(mat_rfdata)
    vmax = np.max(mat_rfdata)
    assert vmax > vmin
    mat_normalized = np.copy(mat_rfdata)
    mat_normalized -= vmin
    mat_normalized /= (vmax-vmin)

    rfdata = RFdata(
        pos = arr_cart_pos,
        data = mat_normalized,
        src = np.array(param["source"]["point_map"])-1,
        rcv = np.arange(param["ringarray"]["num_points"]),
        dt = 1.0/param["sensor"]["freq"],
        c = np.mean(map_speed)
    )

    return param, arr_cart_pos, mask_points, map_speed, map_dens, rfdata

def draw_input(mask_points, map_speed, map_dens, save_path = None):
    
    points_pos = np.where(mask_points>0)
    
    fig = plt.figure(figsize=(12,9))
    
    ax = plt.subplot(121)
    image = ax.imshow(map_speed, cmap='gray')
    ax.axis("image")
    plt.scatter( points_pos[0], points_pos[1], s=3)
    plt.title("sound speed")

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)    
    plt.colorbar(image, cax = ax_cb)    

    ax = plt.subplot(122)
    image = ax.imshow(map_dens, cmap='gray')
    ax.axis("image")
    plt.scatter( points_pos[0], points_pos[1], s=3)
    plt.title("density")
    
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)    
    plt.colorbar(image, cax = ax_cb)    

    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    