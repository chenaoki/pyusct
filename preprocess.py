
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
import scipy.io

from . import rfdata

def preprocess(result_path, list_pos, offset = 0):
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
    assert len(list_pos) > 0

    param_path = os.path.join(result_path, 'param.json')
    medium_path = os.path.join(result_path, 'medium.mat')
    sensor_path = os.path.join(result_path, 'sensor.mat')
    rfdata_path = os.path.join(result_path, 'rfdata.mat')

    with open(param_path, 'r') as f: param = json.load(f)
    arr_pos = scipy.io.loadmat(sensor_path)["sensor"]["mask"][0][0].T
    mat_rfdata = h5py.File(rfdata_path, "r")["rfdata"]
    map_speed = h5py.File(medium_path, "r")["medium"]["sound_speed"]
    map_dens = h5py.File(medium_path, "r")["medium"]["density"]
    
    vmin = np.min(mat_rfdata)
    vmax = np.max(mat_rfdata)
    assert vmax > vmin
    mat_normalized = np.copy(mat_rfdata)
    mat_normalized -= vmin
    mat_normalized /= (vmax-vmin)

    rf = rfdata.RFdata(
        pos = arr_pos,
        data = mat_normalized,
        src = np.array(param["source"]["point_map"])-1,
        rcv = np.arange(param["ringarray"]["num_points"]),
        dt = 1.0/param["sensor"]["freq"],
        c = np.mean(map_speed)
    )

    preprocessed = np.zeros( (len(list_pos), rf.data.shape[0], rf.data.shape[1], 1+offset*2 ) )
    mat_time = np.zeros_like(rf.data, dtype=np.int16)
    for i in range(mat_time.shape[2]) : mat_time[:,:,i] = i

    for p, pos in enumerate(list_pos):

        dist = rf.relevant_submat(np.array(pos)).astype(np.int16)
        mat_time_diff = mat_time - dist[:,:,np.newaxis]

        for o, offset in enumerate(np.arange(-offset, offset+1, 1)):

            mat_filter = np.ones_like(mat_time)
            mat_filter *= (mat_time_diff == offset)*1
            rf_filtered = rf.data * mat_filter
            map_filtered = np.sum(rf_filtered, axis=2)

            preprocessed[p, :, :, o] = map_filtered

    return map_speed, map_dens, preprocessed
