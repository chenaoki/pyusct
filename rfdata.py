import numpy as np
import os, h5py, scipy.io, scipy.signal
import util
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class RFdata(object):
    
    def __init__(self, result_path):
        """
        * argument
            - result_path : by USCTSim
        """
        assert os.path.exists(result_path)
        
        # simulation inputs
        self.param        = util.load_matlab_struct(result_path, 'param')
        medium            = h5py.File(os.path.join(result_path, 'medium.mat'), "r")["medium"]
        self.medium_c     = np.array(medium["sound_speed"])
        self.medium_d     = np.array(medium["density"])
        
        self.kgrid        = util.load_matlab_struct(result_path, 'str_kgrid')
        self.dt           = self.kgrid["t_array"][1] - self.kgrid["t_array"][0]
        
        # sensors
        self.sensor_pos    = util.load_matlab_struct(result_path, 'sensor')["mask"].T
        
        path = os.path.join(result_path, 'mask_points.mat')
        self.sensor_map    = scipy.io.loadmat(path)["mask_points"]
        
        # sources
        path = os.path.join(result_path, "source_wave.mat")
        self.source_wave   = scipy.io.loadmat(path)["source_wave"][0,:]
        
        self.source_pos = self.sensor_pos[self.param["source"]["point_map"]-1,:]
        
        # simulation outputs
        path = os.path.join(result_path, "rfdata.mat")
        self.rawdata       = np.array(h5py.File(path, "r")["rfdata"])
        
        # hilbert transform
        comp_ = scipy.signal.hilbert(self.rawdata - self.rawdata.mean(), axis=2)
        self.phase = np.angle(comp_)
        self.amp = abs(comp_)
        
        # size info 
        self.T, self.R, self.L = list(self.rawdata.shape)
        self.n_dim = self.sensor_pos.shape[0]
        
        # T-R mesh 
        self.mesh_n_rcv, self.mesh_n_src = np.meshgrid(np.arange(self.R), self.param["source"]["point_map"]-1)
        self.mesh_pos_rcv = self.sensor_pos[self.mesh_n_rcv,:]
        self.mesh_pos_src = self.sensor_pos[self.mesh_n_src,:]
        
        return
        
    def getPointSubset(self, target, offset_arr=[0]):

        # travel distance
        map_dist_src = np.linalg.norm( self.mesh_pos_src - target, axis = 2)
        map_dist_rcv = np.linalg.norm( self.mesh_pos_rcv - target, axis = 2)
        map_dist = map_dist_src + map_dist_rcv

        # sampling index of arrival time 
        c = np.median(self.medium_c)
        map_time_pos = (map_dist/(c*self.dt)).astype(np.uint16)
        
        def pairwise_extraction(RF, map_time_pos, offset_arr):
            D = np.zeros( (self.T, self.R, len(offset_arr)), dtype=RF.dtype)
            for i in range(self.T):
                for j in range(self.R):
                    pos = map_time_pos[i,j]
                    D[i,j,:] = RF[i,j,pos+offset_arr]
            return D
        
        subset = pairwise_extraction(self.amp, map_time_pos, offset_arr)        
        
        return map_time_pos, subset
    

    def draw_input(self):
    
        points = np.where(self.sensor_map>0)

        fig = plt.figure(figsize=(12,9))

        ax = plt.subplot(121)
        image = ax.imshow(self.medium_c, cmap='gray')
        ax.axis("image")
        plt.scatter( points[0], points[1], s=3, c='blue')
        plt.title("sound speed")

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="2%", pad=0.05)
        fig.add_axes(ax_cb)    
        plt.colorbar(image, cax = ax_cb)    

        ax = plt.subplot(122)
        image = ax.imshow(self.medium_d, cmap='gray')
        ax.axis("image")
        plt.scatter( points[0], points[1], s=3, c='blue')
        plt.title("density")

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="2%", pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(image, cax = ax_cb)    

        plt.show()
        