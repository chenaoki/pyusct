import numpy as np
import os, h5py, scipy.io, scipy.signal, scipy.ndimage
import util
import matplotlib.pyplot as plt
import cv2

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
        self.medium_imp   = self.medium_c * self.medium_d
        
        #img               = self.medium_imp
        #img_norm          = ((img - img.min()) * 255 / (img.max() - img.min()) ).astype(np.uint8)
        img_lap           = np.abs(cv2.Laplacian(self.medium_imp,cv2.CV_64F))
        self.medium_sct   = (img_lap > 0.0 )*1.0
        
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
        
        # hilbert transformation
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
        
        # T-R mask
        self.TRmask = None
        
        return
        
    def getPointSubset(self, ngrid, offset_arr=[0], flg_mask = True):

        # travel distance
        pos = self.ngrid2pos(ngrid)
        map_dist_src = np.linalg.norm( self.mesh_pos_src - pos, axis = 2)
        map_dist_rcv = np.linalg.norm( self.mesh_pos_rcv - pos, axis = 2)
        map_dist = map_dist_src + map_dist_rcv

        # sampling index of arrival time 
        c = np.median(self.medium_c)
        map_time_pos = (map_dist/(c*self.dt)).astype(np.uint16)
        
        def pairwise_extraction(RF, map_time_pos, offset_arr):
            D = np.zeros( (self.T, self.R, len(offset_arr)), dtype=RF.dtype)
            for i in range(self.T):
                for j in range(self.R):
                    pos = map_time_pos[i,j]
                    ts = RF[i,j,:]
                    ts = np.concatenate((ts, np.zeros_like(ts)))
                    D[i,j,:] = ts[pos+offset_arr]
            return D       
        
        if flg_mask and self.TRmask is not None:
            RF = self.amp * self.TRmask[:,:,np.newaxis]
        else:
            RF = self.amp
        
        subset = pairwise_extraction(RF, map_time_pos, offset_arr)
        
        return map_time_pos, subset
    
    def setTRmask(self, maskFunc):
        self.TRmask = maskFunc(self)
        return self.TRmask
    
    def syntheticAperture(self, c = 1):
        
        mesh_grid = np.array(np.meshgrid( np.arange(self.kgrid["Ny"]//c), np.arange(self.kgrid["Nx"]//c) ))
        ngrids = mesh_grid.reshape(2, mesh_grid.size//2).T
        
        def gridwise_summation(ngrid):
            _, subset = self.getPointSubset(ngrid*c)
            return np.sum(subset)
        
        sa = np.array([gridwise_summation(ngrid) for ngrid in ngrids])
        sa = np.log(sa.reshape( self.kgrid["Nx"]//c, self.kgrid["Ny"]//c ).T)
        return sa
        
    
    def ngrid2pos(self, ngrid):
        return np.array([ 
                self.kgrid["x"].T[ngrid[0], ngrid[1]], 
                self.kgrid["y"].T[ngrid[0], ngrid[1]]])
    
    def pos2ngrid(self, pos):
        pos_array = np.array([self.kgrid["x"].T, self.kgrid["y"].T] )
        dist = np.linalg.norm(pos_array - pos[:, np.newaxis, np.newaxis], axis=0)
        ngrid = np.unravel_index( np.argmin(dist), dist.shape)
        return ngrid    

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
        