import numpy as np

class RFdata(object):
    
    def __init__(self, pos, data, src, rcv, dt, c):
        
        self.pos = pos        # sensor positions
        self.data = data      # rfdata matrix
        self.dt = dt          # sampling interval
        self.c = c            # sound velocity
        
        self.n_src, self.n_rcv, self.n_time = list(self.data.shape)
        self.n_pos, self.n_dim = list(self.pos.shape)
        
        assert self.n_src == len(src)
        assert self.n_rcv == len(rcv)
        assert self.n_src <= self.n_pos
        assert self.n_rcv <= self.n_pos
        
        self.mesh_rcv, self.mesh_src = np.meshgrid(rcv, src)
        
        self.pos_src = pos[self.mesh_src.flatten()].reshape(self.n_src, self.n_rcv, self.n_dim)
        self.pos_rcv = pos[self.mesh_rcv.flatten()].reshape(self.n_src, self.n_rcv, self.n_dim)
        
        self.mat_time = np.zeros_like(self.data, dtype=np.int16)
        for i in range(self.mat_time.shape[2]) : self.mat_time[:,:,i] = i
        
    def getArrivalTimeArray(self, target):
        
        dist_src = np.linalg.norm( self.pos_src - target, axis= 2)
        dist_rcv = np.linalg.norm( self.pos_rcv - target, axis = 2)
        dist = dist_src + dist_rcv
        return (dist/(self.c*self.dt)).astype(np.uint16)
    
    def getPointSubset(self, target, offset_array):
        
        ret = np.zeros((self.data.shape[0], self.data.shape[1],len(offset_array)))
        
        time_arrive = self.getArrivalTimeArray(target)
        mat_time_diff = self.mat_time - time_arrive[:,:,np.newaxis]
        
        for i, offset in enumerate(offset_array):
            mat_filter = np.ones_like(self.mat_time)
            mat_filter *= (mat_time_diff == offset)*1
            rf_filtered = self.data * mat_filter
            map_filtered =np.sum(rf_filtered, axis=2)
            ret[:,:,i] = map_filtered
        
        return ret