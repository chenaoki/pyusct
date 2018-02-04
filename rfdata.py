import numpy as np

class RFdata(object):
    
    def __init__(self, pos, data, src, rcv, dt, c):
        """
        *argument
            - src : index list of source transducer
            - rcv : index list of sensor transducer
        *return
        """
        
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
        assert self.data.shape[:2] == (self.n_src, self.n_rcv)
        
        mesh_rcv, mesh_src = np.meshgrid(rcv, src) # mesh shape:(n_src, n_rcv)
        
        self.pos_rcv = pos[mesh_rcv.flatten()].flatten().reshape(self.n_src, self.n_rcv, self.n_dim)
        self.pos_src = pos[mesh_src.flatten()].flatten().reshape(self.n_src, self.n_rcv, self.n_dim)
        
        return
        
    def getPointSubset(self, target, offset_min=0, offset_max=1, offset_int=1):
        
        s_ = self.data.shape
        len_offsets = len(np.arange(offset_min, offset_max, offset_int))

        # travel distance
        map_dist_src = np.linalg.norm( self.pos_src - target, axis = 2)
        map_dist_rcv = np.linalg.norm( self.pos_rcv - target, axis = 2)
        map_dist = map_dist_src + map_dist_rcv

        # sampling index of arrival time (with offset)
        map_time_pos = (map_dist/(self.c*self.dt)).astype(np.uint16)
        arr_time_pos = map_time_pos.flatten()

        data_ = self.data.flatten().reshape(len(arr_time_pos), s_[2])
        mat_out = np.zeros( (len(arr_time_pos), len_offsets),  dtype = np.float64)

        for i, t in enumerate(arr_time_pos): 
            try:
                extracted = data_[i, t+offset_min:t+offset_max:offset_int]
                mat_out[i, :len(extracted)] = extracted
            except:
                continue

        submat = mat_out.reshape((s_[0], s_[1], len_offsets))
        
        return map_time_pos, submat