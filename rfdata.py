import numpy as np

class RFdata(object):
    
    def __init__(self, pos, mat, src, rcv, dt, c):
        
        self.pos = pos   # tdr geometry
        self.data = mat  # rfdata matrix
        self.dt = dt      # sampling interval
        self.c = c        # sound velocity
        
        assert mat.shape[1] == len(src)
        assert mat.shape[2] == len(rcv)
        
        self.mesh_src, self.mesh_rcv = np.meshgrid(src, rcv)
        self.pos_src = pos[self.mesh_src.flatten()].reshape(
            np.array([self.mesh_src.shape[0], self.mesh_src.shape[1], pos.shape[1]])
        )
        self.pos_rcv = pos[self.mesh_rcv.flatten()].reshape(
            np.array([self.mesh_rcv.shape[0], self.mesh_rcv.shape[1],  pos.shape[1]])
        )
        
    def relevant_submat(self, target):
        dist_src = np.linalg.norm( self.pos_src - target, axis= 2)
        dist_rcv = np.linalg.norm( self.pos_rcv - target, axis = 2)
        dist = dist_src + dist_rcv
        return (dist/(self.c*self.dt)).astype(np.uint16)