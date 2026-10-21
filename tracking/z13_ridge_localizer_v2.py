"""Service-only R1 singleton-channel adapter; C/D behavior is delegated unchanged."""
from tracking.z13_ridge_localizer import CenterRidgeLocalizer as OriginalCenter


class RecoveredCenterRidgeLocalizer(OriginalCenter):
    def start_sequence(self,hexes,masks):
        if self.arm=='R1':
            if hexes.ndim not in (3,4) or (hexes.ndim==4 and hexes.shape[2]!=1):raise ValueError('Expected B,T,N or B,T,1,N retinal input')
            hexes=hexes.reshape(hexes.shape[0],hexes.shape[1],-1)
        return super().start_sequence(hexes,masks)

    def advance(self,eyes,current):
        if self.arm=='R1':
            if current.ndim not in (2,3) or (current.ndim==3 and current.shape[1]!=1):raise ValueError('Expected B,N or B,1,N retinal frame')
            current=current.reshape(current.shape[0],-1)
        return super().advance(eyes,current)
