"""Read-only native state view for common appearance-signature extraction."""
import numpy as np


def xywh_state(track):
    m=np.asarray(track.mean).copy();name=type(track.kalman_filter).__name__
    if name=='KalmanFilterXYWH':return m
    if name!='KalmanFilterXYAH':raise TypeError('Unknown native Kalman representation: '+name)
    a,h,va,vh=m[2],m[3],m[6],m[7]
    m[2]=a*h;m[6]=(a+va)*(h+vh)-a*h
    return m


class TrackView:
    def __init__(self,track):self._track=track
    @property
    def mean(self):return xywh_state(self._track)
    def __getattr__(self,key):return getattr(self._track,key)


class TrackerView:
    def __init__(self,tracker):self.native=tracker
    @property
    def tracked_stracks(self):return [TrackView(t) for t in self.native.tracked_stracks]
    def update(self,results,img=None):return self.native.update(results,img=img)
    def __getattr__(self,key):return getattr(self.native,key)
