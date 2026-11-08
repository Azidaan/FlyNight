"""Independent float32 arithmetic matching the documented native conversion path."""
import numpy as np


def native_cost(mean,box,confidence):
    pred=np.array(mean[:4],float);pred[:2]-=pred[2:]/2;pred[2:]+=pred[:2];pred=pred.astype(np.float32)
    raw=np.asarray(box,np.float32);center=(raw[:2]+raw[2:])/2;size=raw[2:]-raw[:2]
    det=np.r_[center,size].astype(np.float64);det[:2]-=det[2:]/2;det=det.astype(np.float32);det[2:]+=det[:2]
    # Keep one-element arrays so NumPy retains the native float32 promotion.
    a=pred.reshape(1,4);b=det.reshape(1,4);intersection=np.maximum(0,np.minimum(a[:,2],b[:,2])-np.maximum(a[:,0],b[:,0]))*np.maximum(0,np.minimum(a[:,3],b[:,3])-np.maximum(a[:,1],b[:,1]));area=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-intersection
    value=intersection/(area+1e-7);distance=1-value;similarity=1-distance;return float((1-similarity*np.asarray([confidence],np.float32))[0])
