"""Generic label-free integer translation using robust 5x5 Census matching."""
import numpy as np

OFFSETS=tuple((dy,dx) for dy in range(-2,3) for dx in range(-2,3) if dy or dx)
POPCOUNT=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)

def census_field(gray):
    a=np.asarray(gray)
    if a.ndim!=2 or a.dtype!=np.uint8:
        raise ValueError('uint8 gray image required')
    field=np.zeros(a.shape,dtype=np.uint32)
    center=a[2:-2,2:-2]
    for bit,(dy,dx) in enumerate(OFFSETS):
        field[2:-2,2:-2]|=(a[2+dy:a.shape[0]-2+dy,2+dx:a.shape[1]-2+dx]<center).astype(np.uint32)<<bit
    return field

def register_translation(reference_gray,current_gray,reference_box,reference_crop,current_box,current_crop,source_world,mesh,eye_size):
    """No labels, learned scores, object IDs, or hand-selected displacement.

    Search ALL integer translations which keep source pixel template with its
    5x5 neighborhoods inside the current crop AND every supplied source map
    point inside the current eye hull. Minimize median Hamming, mean Hamming,
    distance to detector center translation, dx, dy in that exact order.
    """
    rb=np.asarray(reference_box,float);rc=np.asarray(reference_crop,float)
    cb=np.asarray(current_box,float);cc=np.asarray(current_crop,float)
    source=np.asarray(source_world,float)
    xmin=max(int(np.ceil(rb[0])),int(rc[0])+2,2)
    xmax=min(int(np.floor(rb[2])),int(rc[2])-3,reference_gray.shape[1]-3)
    ymin=max(int(np.ceil(rb[1])),int(rc[1])+2,2)
    ymax=min(int(np.floor(rb[3])),int(rc[3])-3,reference_gray.shape[0]-3)
    if xmin>xmax or ymin>ymax:
        return dict(ready=False,reason='empty_source_pixel_template')
    xx,yy=np.meshgrid(np.arange(xmin,xmax+1),np.arange(ymin,ymax+1))
    pixels=np.c_[xx.ravel(),yy.ravel()]
    dxlo=max(int(cc[0])+2,2)-xmin
    dxhi=min(int(cc[2])-3,current_gray.shape[1]-3)-xmax
    dylo=max(int(cc[1])+2,2)-ymin
    dyhi=min(int(cc[3])-3,current_gray.shape[0]-3)-ymax
    candidates=[]
    for dx in range(dxlo,dxhi+1):
        for dy in range(dylo,dyhi+1):
            query=(source+[dx,dy]-cc[:2])*float(eye_size)/(cc[2:]-cc[:2])
            if (mesh.find_simplex(query)>=0).all():
                candidates.append((dx,dy))
    if not candidates:
        return dict(ready=False,reason='no_full_support_translation',pixel_bounds=[xmin,ymin,xmax,ymax],integer_rectangle=[dxlo,dylo,dxhi,dyhi])
    f0=census_field(reference_gray);f1=census_field(current_gray)
    a=f0[pixels[:,1],pixels[:,0]];costs=[]
    prior=(cb[:2]+cb[2:]-rb[:2]-rb[2:])/2
    for dx,dy in candidates:
        xor=a^f1[pixels[:,1]+dy,pixels[:,0]+dx]
        distance=POPCOUNT[xor&255]+POPCOUNT[(xor>>8)&255]+POPCOUNT[(xor>>16)&255]
        costs.append((float(np.median(distance)),int(distance.sum()),float(np.sum((np.array([dx,dy])-prior)**2)),dx,dy))
    winner=min(range(len(costs)),key=lambda i:costs[i])
    return dict(ready=True,translation=list(candidates[winner]),candidate_translations=candidates,cost_tuples=costs,winner=winner,source_pixels=pixels,pixel_bounds=[xmin,ymin,xmax,ymax],integer_rectangle=[dxlo,dylo,dxhi,dyhi],detector_center_prior=prior,objective='lexicographic median24bitHamming,sum24bitHamming,distance2_to_detectorcenter,dx,dy')
