"""Separate every search query from intervening search-state updates.

The observed four-frame exemplar is kept intact. Each query executes the same
frozen feature producer on that exemplar followed by that query, so no neural
state object is copied or shared between alternative histories.
"""
def anchor_features(torch,arm,backbone,hexes,neighbors,needed,feature_fn,prefix=None):
    if hexes.ndim!=3 or hexes.shape[1]<4:raise ValueError('Four observed frames required')
    needed=set(needed)
    if any(t<3 or t>=hexes.shape[1] for t in needed):raise ValueError('Invalid requested frame')
    out={}
    if 3 in needed:out[3]=prefix[3] if prefix is not None else feature_fn(torch,arm,backbone,hexes[:,:4],neighbors,{3})[3]
    for t in sorted(needed-{3}):
        # These exact causal prefixes are shared by both conditions. Reusing
        # their arrays avoids treating GPU replay jitter as an intervention.
        if t==4 and prefix is not None:
            out[t]=prefix[4];continue
        sequence=torch.cat((hexes[:,:4],hexes[:,t:t+1]),dim=1)
        out[t]=feature_fn(torch,arm,backbone,sequence,neighbors,{4})[4]
    return out
