"""Predeclared response-map observables; no annotations enter this API."""
import numpy as np

NAMES=('peak_z','second_separated_peak_z','peak_gap_z','normalized_entropy',
       'peak_probability_mass','winner_neighborhood_mass','spread_x','spread_y')


def response_features(score, geom):
    score=np.asarray(score,dtype=float)
    if score.shape!=(len(geom.points),) or not np.isfinite(score).all():
        raise ValueError('A finite score per eye column is required')
    z=(score-score.mean())/(score.std()+1e-9)
    winner=int(np.argmax(score)); local=np.unique(geom.neighbors[winner])
    outside=np.ones(len(score),bool); outside[local]=False
    second=float(z[outside].max()) if outside.any() else float(z[winner])
    exp=np.exp(z-z.max()); p=exp/exp.sum()
    mean=p@geom.points
    variance=p@((geom.points-mean)**2)
    span=np.maximum(np.ptp(geom.points,axis=0),1.)
    values=np.array([z[winner],second,z[winner]-second,
           -np.sum(p*np.log(np.maximum(p,1e-300)))/np.log(len(p)),
           p[winner],p[local].sum(),*(np.sqrt(variance)/span)],dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite map statistic')
    return values
