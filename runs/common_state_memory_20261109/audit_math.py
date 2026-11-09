"""New B/C sampling proof and a balanced-reader counterexample, no real scores."""
import study as s
from tracking.channel_balanced_correspondence import channel_balanced_similarity as sim
np=s.np
p=s.read(s.OUT/'protocol.json');s.verify(p)
v=np.array([1.,-1.,0.])/np.sqrt(2);w=np.array([1.,1.,-2.])/np.sqrt(6);epsilon=.1
a1=v;a2=(1-epsilon)*v+np.sqrt(1-(1-epsilon)**2)*w
early=np.column_stack([a1,v]);last=np.column_stack([a1,-v]);own=last.copy();other=np.column_stack([a2,v])
values=dict(B_own=sim(last,own),B_foreign=sim(last,other),C_own=sim(early,own),C_foreign=sim(early,other))
assert abs(values['B_own']-1)<1e-12 and abs(values['B_foreign']+epsilon/2)<1e-12
assert abs(values['C_own'])<1e-12 and abs(values['C_foreign']-(1-epsilon/2))<1e-12
scaled=early.copy();scaled[:,1]*=1000;assert abs(sim(scaled,own)-values['C_own'])<1e-12
s.write('balanced_math_audit.json',dict(passed=True,source_sha256=s.sha(__file__),values=values,per_channel_positive_scale_invariance=True,interpretation='The Gemini global-energy gamma inequality does not transfer to equal-channel normalization. A two-channel zero-mean spatial counterexample still defeats unconditional early-is-better: one changing channel can reverse a weak appearance difference, independent of its positive amplitude. This is an existence proof, not a measured FlyVis mechanism or a motion-reversal-latency experiment.',no_retinal_model_or_real_pair_score_calls=True))
print('Balanced mathematical audit passed',values,flush=True)
