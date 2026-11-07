"""Эквивалентная запись стимула в заранее нулевой буфер без операции +=."""
import numpy as np


def assign_zero_stimulus(torch, buffer, frames, indices):
    index = torch.as_tensor(indices, device=buffer.device)
    if len(np.unique(np.asarray(indices))) != np.asarray(indices).size:
        raise ValueError('Distinct receptor node indices required')
    buffer[:, :, index] = frames[:, :, None].expand(-1, -1, index.shape[0], -1)


def fast_init(self, hex0):
    torch=self.torch;net=self.net
    steps=int(self.fade_in_seconds/self.dt);batch=hex0.shape[0]
    net.stimulus.zero(batch,steps)
    frames=torch.linspace(0,1,steps,device=hex0.device)[None,:,None]*(hex0.reshape(batch,1,-1).repeat(1,steps,1)-.5)+.5
    assign_zero_stimulus(torch,net.stimulus.buffer,frames,net.stimulus.input_index)
    net.stimulus._nonzero=True
    with torch.no_grad():
        x=net.stimulus();net.clamp();params=net._param_api();state=net._initial_state(params,batch)
        for i in range(x.shape[1]):state=net._next_state(params,state,x[:,i],self.dt)
    return state
