"""Та же запись рецепторов через явное копирование по последней оси."""
import numpy as np


def indexed_stimulus(torch, stimulus, indices, n_nodes):
    index=torch.as_tensor(indices,device=stimulus.device)
    buffer=torch.zeros((stimulus.shape[0],stimulus.shape[1],n_nodes),device=stimulus.device,dtype=stimulus.dtype)
    expanded=stimulus.expand(-1,-1,index.shape[0],-1)
    buffer.index_copy_(2,index.reshape(-1),expanded.reshape(stimulus.shape[0],stimulus.shape[1],-1))
    return buffer


def index_copy_step(self,state,last_hex,cur_hex):
    torch=self.torch
    alpha=torch.arange(1,self.steps+1,device=self.dev,dtype=torch.float32).view(1,self.steps,1,1)/self.steps
    stimulus=last_hex.reshape(-1,1,1,last_hex.shape[-1])*(1-alpha)+cur_hex.reshape(-1,1,1,cur_hex.shape[-1])*alpha
    st=self.net.stimulus
    buffer=indexed_stimulus(torch,stimulus,np.asarray(st.input_index),st.n_nodes)
    states=self.net.forward(buffer,self.dt,state,as_states=True)
    activity=states[-1].nodes.activity
    return states[-1],activity[:,self.index]
