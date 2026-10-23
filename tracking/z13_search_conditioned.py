"""Tiny target-filter predictor, with a matched template-only weight control."""


def make_predictor(torch,activity=True,search_conditioned=True):
    nn=torch.nn
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.attn=nn.MultiheadAttention(16,2,dropout=0,batch_first=True)
            self.n1=nn.LayerNorm(16);self.n2=nn.LayerNorm(16);self.ff=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))
        def forward(self,z,padding,mask):
            h=self.attn(z,z,z,key_padding_mask=padding,attn_mask=mask,need_weights=False)[0]
            z=self.n1(z+h);return self.n2(z+self.ff(z))
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__();self.activity=activity;self.search_conditioned=search_conditioned
            self.embed=nn.Sequential(nn.LayerNorm(68),nn.Linear(68,16),nn.Tanh());self.geometry=nn.Linear(8,16)
            self.foreground=nn.Parameter(torch.zeros(1,1,16));self.search_tag=nn.Parameter(torch.zeros(1,1,16))
            self.blocks=nn.ModuleList([Block(),Block()]);self.decoder=nn.MultiheadAttention(16,2,dropout=0,batch_first=True)
            self.norm=nn.LayerNorm(16);self.weight=nn.Linear(16,16);self.bias=nn.Linear(16,1)
        def parts(self,x,s,g,valid):
            if not self.activity:x=torch.zeros_like(x);s=torch.zeros_like(s)
            seed=self.embed(s)[:,None,:]+self.foreground;query=self.embed(x)+self.geometry(g)+self.search_tag
            z=torch.cat([seed,query],1);padding=torch.cat([torch.zeros((len(x),1),dtype=torch.bool,device=x.device),~valid.bool()],1)
            mask=torch.zeros((z.shape[1],z.shape[1]),dtype=torch.bool,device=x.device)
            if not self.search_conditioned:mask[0,1:]=True
            for block in self.blocks:z=block(z,padding,mask)
            decoder_padding=padding.clone()
            if not self.search_conditioned:decoder_padding[:,1:]=True
            q=self.foreground.expand(len(x),1,-1)
            decoded=self.norm(q+self.decoder(q,z,z,key_padding_mask=decoder_padding,need_weights=False)[0])[:,0]
            return self.weight(decoded),self.bias(decoded),z[:,1:]
        def forward(self,x,s,g,valid):
            w,b,z=self.parts(x,s,g,valid);score=(z*w[:,None,:]).sum(-1)/4+b
            return score.masked_fill(~valid.bool(),-1e4)
    return Predictor()
