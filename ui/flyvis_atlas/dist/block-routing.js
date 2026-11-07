(() => {
  "use strict";
  const EPS=1e-7,CLEAR=4,BEND=9;
  const same=(a,b)=>Math.abs(a[0]-b[0])<EPS&&Math.abs(a[1]-b[1])<EPS;
  const contains=(r,p)=>p[0]>r.left+EPS&&p[0]<r.right-EPS&&p[1]>r.top+EPS&&p[1]<r.bottom-EPS;
  const crosses=(r,a,b)=>{
    if(Math.abs(a[0]-b[0])<EPS)return a[0]>r.left+EPS&&a[0]<r.right-EPS&&Math.max(a[1],b[1])>r.top+EPS&&Math.min(a[1],b[1])<r.bottom-EPS;
    if(Math.abs(a[1]-b[1])<EPS)return a[1]>r.top+EPS&&a[1]<r.bottom-EPS&&Math.max(a[0],b[0])>r.left+EPS&&Math.min(a[0],b[0])<r.right-EPS;
    return true;
  };
  class Heap{
    constructor(){this.q=[]}
    push(item){const q=this.q;let i=q.length;q.push(item);while(i){const p=(i-1)>>1;if(q[p][0]<=item[0])break;q[i]=q[p];i=p}q[i]=item}
    pop(){const q=this.q,first=q[0],last=q.pop();if(q.length){let i=0;while(2*i+1<q.length){let c=2*i+1;if(c+1<q.length&&q[c+1][0]<q[c][0])c++;if(q[c][0]>=last[0])break;q[i]=q[c];i=c}q[i]=last}return first}
    get length(){return this.q.length}
  }
  function simplify(points){
    const clean=[];for(const p of points){
      if(clean.length&&same(clean[clean.length-1],p))continue;
      while(clean.length>1){const a=clean[clean.length-2],b=clean[clean.length-1];
        if(Math.abs((b[0]-a[0])*(p[1]-b[1])-(b[1]-a[1])*(p[0]-b[0]))<EPS&&
          (b[0]-a[0])*(p[0]-b[0])+(b[1]-a[1])*(p[1]-b[1])>=0)clean.pop();else break;
      }clean.push([p[0],p[1]]);
    }return clean;
  }
  function route(a,b,obstacles,bounds){
    const boxes=obstacles||[a,b];
    const valid=r=>r&&[r.left,r.top,r.right,r.bottom].every(Number.isFinite)&&r.right>r.left&&r.bottom>r.top;
    if(!valid(a)||!valid(b)||!boxes.every(valid))return [];
    const lim=bounds||{left:Math.min(...boxes.map(r=>r.left))-24,top:Math.min(...boxes.map(r=>r.top))-24,right:Math.max(...boxes.map(r=>r.right))+24,bottom:Math.max(...boxes.map(r=>r.bottom))+24};
    if(!valid(lim))return [];
    const padded=boxes.map(r=>({...r,left:r.left-CLEAR,top:r.top-CLEAR,right:r.right+CLEAR,bottom:r.bottom+CLEAR}));
    const within=p=>p[0]>=lim.left-EPS&&p[0]<=lim.right+EPS&&p[1]>=lim.top-EPS&&p[1]<=lim.bottom+EPS;
    function ports(r){
      const x=(r.left+r.right)/2,y=(r.top+r.bottom)/2;
      return [
        {at:[r.left,y],out:[r.left-CLEAR,y],dir:0},
        {at:[r.right,y],out:[r.right+CLEAR,y],dir:0},
        {at:[x,r.top],out:[x,r.top-CLEAR],dir:1},
        {at:[x,r.bottom],out:[x,r.bottom+CLEAR],dir:1}
      ].filter(p=>within(p.at)&&within(p.out)&&!padded.some(o=>o.id!==r.id&&(contains(o,p.out)||crosses(o,p.at,p.out))));
    }
    const starts=ports(a),ends=ports(b);if(!starts.length||!ends.length)return [];
    const xs=[lim.left,lim.right],ys=[lim.top,lim.bottom];
    for(const r of padded){xs.push(r.left,r.right);ys.push(r.top,r.bottom)}
    for(const p of [...starts,...ends]){xs.push(p.out[0]);ys.push(p.out[1])}
    const uniq=(v,lo,hi)=>[...new Set(v.filter(x=>x>=lo-EPS&&x<=hi+EPS))].sort((x,y)=>x-y);
    const X=uniq(xs,lim.left,lim.right),Y=uniq(ys,lim.top,lim.bottom),nx=X.length,ny=Y.length,n=nx*ny;
    const pos=i=>[X[i%nx],Y[Math.floor(i/nx)]];
    const id=p=>Y.indexOf(p[1])*nx+X.indexOf(p[0]);
    const good=new Uint8Array(n);for(let i=0;i<n;i++)good[i]=!padded.some(r=>contains(r,pos(i)));
    const links=Array.from({length:n},()=>[]);
    for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
      const i=y*nx+x;if(!good[i])continue;
      for(const [j,d] of [[x+1<nx?i+1:-1,0],[y+1<ny?i+nx:-1,1]]){
        if(j<0||!good[j])continue;const u=pos(i),v=pos(j);
        if(!padded.some(r=>crosses(r,u,v))){const cost=Math.abs(u[0]-v[0])+Math.abs(u[1]-v[1]);links[i].push([j,d,cost]);links[j].push([i,d,cost])}
      }
    }
    const self=a.id===b.id;
    let answer=[],best=Infinity;
    function search(sources){
      const dist=new Float64Array(n*2);dist.fill(Infinity);const prev=new Int32Array(n*2);prev.fill(-1);
      const origin=new Int16Array(n*2);origin.fill(-1);const heap=new Heap();
      sources.forEach(si=>{const p=starts[si],node=id(p.out);if(node<0||!good[node])return;const st=node*2+p.dir;dist[st]=CLEAR;origin[st]=si;heap.push([CLEAR,st])});
      while(heap.length){
        const [cost,state]=heap.pop();if(cost>dist[state]+EPS)continue;
        const node=state>>1,dir=state%2;
        for(const [j,d,len] of links[node]){
          const next=j*2+d,score=cost+len+(dir===d?0:BEND);
          if(score+EPS<dist[next]){dist[next]=score;prev[next]=state;origin[next]=origin[state];heap.push([score,next])}
        }
      }
      for(const end of ends){
        const node=id(end.out);
        for(let d=0;d<2;d++){
          const state=node*2+d,si=origin[state];if(si<0||(self&&same(starts[si].at,end.at)))continue;
          const score=dist[state]+CLEAR+(d===end.dir?0:BEND);if(score>=best)continue;
          const pts=[];let s=state;while(s>=0){pts.push(pos(s>>1));s=prev[s]}
          pts.reverse();const candidate=simplify([starts[si].at,...pts,end.at]);
          if(candidate.length<2||candidate.slice(1).some((p,i)=>boxes.some(r=>crosses(r,candidate[i],p))))continue;
          best=score;answer=candidate;
        }
      }
    }
    if(self)starts.forEach((_,i)=>search([i]));else search(starts.map((_,i)=>i));
    return answer;
  }
  function rounded(points,r=5){
    const p=simplify(points||[]);if(p.length<2)return "";
    const f=n=>String(Math.round(n*1000)/1000),xy=v=>v.map(f).join(" ");
    let d="M"+xy(p[0]);for(let i=1;i<p.length-1;i++){
      const a=p[i-1],b=p[i],c=p[i+1],l1=Math.hypot(a[0]-b[0],a[1]-b[1]),l2=Math.hypot(c[0]-b[0],c[1]-b[1]);
      const radius=Math.min(Math.max(0,r),l1/2,l2/2);
      const enter=[b[0]+(a[0]-b[0])*radius/l1,b[1]+(a[1]-b[1])*radius/l1],exit=[b[0]+(c[0]-b[0])*radius/l2,b[1]+(c[1]-b[1])*radius/l2];
      d+=" L"+xy(enter)+" Q"+xy(b)+" "+xy(exit);
    }return d+" L"+xy(p[p.length-1]);
  }
  window.FlyBlockRouting={route,rounded};
})();
