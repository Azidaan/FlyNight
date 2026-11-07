"use strict";
// The camera test uses deterministic animation frames; no browser or FlyVis weights are required.
const assert=require("node:assert/strict");
global.window=global;global.matchMedia=()=>({matches:false,addEventListener(){}});
for(const name of ["data/connectome","block-map","block-camera","block-routing"])require("./dist/"+name+".js");
const g=FLY_CONNECTOME,m=FlyBlockMap.model(g);
assert.equal(m.groups.length,8);assert.equal(new Set(m.groups.flatMap(k=>k.types)).size,65);
assert.deepEqual([...m.pairs.values()].flat().sort(),g.edges.slice().sort());
assert.equal(m.routeList.length,4);
for(const route of m.routeList)route.types.slice(1).forEach((t,i)=>assert.ok(g.edges.some(e=>e[0]===route.types[i]&&e[1]===t)));
assert.ok(m.routeList.some(p=>p.types.includes("T1")&&new Set(p.types).size<p.types.length),"a route includes a lower group and a feedback loop");
for(const t of g.types)for(const dir of ["in","out"]){
 const bundles=FlyBlockMap.bundleEdges(g,t,dir);
 const raw=g.edges.filter(e=>e[dir==="out"?0:1]===t);
 assert.deepEqual(bundles.flatMap(b=>b.pairs).sort(),raw.slice().sort());
 assert.equal(new Set(bundles.map(b=>b.a+"/"+b.b)).size,bundles.length,"one arrow per directed group pair");
}
function crosses(r,a,b){
 return Math.abs(a[0]-b[0])<1e-6?
  a[0]>r.left+1e-6&&a[0]<r.right-1e-6&&Math.max(a[1],b[1])>r.top+1e-6&&Math.min(a[1],b[1])<r.bottom-1e-6:
  a[1]>r.top+1e-6&&a[1]<r.bottom-1e-6&&Math.max(a[0],b[0])>r.left+1e-6&&Math.min(a[0],b[0])<r.right-1e-6;
}
let routed=0;
for(const small of [false,true])for(const detail of [false,true]){
 const {boxes}=FlyBlockMap.diagramLayout(m.groups,small,detail),size=small?{right:380,bottom:630}:{right:785,bottom:460};
 for(const [pair] of m.pairs){
  const [a,b]=pair.split("/"),path=FlyBlockRouting.route(boxes.find(v=>v.id===a),boxes.find(v=>v.id===b),boxes,{left:3,top:3,...size});
  assert.ok(path.length>1,"path exists: "+[small,detail,pair]);
  for(let i=1;i<path.length;i++){
   const u=path[i-1],v=path[i];assert.ok(u[0]===v[0]||u[1]===v[1]);
   assert.ok(!boxes.some(r=>crosses(r,u,v)),"never enter a card or a title: "+pair);
  }routed++;
 }
}
const requests=new Map(),events=new Map();let seq=0,tick=0,state;const frames=[];
global.requestAnimationFrame=cb=>{requests.set(++seq,cb);return seq};
global.cancelAnimationFrame=id=>requests.delete(id);
const svg={viewBox:{baseVal:{width:800,height:480}},addEventListener:(type,cb)=>events.set(type,cb),
 getScreenCTM:()=>({inverse(){return this}}),createSVGPoint:()=>({matrixTransform(){return this}}),classList:{add(){},remove(){}},setPointerCapture(){},hasPointerCapture:()=>false};
const cam=FlyAtlasCamera.create(svg,null,p=>{state={...p};frames.push(state)});
function next(){const f=requests.entries().next().value;if(!f)return false;requests.delete(f[0]);f[1](++tick*16);return true}
function finish(){let i=0;while(next())assert.ok(++i<300,"camera settles")}
const wheel=dy=>events.get("wheel")({clientX:330,clientY:225,deltaMode:0,deltaY:dy,preventDefault(){}});
wheel(-350);next();assert.ok(state.z>1&&state.z<1.78);finish();
assert.ok(state.z>2.2&&frames.length>8);
for(const {x,y,z} of frames){assert.ok(Math.abs((330-x)/z-330)<1e-6);assert.ok(Math.abs((225-y)/z-225)<1e-6)}
const z=state.z,x=state.x;
events.get("pointerdown")({button:0,pointerId:1,clientX:330,clientY:225});
events.get("pointermove")({pointerId:1,clientX:380,clientY:225});finish();
assert.equal(state.z,z);assert.equal(state.x,x+50);
events.get("pointerup")({pointerId:1});
wheel(400);finish();assert.deepEqual(state,{z:1,x:0,y:0});
cam.go(6,400,240);next();cam.go(1);finish();assert.deepEqual(state,{z:1,x:0,y:0});cam.destroy();
console.log("8 groups, 65 types, 605 pairs preserved; four real routes, feedback and bundling verified. "+routed+" obstruction-free paths on desktop/mobile. Camera: smooth frames, cursor anchor, pan, reversed zoom and reset verified.");
const fs=require("node:fs");
if(fs.existsSync(__dirname+"/dist/data/episodes.js")){
 require("./dist/data/episodes.js");require("./dist/frame-path.js");
 const fp=FlyFramePath,e=FLY_EPISODES.episodes.find(x=>x.frame===212),f=e.frames[e.finalIndex],allNames=FLY_EPISODES.channels;
 assert.equal(fp.available,true);assert.equal(fp.stages.length,7);assert.equal(fp.info.correct,1);
 for(const policy of ["all","exclusive"]){
  fp.setPolicy(policy);const expected=fp.info.scores[policy==="all"?"whole":"exclusive"];
  e.masks.forEach((mask,i)=>{
   const own=mask.filter(j=>policy==="all"||!e.masks[1-i].includes(j));
   const actual=allNames.reduce((s,t,c)=>!/^T[45][a-d]$/.test(t)?s:s+e.weights[c]*own.reduce((v,j)=>v+f.maps[c][j],0)/own.length,0);
   assert.ok(Math.abs(actual-expected[i])<1e-12,"displayed scores follow the archived maps and target weights");
  });
  for(let i=0;i<7;i++)assert.ok(!/undefined|NaN/.test(fp.panel(i)+fp.graphic(i)));
 }
 assert.ok(fp.graphic(6).includes("Выбор: рамка А"));fp.setPolicy("all");
 assert.ok(fp.graphic(6).includes("Выбор: рамка Б"));
 console.log("Frame 212: seven stages, both masks, all scores independently recomputed within 1e-12; archived winner changes B to A.");
}else{
 require("./dist/frame-path.js");assert.equal(FlyFramePath.available,false);
 console.log("Local episode images/maps absent: fallback verified; frame-specific check skipped.");
}
