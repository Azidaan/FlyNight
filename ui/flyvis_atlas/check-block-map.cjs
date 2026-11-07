"use strict";
// The camera test uses deterministic animation frames; no browser or FlyVis weights are required.
const assert=require("node:assert/strict");
global.window=global;global.matchMedia=()=>({matches:false,addEventListener(){}});
for(const name of ["data/connectome","block-map","block-camera"])require("./dist/"+name+".js");
const g=FLY_CONNECTOME,m=FlyBlockMap.model(g);
assert.equal(m.groups.length,8);assert.equal(new Set(m.groups.flatMap(k=>k.types)).size,65);
assert.deepEqual([...m.pairs.values()].flat().sort(),g.edges.slice().sort());
assert.deepEqual(m.routeList.map(p=>p.types),[["R1","L1","Mi1","T4a"],["R1","L2","Tm1","T5a"]]);
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
console.log("8 groups, 65 unique types, 605 source pairs; two verified routes. Camera: intermediate frames, cursor anchor, pan, reversed zoom and reset verified.");
