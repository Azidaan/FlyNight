(() => {
  "use strict";
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v)),one=()=>({z:1,x:0,y:0});
  // x/y translate model coordinates into the root SVG viewBox, never page pixels.
  function create(svg,initial,onFrame,onWheelTarget){
    const box=svg.viewBox.baseVal,W=box.width,H=box.height,reduce=matchMedia("(prefers-reduced-motion: reduce)");
    let now=one(),want=one(),ticket=0,time=0,dead=false,drag=null,swallow=false;
    const pointer=new Map();let pinch=null;
    const bound=p=>({z:p.z,x:clamp(p.x,W*(1-p.z),0),y:clamp(p.y,H*(1-p.z),0)});
    if(initial?.W===W&&initial?.H===H)now=bound(initial);
    want={...now};
    function paint(){onFrame(now,W,H)}
    function tick(t){
      ticket=0;if(dead)return;
      const a=reduce.matches?1:1-Math.exp(-Math.min(50,Math.max(1,time?t-time:16))/58);time=t;
      now={z:now.z+(want.z-now.z)*a,x:now.x+(want.x-now.x)*a,y:now.y+(want.y-now.y)*a};
      const done=Math.abs(now.z-want.z)<.0003&&Math.hypot(now.x-want.x,now.y-want.y)<.12;
      if(done){now={...want};time=0}
      paint();if(!done)ticket=requestAnimationFrame(tick);
    }
    function animate(p){want=bound(p);if(!ticket&&!dead){time=0;ticket=requestAnimationFrame(tick)}}
    function point(e){const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;return pt.matrixTransform(svg.getScreenCTM().inverse())}
    // The image point beneath the cursor remains stationary except at a map boundary.
    function zoom(f,x=W/2,y=H/2){
      const z=clamp(want.z*f,1,6),k=z/want.z;animate({z,x:x-(x-want.x)*k,y:y-(y-want.y)*k});
    }
    function go(z,x=W/2,y=H/2){z=clamp(z,1,6);animate({z,x:W/2-x*z,y:H/2-y*z})}
    function wheel(e){
      const dy=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?H:1);
      if(!dy)return;
      e.preventDefault();const p=point(e);
      if(dy<0&&now.z<1.78)onWheelTarget?.((p.x-now.x)/now.z,(p.y-now.y)/now.z);
      zoom(Math.exp(-clamp(dy,-400,400)*.0028),p.x,p.y);
    }
    const ctl=new AbortController(),opts={signal:ctl.signal};
    svg.addEventListener("wheel",wheel,{passive:false,signal:ctl.signal});
    function down(e){
      if(e.button!==0)return;swallow=false;
      const p=point(e);pointer.set(e.pointerId,p);
      if(pointer.size===2){
        const [a,b]=[...pointer.values()];pinch={dist:Math.max(1,Math.hypot(a.x-b.x,a.y-b.y)),cx:(a.x+b.x)/2,cy:(a.y+b.y)/2,cam:{...now}};
        drag=null;swallow=true;for(const id of pointer.keys())svg.setPointerCapture(id);
      }else if(pointer.size===1)drag={id:e.pointerId,p,cam:{...now},active:false};
    }
    function move(e){
      if(!pointer.has(e.pointerId))return;const p=point(e);pointer.set(e.pointerId,p);
      if(pinch&&pointer.size===2){
        const [a,b]=[...pointer.values()],cx=(a.x+b.x)/2,cy=(a.y+b.y)/2,z=clamp(pinch.cam.z*Math.hypot(a.x-b.x,a.y-b.y)/pinch.dist,1,6);
        animate({z,x:cx-(pinch.cx-pinch.cam.x)*z/pinch.cam.z,y:cy-(pinch.cy-pinch.cam.y)*z/pinch.cam.z});return;
      }
      if(!drag||drag.id!==e.pointerId||now.z<=1.002)return;
      const dx=p.x-drag.p.x,dy=p.y-drag.p.y;
      if(!drag.active&&Math.hypot(dx,dy)>4){drag.active=true;swallow=true;svg.setPointerCapture(e.pointerId);svg.classList.add("panning")}
      if(drag.active)animate({z:drag.cam.z,x:drag.cam.x+dx,y:drag.cam.y+dy});
    }
    const up=e=>{pointer.delete(e.pointerId);if(drag?.id===e.pointerId)drag=null;pinch=null;svg.classList.remove("panning");if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId)};
    svg.addEventListener("pointerdown",down,opts);svg.addEventListener("pointermove",move,opts);
    svg.addEventListener("pointerup",up,opts);svg.addEventListener("pointercancel",up,opts);
    svg.addEventListener("pointerleave",e=>{if(!svg.hasPointerCapture(e.pointerId))up(e)},opts);
    svg.addEventListener("click",e=>{if(swallow){swallow=false;e.preventDefault();e.stopImmediatePropagation()}},{capture:true,signal:ctl.signal});
    svg.addEventListener("keydown",e=>{
      if(["+","=","-","0","Home","ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)){
        e.preventDefault();if(e.key==="+"||e.key==="=")zoom(1.45);else if(e.key==="-")zoom(1/1.45);else if(e.key==="0"||e.key==="Home")go(1);
        else animate({...want,x:want.x+(e.key==="ArrowLeft"?60:e.key==="ArrowRight"?-60:0),y:want.y+(e.key==="ArrowUp"?60:e.key==="ArrowDown"?-60:0)});
      }
    },opts);
    paint();
    return {zoom,go,snapshot:()=>({...now,W,H}),destroy(){dead=true;ctl.abort();cancelAnimationFrame(ticket)}};
  }
  window.FlyAtlasCamera={create};
})();
