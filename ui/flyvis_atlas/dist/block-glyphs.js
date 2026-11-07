/* Isometric symbols for the atlas overview. Geometry is local and independent of model data. */
(() => {
  "use strict";

  const paper = "#fffdf8";
  const face = "#f3f2e9";
  const shade = "#e3e6dc";
  const inset = "#d5dbce";
  const warm = "#f5dfcf";
  const copper = "#d28460";

  function project([x, y, z]) {
    return [+(x - y).toFixed(2), +((x + y) / 2 - z).toFixed(2)];
  }

  function vertices(points) {
    return points.map(p => project(p).join(",")).join(" ");
  }

  function panel(points, fill, width = 1.35) {
    return `<polygon points="${vertices(points)}" fill="${fill}" stroke-width="${width}"/>`;
  }

  function trace(points, width = 1, accent = "currentColor") {
    return `<polyline points="${vertices(points)}" fill="none" stroke="${accent}" stroke-width="${width}"/>`;
  }

  function slab(x0, y0, x1, y1, base, top, color = paper) {
    return panel([[x0,y1,base],[x1,y1,base],[x1,y1,top],[x0,y1,top]], face)
      + panel([[x1,y0,base],[x1,y1,base],[x1,y1,top],[x1,y0,top]], shade)
      + panel([[x0,y0,top],[x1,y0,top],[x1,y1,top],[x0,y1,top]], color);
  }

  const stand = () => slab(-19,-17,19,17,3,8);

  function eye() {
    const cap = [];
    const around = 12;
    const latitude = [0, 0.42, 0.85, 1.23, Math.PI / 2];
    function vertex(ring, slice) {
      const angle = slice * 2 * Math.PI / around;
      const radius = 23 * Math.cos(latitude[ring]);
      return [radius * Math.cos(angle), radius * Math.sin(angle), 11 + 27 * Math.sin(latitude[ring])];
    }
    for (let ring = 0; ring < latitude.length - 1; ring++) {
      for (let slice = 0; slice < around; slice++) {
        const pts = [vertex(ring,slice), vertex(ring,slice+1), vertex(ring+1,slice+1), vertex(ring+1,slice)];
        const depth = pts.reduce((sum,[x,y,z]) => sum + x+y+z, 0) / 4;
        const light = Math.cos((slice + 0.5) * 2 * Math.PI / around);
        const color = ring === 3 ? paper : light > 0.45 ? warm : light < -0.35 ? face : paper;
        cap.push({depth, shape:panel(pts,color,0.9)});
      }
    }
    return stand() + '<ellipse cx="0" cy="-9" rx="34" ry="15" fill="#eceee4"/>'
      + cap.sort((a,b) => a.depth-b.depth).map(patch => patch.shape).join("");
  }

  function relay() {
    let parts = slab(-18,-15,18,15,1,6) + slab(-9,-8,9,8,6,32,shade);
    for (const height of [10,18,26,34]) parts += slab(-20,-15,20,15,height-2,height);
    return parts + trace([[-13,-8,34],[10,-8,34],[10,3,34]],1.2,copper);
  }

  function column() {
    let parts = stand();
    const rods = [[-11,-9,38],[0,-9,43],[-11,1,30],[11,-9,35],[0,1,34],[-8,11,24],[11,1,27],[4,11,22]];
    for (const [x,y,height] of rods.sort((a,b) => a[0]+a[1]-b[0]-b[1])) {
      parts += slab(x-2.4,y-2.4,x+2.4,y+2.4,8,height, height===43 ? copper : paper);
    }
    return parts;
  }

  function cross() {
    let parts = stand() + slab(-14,-14,14,14,8,12);
    for (const y of [-9,0,9]) parts += slab(-19,y-1.6,19,y+1.6,12,16);
    for (const x of [-9,0,9]) parts += slab(x-1.6,-16,x+1.6,16,16,22);
    return parts + slab(-4,-4,4,4,22,29,warm);
  }

  function bright() {
    let parts = stand() + slab(-12,-6,12,5,8,12) + slab(-3,-4,3,0,12,20);
    parts += slab(-16,-5,16,-2,17,40)
      + panel([[-13,-1.9,20],[13,-1.9,20],[13,-1.9,37],[-13,-1.9,37]], paper, 0.85)
      + panel([[-12,-1.7,20],[0,-1.7,20],[0,-1.7,36],[-12,-1.7,36]], warm, 0.65)
      + trace([[0,-1.6,20],[0,-1.6,36]],1.6,copper);
    for (const x of [-8,0,8]) parts += trace([[x,5,8],[x,10,8],[x+3,13,8]],0.9);
    return parts;
  }

  function dark() {
    let parts = stand() + slab(-15,-10,15,7,8,32);
    parts += panel([[-12,7,11],[12,7,11],[12,7,29],[-12,7,29]], inset,0.9)
      + panel([[-11,7.1,12],[0,7.1,12],[0,7.1,28],[-11,7.1,28]],"currentColor",0.5)
      + trace([[1,7.4,13],[1,7.4,27]],1.2,copper)
      + slab(-16,-11,16,12,32,35);
    for (const x of [-9,-3,3,9]) parts += trace([[x,-7,35],[x,1,35]],0.85);
    return parts;
  }

  function transfer() {
    let parts = stand() + slab(-17,-12,17,9,8,29);
    for (const x of [-11,0,11]) {
      parts += panel([[x-3.3,9,14],[x+3.3,9,14],[x+3.3,9,23],[x-3.3,9,23]],inset,0.85)
        + trace([[x-1.6,9.2,16],[x-1.6,9.2,21],[x+1.6,9.2,21]],1.1);
    }
    for (const x of [-9,0,9]) parts += trace([[x,-9,29],[x,-3,29],[x+3,0,29],[x+3,5,29]],0.85);
    return parts;
  }

  function mixed() {
    let parts = stand() + slab(-12,-10,12,10,8,12);
    const loop = [[-10,1,12],[-10,1,29],[-10,-6,33],[4,-6,33],[12,-1,31],[12,6,26],[12,6,12]];
    parts += trace(loop,5.5) + trace(loop,2.6,paper);
    const second = [[-6,8,12],[-6,8,19],[1,8,24],[6,4,24],[6,1,20],[6,1,12]];
    return parts + trace(second,4) + trace(second,1.6,copper) + slab(-12,-1,-8,3,12,15,warm);
  }

  const symbols = new Map(Object.entries({eye, relay, column, cross, bright, dark, transfer, mixed})
    .map(([key, draw]) => [key,`<g fill="none" stroke="currentColor" stroke-width="1.45" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">${draw()}</g>`]));
  window.FLY_BLOCK_GLYPH = key => symbols.get(key) || "";
})();
