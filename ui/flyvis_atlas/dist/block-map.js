(() => {
  "use strict";
  // A disjoint display index, not a partition into physiological stages.
  const specs=[
    ["eye","Фоторецепторы","R1–R8","Принимают яркость из зрительного входа модели.",/^R[1-8]$/,[91,213],[96,83]],
    ["relay","Первый слой","L1–L5","Пять типов L передают вход нескольким путям. Например, у L1 есть связь с Mi1, а у L2 — с Tm1.",/^L[1-5]$/,[244,141],[282,103]],
    ["column","Клетки Mi","Mi","Внутренние типы медуллы — одного из зрительных отделов. Среди них есть Mi1 и Mi4; их соседи различаются.",/^Mi\d+$/,[430,91],[115,230]],
    ["cross","Клетки Tm и TmY","Tm · TmY","Здесь собраны типы Tm и TmY. Один из примеров — Tm1, связанный с T5a.",/^Tm(?:Y)?\d+[a-zA-Z]?$/,[440,263],[278,265]],
    ["bright","Светлые границы","T4a–d","Четыре подтипа T4 чувствительны к движению светлых границ в разных направлениях.",/^T4[a-d]$/,[667,121],[92,399]],
    ["dark","Тёмные границы","T5a–d","Четыре подтипа T5 чувствительны к движению тёмных границ в разных направлениях.",/^T5[a-d]$/,[681,319],[280,413]],
    ["transfer","Клетки T1–T3","T1 · T2 · T2a · T3","Четыре типа с близкими обозначениями. Рассмотрите их по отдельности: у них разные входы и выходы.",/^(T1|T2|T2a|T3)$/,[453,392],[278,562]],
    ["mixed","Другие типы","Am · C · CT1 · Lawf","Оставшиеся типы собраны здесь для навигации. Общая буква или соседство на схеме не означают одинаковую функцию.",/^(Am|C[23]|CT1\((Lo1|M10)\)|Lawf[12])$/,[212,328],[94,548]]
  ];
  const paths=[
    {to:"T4",types:["R1","L1","Mi1","T4a"],desc:["Вход модели задаёт яркость фоторецептору R1.","В файле связей есть передача от R1 к L1.","Один из выходов L1 ведёт к Mi1.","От Mi1 есть прямой вход к T4a."]},
    {to:"T5",types:["R1","L2","Tm1","T5a"],desc:["Начинаем с того же фоторецептора R1.","Другой выход R1 ведёт к L2.","От L2 есть прямой переход к Tm1.","У T5a есть вход от Tm1."]}
  ];
  const view={mode:"board",id:null,dir:"out",to:null,path:0,step:0,type:"T4a",scope:"all"};
  const html=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const count=(n,one,few,many)=>n%10===1&&n%100!==11?one:n%10>=2&&n%10<=4&&(n%100<12||n%100>14)?few:many;
  const ntypes=n=>n+" "+count(n,"тип","типа","типов");
  const idFor=t=>specs.find(s=>s[4].test(t))?.[0];
  function model(g){
    const groups=specs.map(([id,title,code,desc,rx,pos,mob])=>({id,title,code,desc,pos,mob,types:g.types.filter(t=>rx.test(t))}));
    const by=Object.fromEntries(groups.map(k=>[k.id,k]));
    if(g.types.some(t=>specs.filter(s=>s[4].test(t)).length!==1))throw Error("Есть тип без однозначной группы");
    const pairs=new Map();for(const edge of g.edges){
      const key=idFor(edge[0])+"/"+idFor(edge[1]);if(!pairs.has(key))pairs.set(key,[]);pairs.get(key).push(edge);
    }
    const routeList=paths.filter(p=>p.types.slice(1).every((t,i)=>g.edges.some(e=>e[0]===p.types[i]&&e[1]===t)));
    return {groups,by,pairs,routeList};
  }
  function reveal(t){
    if(view.mode==="types"){view.type=t;if(view.scope!=="all")view.scope=idFor(t);return}
    view.mode="board";view.id=idFor(t)||null;view.to=null;view.dir="out";
  }
  let active=null, savedCamera=null;
  const narrow=window.matchMedia("(max-width: 640px)");
  narrow.addEventListener("change",()=>{if(active?.host.isConnected)active.draw()});
  function mount(host,g,onType){
    const {groups,by,pairs,routeList}=model(g);
    const group=id=>by[id], coords=k=>k[narrow.matches?"mob":"pos"];
    const typeButton=t=>`<button class="bm-chip" data-nt="${html(t)}" title="Открыть связи типа ${html(t)}">${html(t)}</button>`;
    const pairRows=(a,b)=>pairs.get(a+"/"+b)||[];
    const icon=(id,scale=1)=>`<g transform="scale(${scale})" aria-hidden="true">${window.FLY_BLOCK_GLYPH(id)}</g>`;
    function edges(){
      if(view.mode==="walk"){
        const p=routeList[view.path].types;
        return p.slice(1).map((t,i)=>({a:idFor(p[i]),b:idFor(t),style:i<view.step?"walked":"wait",label:p[i]+" → "+t}));
      }
      if(view.id){
        return groups.filter(k=>k.id!==view.id).map(k=>({a:view.dir==="out"?view.id:k.id,b:view.dir==="out"?k.id:view.id,style:view.dir==="out"?"exit":"entry"}))
          .filter(e=>pairRows(e.a,e.b).length).map(e=>({...e,style:view.to&&view.to!==e.a&&view.to!==e.b?"quiet":e.style,label:group(e.a).title+" → "+group(e.b).title}));
      }
      const shown=new Map();
      function add(a,b,style){if(a!==b&&pairRows(a,b).length)shown.set(a+"/"+b,{a,b,style,label:group(a).title+" → "+group(b).title})}
      for(const p of routeList)p.types.slice(1).forEach((t,i)=>add(idFor(p.types[i]),idFor(t),"overview"));
      for(const k of ["mixed","transfer"]){
        const best=groups.filter(v=>v.id!==k&&pairRows(v.id,k).length).sort((x,y)=>pairRows(y.id,k).length-pairRows(x.id,k).length)[0];
        if(best)add(best.id,k,"quiet");
      }
      return [...shown.values()];
    }
    function wire(e){
      const [x,y]=coords(group(e.a)),[X,Y]=coords(group(e.b)),dx=X-x,dy=Y-y,l=Math.hypot(dx,dy),ux=dx/l,uy=dy/l;
      // Elliptical ports on the middle of each device, outside its artwork.
      const r=1/Math.sqrt((ux/44)**2+(uy/31)**2),off=-13,s=[x+ux*r,y+off+uy*r],end=[X-ux*(r+4),Y+off-uy*(r+4)];
      const bend=Math.min(28,l*.095)*(e.a<e.b?1:-1),cx=(s[0]+end[0])*.5-uy*bend,cy=(s[1]+end[1])*.5+ux*bend;
      return `<path class="bm-wire ${e.style}" d="M${s} Q${cx} ${cy} ${end}" marker-end="url(#bm-tip-${e.style})"><title>${html(e.label)}</title></path>`;
    }
    function board(){
      const small=narrow.matches,lines=edges(),seq=view.mode==="walk"?routeList[view.path].types:null,chosen=seq?idFor(seq[view.step]):view.id;
      const marks=["overview","quiet","wait","entry","exit","walked"].map(s=>`<marker id="bm-tip-${s}" viewBox="0 0 7 7" markerWidth="6.5" markerHeight="6.5" refX="6" refY="3.5" orient="auto"><path class="bm-tip ${s}" d="M0 0 L7 3.5 L0 7z"/></marker>`).join("");
      const nodes=groups.map(k=>{
        const [x,y]=coords(k), selected=k.id===chosen,related=seq?seq.some(t=>idFor(t)===k.id):!chosen||selected||lines.some(e=>e.a===k.id||e.b===k.id);
        return `<g role="button" tabindex="0" aria-pressed="${selected}" aria-label="Блок ${html(k.title)}, ${ntypes(k.types.length)}" data-block="${k.id}" class="bm-node${selected?" chosen":""}${related?"":" distant"}" transform="translate(${x} ${y})"><title>${html(k.code)} · ${ntypes(k.types.length)}</title><rect class="bm-hit" x="-83" y="-61" width="166" height="122" rx="14"/><ellipse class="bm-base" cy="9" rx="45" ry="11"/>${icon(k.id,small?.91:1.03)}<text class="bm-name" y="39" text-anchor="middle">${html(k.title)}</text><text class="bm-id" y="56" text-anchor="middle">${html(k.code)}</text></g>`;
      }).join("");
      return `<svg class="bm-svg" viewBox="0 0 ${small?"380 630":"785 460"}" tabindex="0" role="group" aria-label="Интерактивная карта FlyVis. Колесо или клавиши плюс и минус меняют масштаб, стрелки перемещают карту"><defs><pattern id="bm-grid" width="54" height="31.18" patternUnits="userSpaceOnUse"><path d="M-27 15.59L27 46.77 M0 0L54 31.18 M27 -15.59L81 15.59 M-27 15.59L27 -15.59 M0 31.18L54 0 M27 46.77L81 15.59" fill="none" stroke="#d9d3c8" stroke-opacity=".58" stroke-width=".65"/></pattern>${marks}</defs><rect width="100%" height="100%" fill="#fbfaf6"/><g class="bm-world"><rect width="100%" height="100%" fill="url(#bm-grid)"/><g class="bm-far"><g aria-hidden="true">${lines.map(wire).join("")}</g>${nodes}</g><g class="bm-near" opacity="0" aria-hidden="true">${detailField()}</g></g></svg>`;
    }
    function overview(){
      return `<div class="bm-eyebrow">Путеводитель</div><h2>С чего начать</h2><p>Нажмите на фоторецепторы слева: увидите, к каким группам от них идут связи. Затем выберите тип клетки, чтобы рассмотреть его соседей.</p><div class="bm-rule"></div><div class="bm-smallhead">Две цепочки для знакомства</div>${routeList.map((p,i)=>`<button class="bm-tourlink" data-tour="${i}"><b>К семейству ${p.to}</b><span>${p.types.join(" → ")}</span><i>↗</i></button>`).join("")}<div class="bm-note">На обзоре показана часть связей. У выбранного блока можно посмотреть все входы и выходы.</div>`;
    }
    function walk(){
      const p=routeList[view.path],t=p.types[view.step];
      return `<div class="bm-eyebrow">Маршрут · ${view.step+1} из ${p.types.length}</div><h2>Путь к ${p.to}</h2><label class="bm-field">Цепочка<select id="bm-path">${routeList.map((r,i)=>`<option value="${i}" ${i===view.path?"selected":""}>${r.types.join(" → ")}</option>`).join("")}</select></label><div class="bm-stops">${p.types.map((v,i)=>`<button data-step="${i}" aria-pressed="${view.step===i}"><small>0${i+1}</small> ${v}</button>`).join("")}</div><h3>${t}</h3><p>${p.desc[view.step]}</p><div class="bm-nav"><button data-step="${view.step-1}" ${view.step===0?"disabled":""}>← Назад</button><button class="bm-next" data-step="${view.step+1}" ${view.step===p.types.length-1?"disabled":""}>Далее →</button></div>${typeButton(t)}<div class="bm-note">Шаг означает переход по записанной связи, а не замер времени или активности.</div>`;
    }
    function detail(){
      const k=group(view.id),incoming=view.dir==="in";
      const all=groups.map(j=>({j,rows:incoming?pairRows(j.id,k.id):pairRows(k.id,j.id)})).filter(s=>s.rows.length).sort((a,b)=>b.rows.length-a.rows.length);
      const selected=all.find(s=>s.j.id===view.to);
      return `<div class="bm-eyebrow">Выбранный блок · ${ntypes(k.types.length)}</div><h2>${html(k.title)}</h2><p>${html(k.desc)}</p><div class="bm-smallhead">Типы клеток</div><div class="bm-chips">${k.types.map(typeButton).join("")}</div><div class="bm-rule"></div><div class="bm-dir"><button data-dir="in" aria-pressed="${incoming}">Входы</button><button data-dir="out" aria-pressed="${!incoming}">Выходы</button></div><p class="bm-hint">Выберите строку: число справа — количество пар типов.</p><div class="bm-links">${all.map(({j,rows})=>`<button data-pair="${j.id}" aria-expanded="${j.id===view.to}"><span>${j.id===k.id?"↻ Внутри блока":(incoming?"← ":"→ ")+html(j.title)}</span><b>${rows.length}</b></button>`).join("")||"В файле модели таких связей нет."}</div>${selected?`<div class="bm-edgepairs"><strong>${incoming?html(selected.j.title)+" → "+html(k.title):html(k.title)+" → "+html(selected.j.title)}</strong><div tabindex="0" aria-label="Все пары типов">${selected.rows.map(([a,b])=>`<span>${html(a)} → ${html(b)}</span>`).join("")}</div></div>`:""}<button class="bm-reset" data-clear>← Вся схема</button>`;
    }
    function positions(){
      const map=new Map();for(const k of groups){
        const [cx,cy]=coords(k),cols=Math.min(k.types.length,k.types.length>14?6:4),rows=Math.ceil(k.types.length/cols);
        k.types.forEach((t,i)=>{
          const row=Math.floor(i/cols),lineCount=Math.min(cols,k.types.length-row*cols);
          const dx=(i%cols-(lineCount-1)/2)*25,dy=(row-(rows-1)/2)*24;
          map.set(t,{x:cx+dx,y:cy+dy,dx,dy,k});
        });
      }return map;
    }
    function detailField(){
      const ps=positions(),t=view.type,from=view.dir==="out";
      const selected=g.edges.filter(e=>e[from?0:1]===t),neighbors=new Set(selected.map(e=>e[from?1:0]));
      const col=from?"#ad5335":"#34796b";
      const wires=selected.map(([a,b])=>{
        const v=ps.get(a),w=ps.get(b),dx=w.x-v.x,dy=w.y-v.y,len=Math.hypot(dx,dy);
        if(!len)return `<path d="M${v.x+5},${v.y-3} c12,-12 13,9 3,5" marker-end="url(#bm-dtip)"/>`;
        const ax=v.x+dx/len*7,ay=v.y+dy/len*7,bx=w.x-dx/len*9,by=w.y-dy/len*9;
        return `<path d="M${ax} ${ay} Q${(ax+bx)/2-dy*.1} ${(ay+by)/2+dx*.1} ${bx} ${by}" marker-end="url(#bm-dtip)"/>`;
      }).join("");
      return `<defs><marker id="bm-dtip" viewBox="0 0 5 5" markerWidth="4" markerHeight="4" refX="4.5" refY="2.5" orient="auto"><path fill="${col}" d="M0 0 L5 2.5 L0 5z"/></marker></defs><g class="bm-localEdges" stroke="${col}" stroke-width=".95" fill="none">${wires}</g>`+groups.map(k=>{
        const [cx,cy]=coords(k),slots=k.types.map(t=>ps.get(t)),top=Math.min(...slots.map(s=>s.dy))-21;
        return `<g transform="translate(${cx} ${cy})"><g class="bm-cluster"><text y="${top}" class="bm-cluster-label" text-anchor="middle">${html(k.title)}</text>${k.types.map(name=>{
          const p=ps.get(name),chosen=name===t,adj=neighbors.has(name);
          return `<g transform="translate(${p.dx} ${p.dy})" class="bm-element ${chosen?"chosen":adj?"neighbor":""}" data-unit="${html(name)}" role="button" tabindex="-1" aria-label="Тип ${html(name)}" aria-pressed="${chosen}"><title>${html(k.title)} · ${html(name)}</title><rect x="-12" y="-13" width="24" height="24" rx="3"/>${icon(k.id,.189)}<text y="7.7" text-anchor="middle">${html(name)}</text></g>`;
        }).join("")}</g></g>`;
      }).join("");
    }
    function denseInfo(){
      const t=view.type,k=by[idFor(t)],out=view.dir==="out",list=g.edges.filter(e=>e[out?0:1]===t).sort((a,b)=>b[2]-a[2]),adj=list.map(e=>e[out?1:0]);
      return `<div class="bm-eyebrow">Отдельный тип · ${html(k.title)}</div><h2>${html(t)}</h2><p>Выберите входы или выходы. Цветом отмечены непосредственные соседи этого типа.</p><div class="bm-dir"><button data-dir="in" aria-pressed="${!out}">Входы</button><button data-dir="out" aria-pressed="${out}">Выходы</button></div><div class="bm-smallhead">${out?"Передаёт в":"Получает от"} · ${adj.length}</div><div class="bm-chips">${adj.map(v=>`<button class="bm-chip" data-jump="${html(v)}" title="Выбрать ${html(v)} на карте">${v===t?"↻ ":""}${html(v)}</button>`).join("")||"В файле связей нет."}</div><div class="bm-rule"></div><button data-nt="${html(t)}">Карточка типа ${html(t)} →</button><div class="bm-note">Перетащите карту, чтобы увидеть соседние типы. Направление стрелки — от источника к получателю.</div>`;
    }
    let camera=null,world,far,near,lastScaleLevel=null,wheelGroup=null;
    function tabs(){
      return `<div class="bm-switch" aria-label="Режим схемы"><button data-mode="board" aria-pressed="${view.mode==="board"}"><span aria-hidden="true">◇</span> 8 блоков</button><button data-mode="types" aria-pressed="${view.mode==="types"}"><span aria-hidden="true">▦</span> 65 типов</button><button data-mode="walk" aria-pressed="${view.mode==="walk"}"><span aria-hidden="true">⇢</span> По шагам</button></div>`;
    }
    function panel(){
      const aside=host.querySelector(".bm-side");
      aside.innerHTML=view.mode==="types"?denseInfo():view.mode==="walk"?walk():view.id?detail():overview();
      if(view.mode==="types")aside.insertAdjacentHTML("afterbegin",`<label class="bm-field">Найти тип на карте<select id="bm-find">${g.types.map(n=>`<option ${n===view.type?"selected":""}>${html(n)}</option>`).join("")}</select></label>`);
      if(view.mode==="board"&&view.id)aside.querySelector(".bm-chips").insertAdjacentHTML("afterend",`<button class="bm-expand" data-expand="${view.id}">Приблизить блок →</button>`);
      host.querySelectorAll("[data-mode]").forEach(el=>el.setAttribute("aria-pressed",String(el.dataset.mode===view.mode)));
      host.querySelector(".bm-legend span:last-child").textContent=view.mode==="types"?view.type+" · "+(view.dir==="out"?"выходы":"входы"):view.mode==="walk"?"Выделен выбранный маршрут":view.id?(view.dir==="out"?"Выходы выбранного блока":"Входы выбранного блока"):"Колесо — масштаб · перетаскивание — перемещение";
    }
    function onFrame(p,w,h){
      const phase=Math.max(0,Math.min(1,(p.z-1.20)/1.04)),mix=phase*phase*(3-2*phase);
      world.setAttribute("transform",`translate(${p.x} ${p.y}) scale(${p.z})`);
      far.setAttribute("opacity",1-mix);near.setAttribute("opacity",mix);
      far.style.visibility=mix===1?"hidden":"visible";near.style.visibility=mix===0?"hidden":"visible";
      near.querySelectorAll(".bm-cluster").forEach(k=>k.setAttribute("transform","scale("+(.50+.50*mix)+")"));
      near.querySelector(".bm-localEdges").style.opacity=mix*mix;
      const level=lastScaleLevel===null?p.z>=1.75:lastScaleLevel?p.z>1.63:p.z>=1.78;
      if(lastScaleLevel!==level){
        if(level&&wheelGroup){
          if(idFor(view.type)!==wheelGroup.id)view.type=wheelGroup.types[0];
          near.innerHTML=detailField();wheelGroup=null;
        }
        lastScaleLevel=level;far.setAttribute("aria-hidden",String(level));near.setAttribute("aria-hidden",String(!level));
        far.style.pointerEvents=level?"none":"auto";near.style.pointerEvents=level?"auto":"none";
        far.querySelectorAll("[data-block]").forEach(k=>k.setAttribute("tabindex",level?"-1":"0"));
        near.querySelectorAll("[data-unit]").forEach(k=>k.setAttribute("tabindex",level?"0":"-1"));
        if(level){
          view.mode="types";
        }else if(view.mode==="types"){view.mode="board"}
        panel();
      }
      const scale=host.querySelector(".bm-zval");scale.textContent=Math.round(p.z*100)+"%";
      host.querySelector("[data-zoom='minus']").disabled=p.z<=1.001;
      host.querySelector("[data-zoom='plus']").disabled=p.z>=5.999;
      world.ownerSVGElement.classList.toggle("can-pan",level||p.z>1.03);
      savedCamera={...p,W:w,H:h};
    }
    function draw(){
      camera?.destroy();lastScaleLevel=null;
      host.className="block-map";
      host.innerHTML=`<div class="bm-layout"><section class="bm-plate"><div class="bm-top">${tabs()}<div class="bm-camera" aria-label="Масштаб карты"><button data-zoom="minus" aria-label="Отдалить">−</button><button class="bm-zval" data-zoom="fit" aria-label="Показать карту целиком" title="Показать всю карту">100%</button><button data-zoom="plus" aria-label="Приблизить">+</button></div></div><div class="bm-chart">${board()}</div><div class="bm-legend"><span><i></i> Направление связи</span><span></span></div></section><aside class="bm-side" aria-label="Сведения о выборе"></aside></div><p class="bm-caption">Колесо или жест двумя пальцами меняют масштаб. Вдали видны восемь групп, вблизи — отдельные типы. Блоки условны; направления связей взяты из модели FlyVis.</p>`;
      world=host.querySelector(".bm-world");far=world.querySelector(".bm-far");near=world.querySelector(".bm-near");
      wheelGroup=null;camera=window.FlyAtlasCamera.create(world.ownerSVGElement,savedCamera,onFrame,(x,y)=>{
        const distance=k=>{const [a,b]=coords(k);return Math.hypot(a-x,b-y)};
        wheelGroup=groups.reduce((a,b)=>distance(a)<=distance(b)?a:b);
      });
    }
    function gotoType(t,scale){
      const p=positions().get(t);if(!p)return;
      view.type=t;view.id=p.k.id;view.to=null;draw();
      camera.go(scale??Math.max(3.4,savedCamera?.z||1),p.x,p.y);
    }
    host.onclick=ev=>{
      const el=ev.target.closest("[data-mode],[data-block],[data-nt],[data-dir],[data-pair],[data-tour],[data-step],[data-clear],[data-expand],[data-unit],[data-jump],[data-zoom]");if(!el||el.disabled)return;
      const d=el.dataset;
      if(d.nt){camera?.destroy();onType(d.nt);return}
      if(d.zoom){if(d.zoom==="fit")camera.go(1);else camera.zoom(d.zoom==="plus"?1.48:1/1.48);return}
      if(d.mode==="board"){view.id=null;view.to=null;if(view.mode==="walk")view.mode="board";draw();camera.go(1);return}
      if(d.mode==="types"||d.expand){
        const k=group(d.expand||view.id||idFor(view.type));const t=k.types.includes(view.type)?view.type:k.types[0];
        view.type=t;view.id=k.id;view.to=null;draw();const [x,y]=coords(k);camera.go(3.45,x,y);return;
      }
      if(d.tour!==undefined||d.mode==="walk"){
        if(!routeList.length)return;
        if(d.tour!==undefined)view.path=Number(d.tour);view.step=0;
        // Keep the tour chosen as the camera smoothly returns to its overview.
        view.mode="walk";draw();if(view.mode!=="walk"){view.mode="walk";panel()}camera.go(1);return;
      }
      if(d.step!==undefined)view.step=Math.max(0,Math.min(routeList[view.path].types.length-1,Number(d.step)));
      else if(d.block){view.id=d.block;view.type=by[d.block].types.includes(view.type)?view.type:by[d.block].types[0];view.mode="board";view.dir="out";view.to=null}
      else if(d.unit){view.type=d.unit;view.id=idFor(d.unit);view.to=null}
      else if(d.jump){gotoType(d.jump);return}
      else if(d.dir){view.dir=d.dir;view.to=null}
      else if(d.pair){view.to=view.to===d.pair?null:d.pair}
      else{view.id=null;view.to=null;view.mode="board"}
      const attr=Object.keys(d)[0];draw();
      host.querySelector(`[data-${attr}="${CSS.escape(d[attr]??"")}"]`)?.focus({preventScroll:true});
    };
    host.onchange=e=>{if(e.target.id==="bm-path"){view.path=Number(e.target.value);view.step=0;draw();host.querySelector("#bm-path")?.focus({preventScroll:true})}else if(e.target.id==="bm-find"){gotoType(e.target.value)}};
    host.onkeydown=e=>{if((e.target.hasAttribute("data-block")||e.target.hasAttribute("data-unit"))&&["Enter"," "].includes(e.key)){e.preventDefault();e.target.dispatchEvent(new MouseEvent("click",{bubbles:true}))}};
    active?.disconnect();active={host,draw,disconnect:()=>camera?.destroy()};draw();
  }
  window.FlyBlockMap={mount,reveal,model,leave(){active?.disconnect();active=null}};
})();
