(() => {
  "use strict";
  const root = document.getElementById("main");
  const archive = window.FLY_EPISODES || {episodes:[],channels:[],points:[]};
  const study = window.FLY_STUDY || null;
  const core = window.FLY_CORE || null;
  const graph = window.FLY_CONNECTOME || {types:[],edges:[]};
  const channels = archive.channels || [];
  const episodes = archive.episodes || [];
  const points = archive.points || [];
  const graphTypes = graph.types || [];
  const store = {ep:0, at:0, stage:0, channel:Math.max(0,channels.indexOf("T4a")), cell:null,
    mask:"all", outlines:true, zoom:false, focus:graphTypes.includes("T4a")?"T4a":graphTypes[0],
    sort:"order", running:false, playHandle:0, file:null, fileResult:null, fileFrames:[],
    fileChannel:Math.max(0,channels.indexOf("T4a")),fileAt:0,generation:0,mediaAvailable:null};
  const fmt = (v, n=3) => Number.isFinite(Number(v)) && v!==null ? Number(v).toLocaleString("ru-RU",{maximumFractionDigits:n,minimumFractionDigits:n}) : "—";
  const signed = (v,n=3) => Number.isFinite(Number(v)) ? (Number(v)>0?"+":"")+fmt(v,n) : "—";
  const esc = s => String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const $ = id => document.getElementById(id);
  const selectItems = (items,selected) => items.map(([v,label])=>`<option value="${esc(v)}" ${String(v)===String(selected)?"selected":""}>${esc(label)}</option>`).join("");
  const choices = (items,i) => selectItems(items.map((s,k)=>[k,s]),i);
  const head = (k,h,p,metrics=[]) => `<section class="hero"><div><p class="kicker">${k}</p><h1>${h}</h1><p>${p}</p></div><div class="measures">${metrics.map(([n,s])=>`<div class="measure"><strong>${esc(n)}</strong><small>${esc(s)}</small></div>`).join("")}</div></section>`;
  const steps = ["Видеокадр","Составной глаз","Карта клеток","Выбор трека"];
  const messages = [
    ["Какой участок попал в глаз","Рамка цвета терракоты отмечает область, которую получил зрительный модуль. Справа видна яркость в каждой из 721 позиции его глаза."],
    ["Как глаз представляет кадр","Каждый шестиугольник справа показывает одну позицию глаза после подготовки яркости. Перемотайте запись, чтобы увидеть, как рисунок на сетке меняется при движении."],
    ["Где откликнулся выбранный тип","Цвет обозначает активность относительно среднего этого канала в исходном наборе. Нажмите на ячейку, чтобы увидеть её номер и значение."],
    ["Как две рамки получают балл","Карта показывает один выбранный тип. Балл каждой рамки вычисляется по восьми каналам T4/T5 и прошлому образцу цели. Переключите «Вся рамка» и «Только свои», чтобы увидеть смену решения."]
  ];
  const getEp=()=>episodes[store.ep];
  const stop=()=>{clearInterval(store.playHandle);store.running=false};
  function mapAt(ep,i){return ep.frames[i]?.maps || (i===ep.finalIndex?ep.maps:null)}
  function indices(ep){
    if(store.stage===3)return [ep.finalIndex];
    return ep.frames.map((f,i)=>i).filter(i=>store.stage<2||mapAt(ep,i));
  }
  function clampFrame(ep){const allowed=indices(ep);if(!allowed.includes(store.at))store.at=allowed.at(-1)??ep.finalIndex}
  const channelLabel=i=>channels[i]||String(i);
  function canvas2d(node){
    const {width:w,height:h}=node.getBoundingClientRect();
    if(!w||!h)return null;
    const dpr=Math.min(2,window.devicePixelRatio||1);
    node.width=Math.round(w*dpr);node.height=Math.round(h*dpr);
    const ctx=node.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
    return {ctx,w,h};
  }
  const blend=(a,b,t)=>`rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(",")})`;
  const color=(v,max,isEye)=>isEye ? blend([39,42,38],[249,248,237],Math.max(0,Math.min(1,v))) :
    blend([244,240,226],v<0?[48,114,99]:[190,88,52],Math.min(1,Math.abs(v)/(max||1)));
  const pointBins = new WeakMap();
  function maskOutline(ctx,coords,indices,ink,dash){
    if(!indices?.length)return;
    const v=indices.map(i=>coords[i]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const half=list=>{const a=[];for(const p of list){while(a.length>1&&cross(a.at(-2),a.at(-1),p)<=0)a.pop();a.push(p)}return a.slice(0,-1)};
    const hull=v.length>2?[...half(v),...half(v.slice().reverse())]:v;
    ctx.save();ctx.beginPath();hull.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.lineWidth=1.6;ctx.setLineDash(dash);ctx.strokeStyle=ink;ctx.stroke();ctx.restore();
  }
  const scaleCache = new WeakMap();
  function mapScale(maps,channel,owner,kind){
    if(!owner || !maps || !maps[channel])return 1;
    let c=scaleCache.get(owner);if(!c){c=new Map();scaleCache.set(owner,c)}
    const key=kind+channel;if(c.has(key))return c.get(key);
    const list=kind==="file"?owner.maps:owner.frames.map((f,i)=>mapAt(owner,i)).filter(Boolean);
    let bound=0;
    for(const matrix of list)for(const x of matrix[channel]||[])bound=Math.max(bound,Math.abs(x));
    c.set(key,bound||1);return bound||1;
  }
  function drawMap(node,values,opts={}){
    if(!node)return;
    const d=canvas2d(node);if(!d)return;
    const {ctx,w,h}=d;
    if(!values || values.length!==points.length || !points.length){ctx.fillStyle="#71665a";ctx.textAlign="center";ctx.font="14px Plex, sans-serif";ctx.fillText("Карта для этого кадра отсутствует",w/2,h/2);return}
    const allX=points.map(p=>p[0]), allY=points.map(p=>p[1]);
    const loX=Math.min(...allX),hiX=Math.max(...allX),loY=Math.min(...allY),hiY=Math.max(...allY);
    const scale=Math.min(w*.89/(hiX-loX),h*.88/(hiY-loY));
    const offsetX=(w-(hiX-loX)*scale)/2,offsetY=(h-(hiY-loY)*scale)/2;
    const coords=points.map(([x,y])=>[offsetX+(x-loX)*scale,offsetY+(y-loY)*scale]);
    const nearest=Math.min(...coords.slice(1).map(([x,y])=>Math.hypot(x-coords[0][0],y-coords[0][1])));
    const r=nearest/Math.sqrt(3)*.98;
    const direction=coords.find(([x,y],j)=>j&&Math.abs(Math.hypot(x-coords[0][0],y-coords[0][1])-nearest)<.05);
    const angle=direction?Math.atan2(direction[1]-coords[0][1],direction[0]-coords[0][0])+Math.PI/6:Math.PI/6;
    const ring=Array.from({length:6},(_,k)=>[r*Math.cos(k*Math.PI/3+angle),r*Math.sin(k*Math.PI/3+angle)]);
    const maskA=opts.masks?new Set(opts.masks[0]):null,maskB=opts.masks?new Set(opts.masks[1]):null;
    for(let i=0;i<coords.length;i++){
      const [x,y]=coords[i], active=!maskA || (opts.exclusive?(maskA.has(i)!==maskB.has(i)):(maskA.has(i)||maskB.has(i)));
      ctx.beginPath();ring.forEach(([dx,dy],k)=>k?ctx.lineTo(x+dx,y+dy):ctx.moveTo(x+dx,y+dy));ctx.closePath();
      ctx.globalAlpha=active?1:.14;ctx.fillStyle=color(values[i],opts.bound||1,opts.eye);ctx.fill();
    }
    ctx.globalAlpha=1;
    if(opts.masks&&store.outlines){
      maskOutline(ctx,coords,opts.masks[0],"#6c4a77",[]);
      maskOutline(ctx,coords,opts.masks[1],"#326e66",[4,2]);
    }
    if(opts.cell!=null && coords[opts.cell]){const [x,y]=coords[opts.cell];ctx.strokeStyle="#fff";ctx.lineWidth=4;ctx.beginPath();ctx.arc(x,y,r+1,0,Math.PI*2);ctx.stroke();ctx.strokeStyle="#3c342e";ctx.lineWidth=1.6;ctx.stroke()}
    pointBins.set(node,{coords,max:nearest*1.06});
  }
  function hitCell(node,event){
    const item=pointBins.get(node);if(!item)return null;
    const rect=node.getBoundingClientRect();let best=item.max,id=null;
    item.coords.forEach(([x,y],i)=>{const v=Math.hypot(x-(event.clientX-rect.left),y-(event.clientY-rect.top));if(v<=best){id=i;best=v}});
    return id;
  }
  const imageCache=new Map();
  function imageFor(src){
    if(!src)return null;
    if(!imageCache.has(src)){
      const image=new Image();imageCache.set(src,image);
      image.onload=()=>{if(route()==="lab")drawLab()};image.onerror=()=>{if(route()==="lab")drawLab()};
      image.src=src;
    }
    return imageCache.get(src);
  }
  function frameCanvas(node,ep,f,overlay,final){
    const image=imageFor(f.image);
    if(!image || !image.complete || !image.naturalWidth){
      drawMap(node,f.eye,{eye:true});return image?.complete?"Яркость входа · исходное фото локально недоступно":"Вход глаза · 721 позиция";
    }
    const d=canvas2d(node);if(!d)return "";
    const {ctx,w,h}=d,[ow,oh]=ep.originalSize;
    let scale=Math.min(w/ow,h/oh);
    let l=(w-ow*scale)/2,t=(h-oh*scale)/2;
    if(store.zoom && f.crop){const [x,y,r,b]=f.crop;scale=Math.min(w/Math.max(1,r-x),h/Math.max(1,b-y))*.76;
      l=(w-(x+r)*scale)/2;t=(h-(y+b)*scale)/2}
    ctx.drawImage(image,l,t,ow*scale,oh*scale);
    const box=(coords,ink,label,dash=[])=>{
      if(!coords||coords.length!==4)return;
      const [x,y,r,b]=coords.map((v,k)=>k%2?t+v*scale:l+v*scale);
      ctx.lineWidth=2;ctx.setLineDash(dash);ctx.strokeStyle=ink;ctx.strokeRect(x,y,r-x,b-y);ctx.setLineDash([]);
      ctx.font="600 10px Plex,sans-serif";const tw=ctx.measureText(label).width;
      const ly=Math.max(t+1,y-15);ctx.fillStyle=ink;ctx.fillRect(x-1,ly,tw+8,14);ctx.fillStyle="#fff";ctx.fillText(label,x+3,ly+11);
    };
    if(overlay){box(f.crop,"#b45332","ГЛАЗ",final?[4,3]:[]);if(final)ep.boxes?.forEach((b,i)=>box(b,i?"#326e66":"#6c4a77",i?"Б":"А"))}
    return overlay?"Терракотовая рамка — вход глаза":"Поле исходного кадра";
  }
  function fig(id,title,meta,foot,legend,extra=""){
    return `<figure class="figure"><figcaption class="fighead"><span>${title}</span><span id="${id}-meta" class="code">${meta}</span></figcaption><div class="picture"><canvas id="${id}" ${extra} aria-label="${title}"></canvas></div><div class="figfoot"><span id="${id}-foot">${foot}</span>${legend||""}</div></figure>`;
  }
  function drawLab(){
    if(route()!=="lab"||!$("field-a"))return;
    const ep=getEp(),f=ep?.frames[store.at];if(!f)return;
    const final=store.at===ep.finalIndex,isEye=store.stage<2, maps=mapAt(ep,store.at);
    const avail=indices(ep),vals=isEye?f.eye:maps?.[store.channel];
    $("field-a-foot").textContent=frameCanvas($("field-a"),ep,f,store.outlines,final&&store.stage===3);
    $("field-a-meta").innerHTML=`<span>Кадр ${Number(f.frame)}</span> <button id="frame-zoom" class="zoom" aria-label="${store.zoom?"Показать весь кадр":"Увеличить область глаза"}" aria-pressed="${store.zoom}">${store.zoom?"−":"+ "}</button>`;
    $("frame-zoom").onclick=()=>{store.zoom=!store.zoom;drawLab();$("frame-zoom")?.focus({preventScroll:true})};
    const bound=isEye?1:mapScale(maps,store.channel,ep,"atlas");
    drawMap($("field-b"),vals,{eye:isEye,bound,cell:store.cell,masks:store.stage===3?ep.masks:null,exclusive:store.mask==="exclusive"});
    $("field-b-meta").textContent=isEye?"BoxEye":channelLabel(store.channel);
    $("field-b-foot").textContent=store.cell!=null && vals?`Ячейка ${store.cell} · ${fmt(vals[store.cell])}`:(isEye?"Значение яркости":"Активность, норм. ед.");
    $("map-range").innerHTML=isEye?'<span>0</span><i class="gradient gray"></i><span>1</span>':`<span>−${fmt(bound,1)}</span><i class="gradient"></i><span>+${fmt(bound,1)}</span>`;
    const slider=$("timeline");slider.max=Math.max(0,avail.length-1);slider.value=avail.indexOf(store.at);slider.disabled=avail.length<2;
    slider.setAttribute("aria-valuetext","Видеокадр "+f.frame);
    $("frame-count").textContent=`${avail.indexOf(store.at)+1} / ${avail.length}`;$("play").disabled=avail.length<2;
    $("play").textContent=store.running?"Ⅱ":"▶";$("play").setAttribute("aria-label",store.running?"Приостановить":"Проиграть кадры");$("previous").disabled=avail.indexOf(store.at)<=0;$("next").disabled=avail.indexOf(store.at)>=avail.length-1;
    if($("cell-input"))$("cell-input").value=store.cell??"";
    if($("cell-study")){const entry=study?.positions?.[store.cell];$("cell-study").textContent=store.cell!=null&&entry?`При ослаблении позиции ${store.cell} в исследовании доля верных ответов изменилась на ${signed(entry.delta,2)} п. п.`:""}
    if($("map-desc"))$("map-desc").textContent=store.cell!=null&&vals?`Ячейка ${store.cell}, значение ${fmt(vals[store.cell],4)}`:"";
  }
  function renderLab(){
    const ep=getEp();
    root.innerHTML=head("Лаборатория зрения","Как видит муха","Пройдите путь от кадра до выбора объекта. На каждом шаге — сохранённые данные зрительной модели FlyVis.",[[721,"позиция глаза"],[channels.length||34,"карты клеток"]]);
    if(!ep){root.insertAdjacentHTML("beforeend",'<div class="panel">Для просмотра нужен файл данных эпизодов из комплекта атласа.</div>');return}
    clampFrame(ep);
    const s=store.stage;
    const channelSelect=`<label class="field">Тип клеток<select id="channel">${choices(channels,store.channel)}</select></label>`;
    const cellSelect=`<label class="field">Ячейка 0–720<input id="cell-input" type="number" min="0" max="720" step="1" placeholder="№" aria-label="Номер ячейки"></label>`;
    const tabs=steps.map((x,i)=>`<button data-step="${i}" aria-pressed="${i===s}"><span class="stepnum">${i+1}</span>${x}</button>`).join("");
    root.insertAdjacentHTML("beforeend",`<section class="bench" aria-label="Просмотр эпизода"><div class="bench-head"><h2>Маршрут сигнала</h2><label class="field">Эпизод<select id="episode">${selectItems(episodes.map((e,i)=>[i,`${e.video.slice(0,8)} · кадр ${e.frame}`]),store.ep)}</select></label></div>
    <div class="stagebar" aria-label="Шаг маршрута">${tabs}</div><div class="figure-pair">
    ${fig("field-a","Исходный кадр","","","")}
    ${fig("field-b",s<2?"Глаз мухи":s===2?"Карта активности":"Ячейки в рамках","","",'<span id="map-range" class="legend"></span>','class="cmap" aria-describedby="map-desc"')}
    </div><div class="sys-controls"><button class="icon-button play" id="play" aria-label="Проиграть кадры">▶</button><button class="icon-button" id="previous" aria-label="Предыдущий кадр">‹</button><input class="timeline" id="timeline" type="range" min="0" step="1" aria-label="Кадры эпизода"><button class="icon-button" id="next" aria-label="Следующий кадр">›</button><span class="count" id="frame-count"></span><span class="playlabel">${s===3?"момент выбора":s===2?"кадры с картой":"кадры входа"}</span></div>
    <div class="detail"><div class="text"><h2>${messages[s][0]}</h2><p>${messages[s][1]}</p></div><div class="side-settings">${s>=2?channelSelect:""}${s<3?cellSelect:""}<label class="tools"><span>Разметка кадра</span><input id="outlines" type="checkbox" ${store.outlines?"checked":""}></label></div></div></section>
    <p id="map-desc" class="sr" aria-live="polite"></p>${s===3?scoreDetails(ep):labBelow(s)}`);
    $("episode").onchange=e=>{stop();store.ep=Number(e.target.value);store.cell=null;store.at=getEp().finalIndex;renderLab()};
    root.querySelectorAll("[data-step]").forEach(b=>b.onclick=()=>{stop();store.stage=Number(b.dataset.step);renderLab();root.querySelector(`[data-step="${store.stage}"]`).focus({preventScroll:true})});
    if($("channel"))$("channel").onchange=e=>{store.channel=Number(e.target.value);drawLab();updateChannelDescription()};
    if($("cell-input"))$("cell-input").oninput=e=>{const raw=e.target.value,v=Number(raw);if(raw===""||Number.isInteger(v)&&v>=0&&v<points.length){store.cell=raw===""?null:v;drawLab()}};
    $("field-b").onclick=e=>{const c=hitCell($("field-b"),e);if(c!==null){store.cell=c;drawLab()}};
    $("outlines").onchange=e=>{store.outlines=e.target.checked;drawLab()};
    const setAt=k=>{stop();store.at=indices(ep)[k];drawLab()};
    $("timeline").oninput=e=>setAt(Number(e.target.value));
    $("previous").onclick=()=>setAt(indices(ep).indexOf(store.at)-1);$("next").onclick=()=>setAt(indices(ep).indexOf(store.at)+1);
    $("play").onclick=()=>{if(store.running){stop();drawLab();return}const a=indices(ep);if(a.length<2)return;if(store.at===a.at(-1))store.at=a[0];store.running=true;drawLab();
      store.playHandle=setInterval(()=>{const i=a.indexOf(store.at);if(i===a.length-1)stop();else store.at=a[i+1];drawLab()},325)};
    root.querySelectorAll("[data-policy]").forEach(b=>b.onclick=()=>{store.mask=b.dataset.policy;renderLab();root.querySelector(`[data-policy="${store.mask}"]`).focus({preventScroll:true})});
    if($("learn-type"))$("learn-type").onclick=()=>{store.focus=channelLabel(store.channel);location.hash="network"};
    updateChannelDescription();drawLab();
  }
  function labBelow(s){
    const text=[
      ["Источник","Неподвижный кадр","Последовательные наблюдения приходят из одного видеоролика. Рамка входа может перемещаться вместе с объектом."],
      ["Пространство","721 позиция","Позиция глаза задаёт место в изображении. Для одного и того же места можно посмотреть отклик разных типов клеток."],
      ["Время","Состояние сети","К следующему кадру переходит внутреннее состояние. Поэтому отклик клетки зависит и от предшествующего изображения."]
    ];
    if(s===2)return '<section class="below panel"><div class="section-head"><div><p class="panel-tag">Выбранный тип</p><h2 id="type-title"></h2></div><button id="learn-type">Связи этого типа →</button></div><p id="type-copy" class="bodycopy"></p><p id="cell-study" class="bodycopy tiny"></p></section>';
    return '<div class="below triplet">'+text.map(([k,h,p])=>`<div class="panel"><p class="panel-tag">${k}</p><h3>${h}</h3><p class="bodycopy tiny">${p}</p></div>`).join("")+'</div>';
  }
  function explainType(t){
    if(/^T4[a-d]$/.test(t))return "Один из четырёх типов T4. Эта группа чувствительна к движению светлых границ. Буква различает подтипы, предпочитающие разные направления.";
    if(/^T5[a-d]$/.test(t))return "Один из четырёх типов T5. Эта группа чувствительна к движению тёмных границ. Буква различает подтипы, предпочитающие разные направления.";
    if(/^R[1-8]$/.test(t))return "Фоторецептор. От клеток этой группы начинается путь зрительного сигнала.";
    if(/^L[1-5]$/.test(t))return "Тип клеток первого зрительного слоя. Здесь сигнал фоторецепторов передаётся дальше по нескольким путям.";
    if(/^Mi/.test(t))return "Тип клеток медуллы, одного из зрительных отделов мухи. Схема показывает его записанные входы и выходы.";
    if(/^TmY/.test(t))return "Одна из групп зрительных клеток с разветвляющимися отростками. Её место в модели можно проследить по соседним типам.";
    if(/^Tm/.test(t))return "Тип зрительных клеток, передающих сигнал через медуллу к следующему отделу. Его входы и выходы указаны в схеме модели.";
    return "Тип клеток из схемы связей FlyVis. Его место в сети определяют входящие и исходящие связи.";
  }
  function updateChannelDescription(){if(!$("type-title"))return;const t=channelLabel(store.channel);$("type-title").textContent=t;
    const a=study?.conditions?.find(x=>x.name===t);$("type-copy").textContent=explainType(t)+(a?` В проверке ослабление сигнала этого типа при считывании изменило долю верных ответов на ${signed(a.delta,2)} п. п.`:"")}
  function scoreDetails(ep){
    const b=ep.scores[store.mask==="exclusive"?"exclusive":"whole"];
    const high=Math.max(...b),tie=Math.abs(b[0]-b[1])<=1e-10,pick=tie?"ничья":b[0]===high?"А":"Б", target=ep.correct===0?"А":"Б";
    return `<section class="below panel"><div class="section-head"><div><p class="panel-tag">Решение в кадре ${esc(ep.frame)}</p><h2>Сравнение двух рамок</h2></div><div class="seg" aria-label="Ячейки для вычисления балла"><button data-policy="all" aria-pressed="${store.mask==="all"}">Вся рамка</button><button data-policy="exclusive" aria-pressed="${store.mask==="exclusive"}">Только свои</button></div></div><div class="detail"><div class="scorelist">${b.map((v,i)=>`<div class="scoreline"><span>Рамка ${i?"Б":"А"}</span><meter min="${Math.min(...b,0)-.01}" max="${Math.max(...b,0)+.01}" value="${v}"></meter><strong>${signed(v,5)}</strong></div>`).join("")}</div><div><strong class="answer">Выбор: ${pick}</strong><p class="tiny muted">Рамка продолжения по разметке: ${target}. Больше балл — выше сходство с прошлым образцом.</p></div></div></section>`;
  }
  function renderNetwork(){
    root.innerHTML=head("Архитектура","Связи между клетками","Выберите тип и проследите один шаг сигнала: от кого он получает вход и кому передаёт его дальше.",[[graphTypes.length,"типов клеток"],[graph.outputs?.length||channels.length,"доступны в картах"]]);
    if(!graphTypes.length){root.insertAdjacentHTML("beforeend",'<div class="panel">Схема связей отсутствует в этом комплекте.</div>');return}
    const t=store.focus||graphTypes[0], ins=graph.edges.filter(e=>e[1]===t).sort((a,b)=>b[2]-a[2]), outs=graph.edges.filter(e=>e[0]===t).sort((a,b)=>b[2]-a[2]);
    const chip=(name,v)=>`<button class="chip" data-focus="${esc(name)}" title="${esc(name)} · контактов в шаблоне: ${Number(v)}">${esc(name)}</button>`;
    const self=ins.some(e=>e[0]===t);
    const pointLinks=(list,k)=>list.length?list.map(e=>chip(e[k],e[2])).join(""):'<span class="tiny muted">В файле модели нет связей этого направления.</span>';
    const neighbors=(list,k,x)=>{
      const shown=list.filter(e=>e[k]!==t).slice(0,6);
      return shown.map((e,i)=>({x,y:shown.length===1?157:65+184*(i/Math.max(1,shown.length-1)),s:e[k],v:e[2]}));
    };
    const a=neighbors(ins,0,92),b=neighbors(outs,1,508);
    const curves=(rows,left)=>rows.map(r=>{
      const start=left?[r.x+48,r.y]:[355,157], end=left?[245,157]:[r.x-48,r.y], mid=(start[0]+end[0])/2;
      return `<path d="M${start} C${mid},${start[1]} ${mid},${end[1]} ${end}" fill="none" stroke="${left?"#c88768":"#82a398"}" stroke-width="1.8" marker-end="url(#arrow${left?1:2})"/>`;
    }).join("");
    const node=r=>`<g tabindex="0" role="button" aria-label="Показать тип ${esc(r.s)}" class="network-node" data-focus="${esc(r.s)}"><rect x="${r.x-47}" y="${r.y-15}" width="94" height="30" rx="6" fill="white" stroke="#bcb4a5"/><text x="${r.x}" y="${r.y+5}" text-anchor="middle" font-family="PlexMono,monospace" font-size="13" fill="#302b25">${esc(r.s)}</text></g>`;
    root.insertAdjacentHTML("beforeend",`<div class="scheme"><section class="network-card"><div class="section-head"><h2>Соседи типа ${esc(t)}</h2><label class="field">Тип клеток<select id="network-type">${selectItems(graphTypes.map(s=>[s,s]),t)}</select></label></div>
    <div class="network-canvas"><svg viewBox="0 0 600 290" role="group" aria-label="Направленные связи выбранного типа"><defs><marker id="arrow1" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#c88768"/></marker><marker id="arrow2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#82a398"/></marker></defs><g font-size="11" font-family="PlexMono,monospace" fill="#786f60" text-anchor="middle"><text x="92" y="29">ПОЛУЧАЕТ ОТ</text><text x="300" y="29">ВЫБРАННЫЙ ТИП</text><text x="508" y="29">ПЕРЕДАЁТ В</text></g>
    ${curves(a,true)}${curves(b,false)}${a.map(node).join("")}${b.map(node).join("")}<rect x="245" y="126" width="110" height="62" rx="10" fill="#315f56"/><text x="300" y="165" text-anchor="middle" font-family="PlexMono,monospace" font-size="23" fill="white">${esc(t)}</text></svg></div>
    <p class="bodycopy tiny">До шести соседних типов с каждой стороны, в порядке числа контактов в шаблоне связей. Стрелки показывают направление передачи сигнала.</p></section>
    <aside class="network-card story"><p class="panel-tag">Тип ${esc(t)}</p><h2>${esc(t)}</h2><p>${explainType(t)}</p><div class="chips-label">Все входы · ${ins.length}</div><div class="connection-list">${pointLinks(ins,0)}</div><div class="chips-label">Все выходы · ${outs.length}</div><div class="connection-list">${pointLinks(outs,1)}</div><p class="tiny">${self?"В схеме есть и связь этого типа с собственным типом.":""}</p>${channels.includes(t)?'<button id="open-map">Открыть карту этого типа →</button>':'<p class="tiny">Для этого внутреннего типа здесь доступна схема связей. Архив карт содержит выходные типы.</p>'}</aside></div>
    <div class="triplet"><div class="panel"><p class="panel-tag">Тип</p><h3>Что за клетка</h3><p class="bodycopy tiny">T4 и T5 — семейства. T4a и T4b — два разных типа одного семейства. Каждый тип может встречаться в разных местах глаза.</p></div><div class="panel"><p class="panel-tag">Позиция</p><h3>Где в поле зрения</h3><p class="bodycopy tiny">У карт одна сетка из 721 позиции. Переключение типа оставляет координату на месте и показывает другой сигнал.</p></div><div class="panel"><p class="panel-tag">Считывание</p><h3>Как получается решение</h3><p class="bodycopy tiny">После работы сети мы выбираем, какие карты войдут в балл рамки. Схема выше описывает связи внутри сети.</p></div></div>`);
    root.querySelector(".story").insertAdjacentHTML("beforeend",coreInfo(t));
    const choose=name=>{if(!graphTypes.includes(name))return;store.focus=name;renderNetwork();$("network-type").focus({preventScroll:true})};
    $("network-type").onchange=e=>choose(e.target.value);
    root.querySelectorAll("[data-focus]").forEach(el=>{el.onclick=()=>choose(el.dataset.focus);if(el.tagName.toLowerCase()==="g")el.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose(el.dataset.focus)}}});
    if($("open-map"))$("open-map").onclick=()=>{store.channel=channels.indexOf(t);store.stage=2;store.at=getEp().finalIndex;location.hash="lab"};
  }
  function coreInfo(t){
    const x=core?.types?.find(v=>v.name===t);
    if(!core || !x)return `<div class="citation"><strong>Внутри сети</strong><p>${esc(core?.message||"Для этого типа численный результат вмешательства во внутреннее состояние пока отсутствует.")}</p></div>`;
    const ci=Array.isArray(x.adjustedCI)?` Интервал с поправкой на все типы: от ${signed(x.adjustedCI[0],2)} до ${signed(x.adjustedCI[1],2)} п. п.`:"";
    return `<div class="citation"><strong>Внутри сети · ${esc(t)}</strong><p>Перед последним кадром состояние этого типа умножили на 0,9. Доля верных ответов изменилась на ${signed(x.delta,2)} п. п. по ${Number(core.n)} парам. Решение изменилось в ${x.changes} случаях.${ci}</p></div>`;
  }
  function renderStudy(){
    root.innerHTML=head("Эксперимент","Из чего складывается выбор","На одних и тех же парах рамок меняем вклад одного типа и смотрим, как часто меняется решение.");
    if(!study){root.insertAdjacentHTML("beforeend",'<div class="panel">В этой версии таблица эксперимента отсутствует.</div>');return}
    const b=study.baseline,t=study.t4t5,compare=study.comparison,n=study.n;
    const ciText=x=>Array.isArray(x)?`[${signed(x[0],2)}; ${signed(x[1],2)}]`:"—";
    const rows=study.conditions.slice();
    if(store.sort==="change")rows.sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));
    if(store.sort==="name")rows.sort((a,b)=>a.name.localeCompare(b.name,"en",{numeric:true}));
    root.insertAdjacentHTML("beforeend",`<div class="statrow"><div class="stat"><b>${n}</b>размеченные пары рамок</div><div class="stat"><b>${t.credit} / ${n}</b>верных · 8 каналов T4/T5</div><div class="stat"><b>${b.credit} / ${n}</b>верных · 34 прямых канала</div></div>
    <div class="panel study-hero"><p class="panel-tag">Целые рамки · штатная FlyVis</p><h2>Какие карты помогли найти цель</h2><p class="bodycopy">Переход с восьми на 34 канала изменил долю верных ответов на <strong>${signed(compare.delta,2)} п. п.</strong> 95%-й парный интервал по видео — ${ciText(compare.ci)} п. п. Веса заданы заранее. Пары происходят из исторической выборки, по которой уже проводились разборы.</p></div>
    <div class="results-card"><div class="section-head"><div><p class="panel-tag">По одному типу</p><h2>Если ослабить сигнал на 10%</h2></div><label class="field">Порядок<select id="study-sort">${selectItems([["order","Как в модели"],["change","Сначала наибольшее изменение"],["name","По названию"]],store.sort)}</select></label></div>
    <p class="bodycopy">Остальные 33 канала сохраняют прежний вклад. Все значения в таблице сравниваются с исходным вариантом из 34 каналов. «Исправлено» и «потеряно» — число решений, поменявших результат. Δ — изменение доли верных ответов в процентных пунктах.</p>
    <div class="table-wrap"><table class="data-table"><thead><tr><th>Тип клеток</th><th>Верных из ${n}</th><th>Исправлено</th><th>Потеряно</th><th>Δ, п. п.</th><th>95% интервал, п. п.</th></tr></thead><tbody>
    ${rows.map(r=>`<tr><td><button data-celltype="${esc(r.name)}" class="chip">${esc(r.name)} ↗</button></td><td>${fmt(r.credit,Number.isInteger(r.credit)?0:1)}</td><td>${r.wrongToCorrect}</td><td>${r.correctToWrong}</td><td>${signed(r.delta,2)}</td><td>${ciText(r.ci)}</td></tr>`).join("")}</tbody></table></div>
    <p class="citation">Индивидуальные интервалы приведены для описания. После поправки на проверку нескольких типов подтвердилось эффектов: ${Number(study.familySignificant)||0}. Проверка относится к чтению сохранённой активности в момент выбора рамки.</p></div>`);
    const items=core?.types||[];
    root.querySelector(".results-card").insertAdjacentHTML("beforebegin",`<section class="below panel"><p class="panel-tag">Перед последним кадром</p><div class="section-head"><h2>Вмешательство во внутреннее состояние</h2>${items.length?`<label class="field">Тип клеток<select id="core-type">${selectItems(items.map(x=>[x.name,x.name]),items.some(x=>x.name==="T4a")?"T4a":items[0].name)}</select></label>`:""}</div><p class="bodycopy">${esc(core?.message||"В этом комплекте пока доступны результаты вмешательств при считывании.")}</p>${items.length?'<div id="core-result"></div>':""}</section>`);
    if($("core-type")){const sync=()=>{$("core-result").innerHTML=coreInfo($("core-type").value)};$("core-type").onchange=sync;sync()}
    $("study-sort").onchange=e=>{store.sort=e.target.value;renderStudy();$("study-sort").focus()};
    root.querySelectorAll("[data-celltype]").forEach(b=>b.onclick=()=>{store.focus=b.dataset.celltype;location.hash="network"});
  }
  function loopback(){return ["http:","https:"].includes(location.protocol)&&["localhost","127.0.0.1","[::1]"].includes(location.hostname)}
  async function checkServer(){
    if(!loopback())return false;
    try{const r=await fetch("/api/status",{cache:"no-store",signal:AbortSignal.timeout(5000)});if(!r.ok)return false;return (await r.json()).available===true}catch{return false}
  }
  function renderUpload(){
    root.innerHTML=head("Локальный расчёт","Свой кадр в FlyVis","Откройте изображение или начало ролика и проследите, как на него отвечают клетки модели.");
    root.insertAdjacentHTML("beforeend",`<section class="panel"><p class="panel-tag">Вход модели</p><div class="uploadbox"><label for="stimulus"><strong>Изображение или видео</strong></label><input id="stimulus" type="file" accept="image/*,video/*"><p class="tiny muted">Из центра кадра берётся квадрат 256 × 256. Для видео обрабатываются до 12 кадров с шагом 1/30 секунды. Для изображения — один кадр.</p><div class="tools"><button id="process-file" class="primary" disabled>Получить отклик</button></div></div>
    <p id="upload-message" class="feedback" role="status" aria-live="polite"></p><div id="file-visual"></div></section>`);
    const fileInput=$("stimulus");
    const refresh=()=>{
      if(!$("process-file"))return;
      $("process-file").disabled=!store.mediaAvailable||!store.file||!!store.busy;fileInput.disabled=!!store.busy||store.mediaAvailable===false;
      $("upload-message").textContent=store.fileMessage || (store.mediaAvailable===null?"Проверяем локальный вычислитель…":store.mediaAvailable?"Вычислитель доступен на этом компьютере.":"Расчёт своего файла доступен через START.cmd в папке атласа. Откройте адрес, который он выведет. Сохранённые эпизоды доступны и без сервера.");
    };
    fileInput.onchange=()=>{store.file=fileInput.files[0]||null;store.fileMessage=store.file?store.file.name:"";refresh()};
    $("process-file").onclick=()=>runFile(refresh);
    if(!loopback()){store.mediaAvailable=false;refresh()}
    else{refresh();checkServer().then(ok=>{store.mediaAvailable=ok;refresh()})}
    renderFileVisual();
  }
  function once(media,ev){
    return new Promise((resolve,reject)=>{const timer=setTimeout(()=>done(false),10000);
      function done(ok){clearTimeout(timer);media.removeEventListener(ev,success);media.removeEventListener("error",fail);ok?resolve():reject(new Error("Браузеру не удалось прочитать этот медиафайл."))}
      function success(){done(true)}function fail(){done(false)}
      media.addEventListener(ev,success,{once:true});media.addEventListener("error",fail,{once:true})});
  }
  async function sampleFile(file){
    const video=file.type.startsWith("video/")||/\.(mp4|webm|mov|ogv|m4v)$/i.test(file.name);
    const media=video?document.createElement("video"):new Image(), url=URL.createObjectURL(file);
    const previews=[],frames=[];
    try{
      if(video){media.muted=true;media.playsInline=true;media.preload="auto"}
      const loaded=once(media,video?"loadeddata":"load");media.src=url;await loaded;
      const w=video?media.videoWidth:media.naturalWidth,h=video?media.videoHeight:media.naturalHeight,s=Math.min(w,h);
      if(!s)throw new Error("Не удалось определить размеры кадра.");
      const canvas=document.createElement("canvas");canvas.width=canvas.height=256;
      const ctx=canvas.getContext("2d",{willReadFrequently:true});
      const total=video?Math.min(12,Math.max(1,Math.ceil((Number.isFinite(media.duration)?media.duration:.4)*30))):1;
      for(let j=0;j<total;j++){
        if(video&&j){const moved=once(media,"seeked");media.currentTime=j/30;await moved}
        ctx.drawImage(media,(w-s)/2,(h-s)/2,s,s,0,0,256,256);
        previews.push(canvas.toDataURL("image/jpeg",.85));
        const rgba=ctx.getImageData(0,0,256,256).data,gray=new Uint8Array(256*256);
        for(let i=0;i<gray.length;i++)gray[i]=Math.round(.299*rgba[i*4]+.587*rgba[i*4+1]+.114*rgba[i*4+2]);
        let str="";for(let k=0;k<gray.length;k+=4096)str+=String.fromCharCode(...gray.subarray(k,k+4096));
        frames.push(btoa(str));
      }
      return {frames,previews,kind:video?"video":"image"};
    }finally{if(video){media.removeAttribute("src");media.load()}URL.revokeObjectURL(url)}
  }
  async function runFile(refresh){
    if(store.busy||!store.mediaAvailable||!store.file||!loopback())return;
    store.busy=true;store.fileMessage="Подготавливаем вход…";refresh();
    try{
      const a=await sampleFile(store.file);store.fileMessage=`Вычисляем отклик FlyVis для ${a.frames.length} кадр(ов)…`;refresh();
      const r=await fetch("/api/respond",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({frames:a.frames,kind:a.kind})});
      let result;try{result=await r.json()}catch{throw new Error("Вычислитель вернул ответ в неизвестном формате.")}
      if(!r.ok||result.error)throw new Error(result.error||`Ошибка вычислителя ${r.status}`);
      if(result.maps?.length!==a.frames.length||!Array.isArray(result.channels)||result.maps.some(m=>m.length!==result.channels.length||m.some(x=>x.length!==721)))throw new Error("Размер карт в ответе не совпал с входом.");
      store.fileResult=result;store.fileFrames=a.previews;store.fileAt=0;store.fileMessage="Отклик получен.";renderFileVisual();
    }catch(e){store.fileMessage=e.message||String(e)}
    finally{store.busy=false;refresh()}
  }
  function renderFileVisual(){
    const box=$("file-visual"),data=store.fileResult;if(!box||!data)return;
    const i=store.fileAt, ch=store.fileChannel;const bound=mapScale(data.maps[i],ch,data,"file");
    box.innerHTML=`<div class="figure-pair upload-figure"><figure class="figure"><figcaption class="fighead">Область входа · 256 × 256</figcaption><div class="picture"><img class="preview-frame" alt="Выбранный кадр" src="${store.fileFrames[i]}"></div></figure>${fig("custom-map","Отклик клеток",esc(data.channels[ch]),"Исходная активность",`<span class="legend">−${fmt(bound,2)}<i class="gradient"></i>+${fmt(bound,2)}</span>`)}</div><div class="toolbar"><label class="field">Тип клеток<select id="custom-type">${choices(data.channels,ch)}</select></label><label class="field">Кадр<select id="custom-time">${selectItems(data.maps.map((m,j)=>[j,`${j+1} из ${data.maps.length}`]),i)}</select></label></div><p class="bodycopy tiny">${esc(data.note||"Цвет показывает численную активность клеток модели на выбранном кадре.")}</p>`;
    drawMap($("custom-map"),data.maps[i][ch],{bound});
    $("custom-type").onchange=e=>{store.fileChannel=Number(e.target.value);renderFileVisual();$("custom-type").focus()};
    $("custom-time").onchange=e=>{store.fileAt=Number(e.target.value);renderFileVisual();$("custom-time").focus()};
  }
  function route(){const name=location.hash.replace(/^#/,"");return ["lab","network","study","upload"].includes(name)?name:"lab"}
  function navigate(){
    stop();const page=route();
    document.querySelectorAll(".nav a").forEach(a=>a.hash==="#"+page?a.setAttribute("aria-current","page"):a.removeAttribute("aria-current"));
    ({lab:renderLab,network:renderNetwork,study:renderStudy,upload:renderUpload})[page]();
    document.title=(document.querySelector("h1")?.textContent||"Лаборатория зрения")+" · Атлас FlyVis";
  }
  store.at=getEp()?.finalIndex??0;
  window.addEventListener("hashchange",()=>{navigate();window.scrollTo(0,0);root.focus({preventScroll:true})});
  let resizeTicket=0;window.addEventListener("resize",()=>{cancelAnimationFrame(resizeTicket);resizeTicket=requestAnimationFrame(()=>{if(route()==="lab")drawLab();else if(route()==="upload")renderFileVisual()})});
  document.addEventListener("visibilitychange",()=>{if(document.hidden&&store.running){stop();drawLab()}});
  navigate();
})();
