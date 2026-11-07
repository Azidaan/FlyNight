(() => {
  "use strict";
  const source=window.FLY_EPISODES;
  const episode=source?.episodes?.find(e=>e.video==="0199e7fc-0b6d1a35"&&e.frame===212);
  const frame=episode?.frames?.[episode.finalIndex];
  const available=!!(frame?.image&&frame.eye?.length===721&&frame.maps&&episode.scores?.whole&&episode.scores?.exclusive);
  const channels=source?.channels||[],points=source?.points||[];
  const names=["T4a","T4b","T4c","T4d","T5a","T5b","T5c","T5d"].filter(t=>channels.includes(t));
  let policy="all",channel=names[0]||"T4a";
  const stages=[
    {id:"video",short:"Видео",title:"Исходный кадр"},
    {id:"crop",short:"Область",title:"Участок для зрительного модуля"},
    {id:"eye",short:"Глаз",title:"Вход в 721 позицию"},
    {id:"network",short:"Сеть",title:"Сеть с внутренним состоянием"},
    {id:"maps",short:"T4 / T5",title:"Пространственные карты движения"},
    {id:"compare",short:"Рамки",title:"Сравнение двух продолжений"},
    {id:"answer",short:"Выбор",title:"Какой объект выбран"}
  ];
  const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const fmt=(n,d=0)=>Number(n).toLocaleString("ru-RU",{minimumFractionDigits:d,maximumFractionDigits:d});
  const letter=i=>i===0?"А":"Б";
  const signed=(n,d=6)=>(n>=0?"+":"")+fmt(n,d);
  const activeScore=()=>episode.scores[policy==="all"?"whole":"exclusive"];
  const winner=s=>Math.abs(s[0]-s[1])<1e-10?null:s[0]>s[1]?0:1;
  const selected=(v,key)=>`aria-pressed="${v===key}"`;
  const missing=()=>'<div class="bm-note">В этом комплекте нет сохранённых данных кадра 212. Откройте полную локальную копию атласа.</div>';
  const controls=()=>`<div class="bm-dir"><button data-frame-policy="all" ${selected(policy,"all")}>Вся рамка</button><button data-frame-policy="exclusive" ${selected(policy,"exclusive")}>Только свои</button></div>`;
  const channelControls=()=>`<div class="bm-chips">${names.map(n=>`<button class="bm-chip" data-frame-channel="${n}" ${selected(channel,n)}>${n}</button>`).join("")}</div>`;
  function panel(index){
    if(!available)return missing();
    const step=stages[index]||stages[0],c=frame.crop,[W,H]=episode.originalSize;
    const a=new Set(episode.masks[0]),b=new Set(episode.masks[1]),common=[...a].filter(v=>b.has(v)).length;
    const score=activeScore(),pick=winner(score),correct=episode.correct;let body;
    switch(step.id){
      case "video":body=`<p>Кадр <strong>${episode.frame}</strong> из видео <strong>${esc(episode.video)}</strong>. Исходный размер — ${W} × ${H}.</p><div class="bm-note">В архиве эпизода ${episode.frames.length} входных шагов, от кадра ${episode.frames[0].frame} до ${frame.frame}. Это реальные входы; некоторые номера пропущены.</div>`;break;
      case "crop":body=`<p>В зрительный модуль подан квадрат от (${c[0]}, ${c[1]}) до (${c[2]}, ${c[3]}): <strong>${c[2]-c[0]} × ${c[3]-c[1]}</strong> пикселя исходного кадра.</p><p>Слева увеличена именно эта область. Далее показано её представление на сетке глаза.</p>`;break;
      case "eye":body=`<p>Составной глаз преобразует область в <strong>${frame.eye.length}</strong> значений яркости. Координаты и значения взяты из архива.</p><p>Минимальная яркость — ${fmt(Math.min(...frame.eye),4)}, максимальная — ${fmt(Math.max(...frame.eye),4)}.</p><div class="bm-note">Это пространственный вход. Цвет обозначает яркость, а не ответ нейронов.</div>`;break;
      case "network":body=`<p>В используемом коннектоме <strong>${window.FLY_CONNECTOME?.types?.length||65} типов</strong> клеток. На кадр влияют и вход глаза, и сохранённое внутреннее состояние.</p><p>Клетки обновляются совместно: здесь нет последовательного запуска «сначала L1, затем Mi1, затем T4».</p><div class="bm-note">На этом этапе показано устройство сети с обратными связями. Активность всех внутренних типов для кадра 212 в этом комплекте не сохранена.</div>`;break;
      case "maps":body=`<p>Из доступных карт смотрим восемь направленных типов T4/T5. Выберите тип — слева изменится измеренная карта.</p>${channelControls()}<div class="bm-smallhead">Выбран ${channel}</div><p>Карта содержит ${frame.maps[channels.indexOf(channel)].length} позиций. Цвет показывает нормированный отклик относительно среднего канала в исходном наборе.</p><div class="bm-note">Карты сохранены для кадров ${episode.availableMapFrames.join(", ")}. Между ними отклик не дорисовывается.</div>`;break;
      case "compare":body=`<p>У двух допустимых рамок продолжения разные области чтения: А — <strong>${a.size}</strong> позиций глаза, Б — <strong>${b.size}</strong>, общих — <strong>${common}</strong>.</p>${controls()}<div class="bm-smallhead">Чтение карт</div><p>${policy==="all"?"Вся рамка включает и общую часть.":"Общие позиции исключены: у А осталось "+(a.size-common)+", у Б — "+(b.size-common)+"."}</p><p>Баллы из проверенного архива: А ${signed(score[0])}, Б ${signed(score[1])}.</p>`;break;
      default:body=`<p>По варианту «${policy==="all"?"Вся рамка":"Только свои"}» выше балл у ${pick===null?"обеих рамок поровну":"рамки <strong>"+letter(pick)+"</strong>"}.</p>${controls()}<div class="bm-smallhead">Проверка по разметке</div><p>Верное продолжение — рамка <strong>${letter(correct)}</strong>. ${pick===correct?"В этом варианте выбор верный.":pick===null?"Баллы не различили две рамки.":"В этом варианте выбран другой объект."}</p><div class="bm-note">Это один диагностический выбор. Разметка служит для оценки результата и не передаётся зрительному модулю.</div>`;
    }
    return `<div class="bm-eyebrow">Реальный эпизод · шаг ${index+1} / ${stages.length}</div><h2>${step.title}</h2>${body}`;
  }
  function svg(body,title,subtitle){
    return `<svg class="fp-graphic" viewBox="0 0 760 440" role="img" aria-label="${esc(title)}" style="display:block;width:100%;height:auto;font-family:Plex,sans-serif"><rect width="760" height="440" fill="#fbfaf6"/><text x="30" y="31" fill="#473a2b" font-size="18" font-weight="600">${esc(title)}</text><text x="30" y="52" fill="#7d6d59" font-size="11">${esc(subtitle)}</text>${body}</svg>`;
  }
  function imageSvg(cropped=false){
    const [W,H]=episode.originalSize,c=frame.crop;
    const [vx,vy,vw,vh]=cropped?[c[0],c[1],c[2]-c[0],c[3]-c[1]]:[0,0,W,H];
    const [x,y,w,h]=cropped?[206,77,310,310]:[26,74,708,330];
    const rect=cropped?"":`<rect x="${c[0]}" y="${c[1]}" width="${c[2]-c[0]}" height="${c[3]-c[1]}" fill="none" stroke="#d97757" stroke-width="4"/>`;
    return `<svg x="${x}" y="${y}" width="${w}" height="${h}" viewBox="${vx} ${vy} ${vw} ${vh}" preserveAspectRatio="xMidYMid meet"><image href="${esc(frame.image)}" x="0" y="0" width="${W}" height="${H}"/>${rect}</svg>`;
  }
  function candidates(){
    const boxes=episode.boxes,[W,H]=episode.originalSize;
    const x=Math.min(...boxes.map(b=>b[0]))-8,y=Math.min(...boxes.map(b=>b[1]))-14;
    const w=Math.max(...boxes.map(b=>b[2]))-x+8,h=Math.max(...boxes.map(b=>b[3]))-y+8;
    return `<svg x="30" y="143" width="200" height="155" viewBox="${x} ${y} ${w} ${h}" preserveAspectRatio="xMidYMid meet"><image href="${esc(frame.image)}" width="${W}" height="${H}"/>${boxes.map((b,i)=>`<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="none" stroke="${i?"#4fe4bb":"#ffb27e"}" stroke-width="1.6"/><text x="${b[0]+2}" y="${b[1]-2}" font-size="8" font-weight="600" fill="${i?"#4fe4bb":"#ffb27e"}" stroke="#302b25" stroke-width=".35" paint-order="stroke">${letter(i)}</text>`).join("")}</svg><text x="130" y="326" text-anchor="middle" font-size="12" fill="#7b6954">Реальные рамки в кадре</text>`;
  }
  const hex=(x,y,r)=>[0,1,2,3,4,5].map(i=>`${x+Math.cos((60*i+30)*Math.PI/180)*r},${y+Math.sin((60*i+30)*Math.PI/180)*r}`).join(" ");
  function mapGrid(values,x=220,y=76,size=310,kind="eye",mask=null){
    const bound=kind==="eye"?1:Math.max(...values.map(Math.abs),1e-9);
    const a=new Set(episode.masks[0]),b=new Set(episode.masks[1]),selected=mask===null?null:(mask===0?a:b),other=mask===0?b:a;
    const polys=points.map(([px,py],i)=>{
      const v=(values[i]||0)/bound;let shade;
      if(kind==="eye")shade=`rgb(${Math.round(v*255)},${Math.round(v*255)},${Math.round(v*255)})`;
      else if(v>=0)shade=`rgb(${Math.round(242-70*v)},${Math.round(236-155*v)},${Math.round(218-167*v)})`;
      else shade=`rgb(${Math.round(242+202*v)},${Math.round(236+125*v)},${Math.round(218+109*v)})`;
      const on=selected?.has(i)&&!(policy==="exclusive"&&other.has(i));
      return `<polygon points="${hex(x+px*size,y+py*size,size*.0138)}" fill="${shade}"${selected?` opacity="${on?1:.14}" stroke="${on?(mask===0?"#ad6341":"#397d70"):"none"}" stroke-width=".6"`:""}/>`;
    }).join("");return `<g>${polys}</g>`;
  }
  function netGraphic(){
    const labs=[["Вход глаза",104,217],["Состояние сети",366,217],["Карты отклика",641,217]];
    return `<defs><marker id="fp-core-arrow" viewBox="0 0 7 7" refX="6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#ad6746"/></marker></defs><path d="M180 217H260M475 217H566M410 163V116H320V163" fill="none" stroke="#ad6746" stroke-width="2" marker-end="url(#fp-core-arrow)"/>${labs.map(([t,x,y],i)=>`<rect x="${x-(i===1?106:76)}" y="${y-52}" width="${i===1?212:152}" height="104" rx="12" fill="${i===1?"#f1e3d1":"#fffdf7"}" stroke="#bdac93"/><text x="${x}" y="${y-4}" text-anchor="middle" font-size="17" font-weight="600" fill="#584532">${t}</text><text x="${x}" y="${y+20}" text-anchor="middle" font-size="12" fill="#7b6954">${i===0?"721 значение":i===1?"65 типов · совместное обновление":"34 сохранённые карты"}</text>`).join("")}<text x="366" y="99" text-anchor="middle" font-size="12" fill="#8b5d43">обратные связи</text><text x="380" y="330" text-anchor="middle" font-size="13" fill="#7b6a55">Внутреннее состояние перенесено с предыдущих входных шагов</text>`;
  }
  function graphic(index){
    if(!available)return missing();
    const step=stages[index]||stages[0],score=activeScore(),pick=winner(score);
    switch(step.id){
      case "video":return svg(imageSvg(),`Видео · кадр ${frame.frame}`,episode.video+" · прямоугольник отмечает вход зрительного модуля");
      case "crop":return svg(imageSvg(true),`Участок ${frame.crop[2]-frame.crop[0]} × ${frame.crop[3]-frame.crop[1]}`,"Область реального кадра, увеличенная для показа");
      case "eye":return svg(mapGrid(frame.eye),"Вход составного глаза","Реальные 721 значение яркости · чёрный 0, белый 1");
      case "network":return svg(netGraphic(),"Кадр и предыстория встречаются в сети","Структурная схема; цвет не обозначает измеренную активность внутренних типов");
      case "maps":return svg(mapGrid(frame.maps[channels.indexOf(channel)],220,76,310,"map"),`Карта ${channel} · кадр 212`,"Нормированный отклик · зелёный ниже среднего, терракота выше");
      case "compare":return svg(candidates()+mapGrid(frame.maps[channels.indexOf(channel)],270,128,200,"map",0)+mapGrid(frame.maps[channels.indexOf(channel)],510,128,200,"map",1)+`<text x="370" y="105" text-anchor="middle" font-size="19" fill="#a65e3d">Рамка А</text><text x="610" y="105" text-anchor="middle" font-size="19" fill="#347a6d">Рамка Б</text><text x="490" y="365" text-anchor="middle" font-size="12" fill="#7b6954">Позиции глаза, используемые для чтения каждой рамки</text>`,`Один канал ${channel}, две области чтения`,policy==="all"?"Общие позиции входят в обе рамки":"Общие позиции исключены из обеих рамок");
      default:{
        const correct=pick===episode.correct;
        return svg(`<text x="380" y="122" text-anchor="middle" font-size="20" fill="#584a38">Вариант: ${policy==="all"?"вся рамка":"только свои позиции"}</text>${score.map((v,i)=>`<rect x="125" y="${161+i*88}" width="510" height="65" rx="9" fill="${i===pick?"#eee0cc":"#fffdf8"}" stroke="${i===pick?"#ac6a43":"#d7cbb9"}"/><text x="146" y="${200+i*88}" font-size="21" fill="#554330">Рамка ${letter(i)}</text><text x="613" y="${200+i*88}" text-anchor="end" font-size="21" fill="#554330">${signed(v)}</text>`).join("")}<text x="380" y="365" text-anchor="middle" font-size="21" font-weight="600" fill="${correct?"#357c68":"#aa5539"}">Выбор: ${pick===null?"ничья":"рамка "+letter(pick)} · ${correct?"верно":"неверно"}</text><text x="380" y="393" text-anchor="middle" font-size="12" fill="#756450">Верная рамка по разметке: ${letter(episode.correct)}</text>`,"Сохранённые баллы и решение","Выше балл — сильнее сходство с прошлым образцом");
      }
    }
  }
  window.FlyFramePath={
    available,stages,panel,graphic,
    setPolicy(value){if(value==="all"||value==="exclusive")policy=value},
    setChannel(value){if(names.includes(value))channel=value},
    get policy(){return policy},get channel(){return channel},
    get info(){return available?{video:episode.video,frame:episode.frame,uid:episode.uid,correct:episode.correct,scores:episode.scores,steps:episode.frames.map(f=>f.frame),maps:episode.availableMapFrames}:null}
  };
})();
