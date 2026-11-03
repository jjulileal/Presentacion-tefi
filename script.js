const $=s=>document.querySelector(s);

/* ---------- Inicio: tarjetas ---------- */
const INFO={
obj:['Objetivos',`<p><strong>General:</strong> concientizar a las personas sobre las consecuencias del uso erróneo de los modelos de lenguaje actuales (LLM).</p><ul><li>Informar las consecuencias en la salud mental del uso desmedido.</li><li>Explicar los riesgos de usarla para relaciones y situaciones personales.</li><li>Analizar los mecanismos de entrenamiento (RLHF y reward hacking) que generan respuestas complacientes y sicofancia en lugar de verídicas.</li></ul><p><strong>Hipótesis:</strong> ¿la IA multiplica o perjudica?</p>`],
llm:['LLMs',`<p>Los modelos de lenguaje son modelos de aprendizaje profundo muy grandes, preentrenados con enormes cantidades de datos. Se apoyan en un transformador: un conjunto de redes neuronales con codificador y decodificador que extraen el significado del texto y las relaciones entre palabras (Amazon Web Services, 2023).</p><p>No leen palabras como las personas: las convierten en vectores dentro de un espacio vectorial. Luego del preentrenamiento vienen el ajuste fino y el aprendizaje por refuerzo (RLHF y RLAIF), que moldean cómo responden.</p>`],
prob:['Problemáticas',`<ul><li><strong>Pérdida de autonomía y de pensamiento crítico:</strong> se confía más en las sugerencias de un algoritmo que en el propio razonamiento.</li><li><strong>Aislamiento social y afectación emocional:</strong> el chatbot reemplaza el contacto humano, y eso debilita las habilidades interpersonales y la empatía.</li><li><strong>Fatiga cognitiva y déficit de atención:</strong> buscar respuestas inmediatas y fragmentadas dificulta concentrarse en tareas complejas.</li></ul><p>(Rojas Marín et al., 2024)</p>`],
cau:['Causas',`<ul><li><strong>RLHF:</strong> las evaluaciones humanas premian fluidez, coherencia, utilidad y acuerdo, pero no distinguen entre lo convincente y lo correcto (San Miguel et al., 2026).</li><li><strong>Reward hacking:</strong> el modelo optimiza la calidad percibida en lugar de la corrección factual.</li><li><strong>Sicofancia:</strong> tendencia a alinearse con las creencias del usuario en vez de corregirlas.</li><li><strong>Uso sin criterio:</strong> falta de capacitación, trabajar sin conocer el tema y dependencia por la inmediatez.</li></ul>`],
rec:['Recomendaciones',`<ul><li>No rechazar la IA: bien usada, potencia el razonamiento humano.</li><li>Verificar las respuestas en fuentes confiables antes de darlas por verdaderas.</li><li>Conservar el criterio propio y el pensamiento crítico.</li><li>Capacitarse para usarla: conocer el tema y preguntar mejor, en lugar de buscar solo validación.</li><li>No reemplazar el contacto humano por el chatbot.</li></ul>`]};
const orbit=$('.orbit'),cards=[...document.querySelectorAll('.card')];let lastW=innerWidth;
cards.forEach(c=>c.querySelector('.body').innerHTML=INFO[c.dataset.k][1]);
const rel=c=>{const o=orbit.getBoundingClientRect(),r=c.getBoundingClientRect();return{l:r.left-o.left,t:r.top-o.top,w:r.width,h:r.height}};
const setBox=(c,b)=>{c.style.left=b.l+'px';c.style.top=b.t+'px';c.style.width=b.w+'px';c.style.height=typeof b.h==='number'?b.h+'px':b.h};
function openCard(c){
  cards.forEach(x=>x!==c&&closeCard(x));clearTimeout(c._t);
  const z=rel(c),d=c.dataset.d,o=orbit.getBoundingClientRect(),W=Math.min(d==='up'?460:340,innerWidth-24);
  c._z=z;c.classList.remove('closing');c.classList.add('anim','meas','open');
  setBox(c,{l:z.l,t:z.t,w:W,h:'auto'});const H=c.offsetHeight;       /* mide la altura natural de la card abierta */
  let l=d==='right'?z.l:d==='left'?z.l+z.w-W:z.l+z.w/2-W/2;           /* el borde pegado al botón queda fijo */
  l=Math.max(8-o.left,Math.min(l,innerWidth-8-o.left-W));
  const t=d==='up'?z.t+z.h-H:z.t;
  setBox(c,z);c.offsetWidth;c.classList.remove('meas');                /* vuelve al tamaño del botón y anima hasta la card */
  setBox(c,{l,t,w:W,h:H});
  orbit.style.marginTop=(t<0?12-t+8:12)+'px';                          /* hace lugar arriba si la card crece hacia arriba */
  c.querySelector('.ttl').setAttribute('aria-expanded','true')}
function closeCard(c){
  if(!c.classList.contains('open'))return;
  c.classList.remove('open');c.classList.add('closing');c.querySelector('.ttl').setAttribute('aria-expanded','false');
  setBox(c,c._z);orbit.style.marginTop='12px';
  c._t=setTimeout(()=>{c.classList.remove('closing','anim');['left','top','width','height'].forEach(p=>c.style[p]='')},520)}
cards.forEach(c=>c.querySelector('.ttl').onclick=()=>c.classList.contains('open')?closeCard(c):openCard(c));
document.addEventListener('click',e=>{if(!e.target.closest('.card'))cards.forEach(closeCard)});
document.addEventListener('keydown',e=>{if(e.key==='Escape')cards.forEach(closeCard)});
addEventListener('resize',()=>{if(innerWidth!==lastW){lastW=innerWidth;cards.forEach(closeCard)}});

/* ---------- Red neuronal 196-16-16-10 ---------- */
const G=28,L=[G*G,16,16,10];let W=[],B=[],ready=false;
for(let l=1;l<4;l++){const n=L[l],m=L[l-1],s=Math.sqrt(2/m),w=new Float32Array(n*m);
  for(let i=0;i<w.length;i++)w[i]=(Math.random()+Math.random()+Math.random()-1.5)*2*s;W.push(w);B.push(new Float32Array(n));}
function fwd(x){const A=[x];
  for(let l=0;l<3;l++){const n=L[l+1],m=L[l],w=W[l],p=A[l],a=new Float32Array(n);
    for(let i=0;i<n;i++){let s=B[l][i];const o=i*m;for(let j=0;j<m;j++)s+=w[o+j]*p[j];a[i]=l<2?Math.max(0,s):s}
    A.push(a)}
  const o=A[3],mx=Math.max(...o),p=new Float32Array(10);let sum=0;
  for(let i=0;i<10;i++){p[i]=Math.exp(o[i]-mx);sum+=p[i]}
  for(let i=0;i<10;i++)p[i]/=sum;A[3]=p;return A}
function step(x,y,lr){const A=fwd(x);let d=new Float32Array(10);
  for(let i=0;i<10;i++)d[i]=A[3][i]-(i===y?1:0);
  for(let l=2;l>=0;l--){const n=L[l+1],m=L[l],w=W[l],p=A[l];let nd=null;
    if(l>0){nd=new Float32Array(m);
      for(let i=0;i<n;i++){const o=i*m,di=d[i];for(let j=0;j<m;j++)nd[j]+=w[o+j]*di}
      for(let j=0;j<m;j++)if(p[j]<=0)nd[j]=0}
    for(let i=0;i<n;i++){const o=i*m,g=d[i]*lr;for(let j=0;j<m;j++)w[o+j]-=g*p[j];B[l][i]-=g}
    d=nd}}

/* Preprocesado: recorta, escala a 10 px y centra por centro de masa (como MNIST) */
const mk=s=>{const c=document.createElement('canvas');c.width=c.height=s;return c};
const c14=mk(G),x14=c14.getContext('2d',{willReadFrequently:true}),c2=mk(G),x2=c2.getContext('2d',{willReadFrequently:true});
function prep(src){const s=src.width,d=src.getContext('2d').getImageData(0,0,s,s).data;
  let x0=s,y0=s,x1=-1,y1=-1;
  for(let y=0;y<s;y++)for(let x=0;x<s;x++)if(d[(y*s+x)*4]>60){if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y}
  if(x1<0)return null;
  const w=x1-x0+1,h=y1-y0+1,k=(G*5/7)/Math.max(w,h);
  x14.fillStyle='#000';x14.fillRect(0,0,G,G);x14.imageSmoothingEnabled=true;x14.imageSmoothingQuality='high';
  x14.drawImage(src,x0,y0,w,h,(G-w*k)/2,(G-h*k)/2,w*k,h*k);
  const e=x14.getImageData(0,0,G,G).data;let m=0,cx=0,cy=0;
  for(let i=0;i<G*G;i++){const v=e[i*4];m+=v;cx+=v*(i%G);cy+=v*((i/G)|0)}
  if(!m)return null;cx/=m;cy/=m;
  x2.fillStyle='#000';x2.fillRect(0,0,G,G);x2.drawImage(c14,(G-1)/2-cx,(G-1)/2-cy);
  const f=x2.getImageData(0,0,G,G).data,out=new Float32Array(G*G);
  for(let i=0;i<G*G;i++)out[i]=f[i*4]/255;return out}

/* Datos de entrenamiento sintéticos: dígitos con tipografías y deformaciones al azar */
const fonts=['sans-serif','serif','monospace','Georgia','Arial','Courier New','Trebuchet MS','Verdana','cursive'];
const sc=mk(56),sx=sc.getContext('2d',{willReadFrequently:true}),R=Math.random;
function sample(d){sx.setTransform(1,0,0,1,0,0);sx.fillStyle='#000';sx.fillRect(0,0,56,56);
  sx.font=(R()<.5?'bold ':'')+((30+R()*16)|0)+'px '+fonts[(R()*fonts.length)|0];
  sx.textAlign='center';sx.textBaseline='middle';sx.fillStyle=sx.strokeStyle='#fff';sx.lineWidth=R()*3;sx.lineJoin='round';
  sx.translate(28+(R()-.5)*8,28+(R()-.5)*8);sx.rotate((R()-.5)*.5);sx.transform(1,0,(R()-.5)*.5,1,0,0);sx.scale(.8+R()*.4,.9+R()*.3);
  sx.fillText(d,0,0);if(sx.lineWidth>.8)sx.strokeText(d,0,0);return prep(sc)}
const tick=()=>new Promise(r=>setTimeout(r,0));
const setP=(p,t)=>{$('#pb').style.width=(p*100)+'%';if(t)$('#st').textContent=t};
async function train(){
  const N=4000,X=[],Y=[];
  for(let i=0;i<N;i++){const d=i%10,v=sample(d);if(v){X.push(v);Y.push(d)}
    if(i%200===199){setP(i/N*.3,'Generando ejemplos de entrenamiento…');await tick()}}
  const n=X.length,nt=(n*.9)|0,idx=[...Array(nt).keys()],E=30;
  for(let e=0;e<E;e++){
    for(let i=nt-1;i>0;i--){const j=(R()*(i+1))|0;[idx[i],idx[j]]=[idx[j],idx[i]]}
    const lr=.005*(1-e/(E*1.3));
    for(let k=0;k<nt;k++)step(X[idx[k]],Y[idx[k]],lr);
    let ok=0;for(let k=nt;k<n;k++){const p=fwd(X[k])[3];if(p.indexOf(Math.max(...p))===Y[k])ok++}
    setP(.3+.7*(e+1)/E,'Entrenando… vuelta '+(e+1)+' de '+E+' · acierto en ejemplos nuevos: '+Math.round(ok/(n-nt)*100)+' %');
    await tick()}
  ready=true;const b=$('#use');b.disabled=false;b.textContent='Utilizar red neuronal';
  $('#st').textContent+=' · Lista.';$('.bar').hidden=true;}

/* ---------- Dibujo ---------- */
const dc=$('#draw'),dx=dc.getContext('2d',{willReadFrequently:true});dc.width=dc.height=280;
function clr(){dx.fillStyle='#000';dx.fillRect(0,0,280,280);dx.strokeStyle=dx.fillStyle='#fff';dx.lineWidth=20;dx.lineCap=dx.lineJoin='round'}
clr();let down=false,lp;
const pos=e=>{const r=dc.getBoundingClientRect();return[(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height]};
dc.onpointerdown=e=>{down=true;dc.setPointerCapture(e.pointerId);lp=pos(e);dx.beginPath();dx.arc(lp[0],lp[1],10,0,7);dx.fill()};
dc.onpointermove=e=>{if(!down)return;const p=pos(e);dx.beginPath();dx.moveTo(lp[0],lp[1]);dx.lineTo(p[0],p[1]);dx.stroke();lp=p};
dc.onpointerup=dc.onpointercancel=()=>{if(down){down=false;recognize()}};
$('#clr').onclick=()=>{clr();S.A=null;S.x=null;S.show=0;$('#pred').textContent='';draw()};
$('#go').onclick=recognize;
$('#use').onclick=()=>{const st=$('#stage'),o=st.classList.toggle('open');$('#use').textContent=o?'Cerrar la red neuronal':'Utilizar red neuronal'};

/* ---------- Visualización ---------- */
const cv=$('#net'),cx=cv.getContext('2d'),CW=720,CH=460;cv.width=CW*2;cv.height=CH*2;cx.scale(2,2);
const X1=290,X2=455,X3=620,IX=20,IY=(CH-112)/2+10;
const ys=[0,0,0,0].map((_,l)=>l===0?null:Array.from({length:L[l]},(_,i)=>l===3?60+i*(350/9):50+i*(375/15)));
const px=[0,X1,X2,X3],S={A:null,x:null,show:0};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function draw(){const C={panel:css('--panel'),ink:css('--ink'),mut:css('--mut'),line:css('--line'),pos:css('--pos'),neg:css('--neg'),ok:css('--ok')};
  cx.clearRect(0,0,CW,CH);const A=S.A,f=l=>A?Math.max(0,Math.min(1,S.show-(l-1))):0;
  /* conexiones tenues */
  cx.lineWidth=.6;cx.strokeStyle=C.line;cx.globalAlpha=.45;cx.beginPath();
  for(let l=2;l<4;l++)for(let i=0;i<L[l];i++)for(let j=0;j<L[l-1];j++){cx.moveTo(px[l-1],ys[l-1][j]);cx.lineTo(px[l],ys[l][i])}
  for(let i=0;i<16;i++){cx.moveTo(IX+112,IY+56);cx.lineTo(X1,ys[1][i])}
  cx.stroke();cx.globalAlpha=1;
  const nrm=[null,null,null,null];
  if(A){for(let l=1;l<4;l++){const mx=Math.max(1e-6,...A[l]);nrm[l]=Array.from(A[l],v=>l===3?v:v/mx)}
    /* conexiones activas */
    for(let l=2;l<4;l++){const m=L[l-1],n=L[l],w=W[l-1],fl=f(l);if(fl<=0)continue;let mc=1e-6;const c=[];
      for(let i=0;i<n;i++)for(let j=0;j<m;j++){const v=w[i*m+j]*A[l-1][j];c.push(v);mc=Math.max(mc,Math.abs(v))}
      let k=0;for(let i=0;i<n;i++)for(let j=0;j<m;j++){const v=c[k++],a=Math.abs(v)/mc*fl;if(a<.08)continue;
        cx.strokeStyle=v>0?C.pos:C.neg;cx.globalAlpha=Math.min(1,a);cx.lineWidth=.5+a*1.6;
        cx.beginPath();cx.moveTo(px[l-1],ys[l-1][j]);cx.lineTo(px[l],ys[l][i]);cx.stroke()}}
    cx.lineWidth=1.4;cx.strokeStyle='#f2a100';
    for(let i=0;i<16;i++){const a=nrm[1][i]*f(1);if(a<.05)continue;cx.globalAlpha=a;cx.beginPath();cx.moveTo(IX+112,IY+56);cx.lineTo(X1,ys[1][i]);cx.stroke()}
    cx.globalAlpha=1}
  /* entrada */
  cx.fillStyle='#000';cx.fillRect(IX,IY,112,112);
  if(S.x)for(let i=0;i<G*G;i++){cx.fillStyle='rgba(255,255,255,'+S.x[i]+')';cx.fillRect(IX+(i%G)*(112/G),IY+((i/G)|0)*(112/G),112/G+.5,112/G+.5)}
  cx.strokeStyle=C.mut;cx.lineWidth=1;cx.strokeRect(IX,IY,112,112);
  /* neuronas */
  const win=A?A[3].indexOf(Math.max(...A[3])):-1;
  for(let l=1;l<4;l++)for(let i=0;i<L[l];i++){const v=A?nrm[l][i]*f(l):0,x=px[l],y=ys[l][i];
    cx.beginPath();cx.arc(x,y,l===3?12:9,0,7);cx.fillStyle=C.panel;cx.fill();
    cx.fillStyle=(l===3&&i===win&&f(3)>=1)?C.ok:'rgba(242,161,0,'+v+')';cx.fill();
    cx.strokeStyle=C.ink;cx.globalAlpha=.55;cx.lineWidth=1.2;cx.stroke();cx.globalAlpha=1;
    if(l===3){cx.fillStyle=(i===win&&f(3)>=1)?C.ink:C.mut;cx.font=(i===win&&f(3)>=1?'800 ':'600 ')+'16px "Bricolage Grotesque",system-ui,sans-serif';cx.textAlign='left';cx.textBaseline='middle';
      cx.fillText(i+(A&&f(3)>0?'  '+Math.round(A[3][i]*f(3)*100)+' %':''),X3+22,y)}}
  cx.fillStyle=C.mut;cx.font='13px "Bricolage Grotesque",system-ui,sans-serif';cx.textAlign='center';cx.textBaseline='alphabetic';
  cx.fillText('Entrada 28×28',IX+56,IY-14);cx.fillText('Capa oculta 1',X1,24);cx.fillText('Capa oculta 2',X2,24);cx.fillText('Salida',X3+30,24)}
let raf=0,t0=0;
function anim(t){if(!t0)t0=t;S.show=Math.min(3,S.show+(t-t0)/700);t0=t;draw();
  if(S.show<3)raf=requestAnimationFrame(anim);else{t0=0;raf=0}}
function recognize(){if(!ready){$('#pred').textContent='La red todavía se está entrenando…';return}
  const x=prep(dc);if(!x){$('#pred').textContent='Dibujá un número primero.';return}
  S.x=x;S.A=fwd(x);S.show=0;const p=S.A[3],w=p.indexOf(Math.max(...p));
  $('#pred').textContent='Es un '+w+' ('+Math.round(p[w]*100)+' % de confianza)';
  cancelAnimationFrame(raf);t0=0;raf=requestAnimationFrame(anim)}
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
draw();
document.fonts&&document.fonts.ready.then(draw);
/* ---------- Ejemplo codificador / decodificador ---------- */
const ek=$('#e-k'),ei=$('#e-in'),ein=ei.getContext('2d'),eo=$('#e-out').getContext('2d'),el=$('#e-lat').getContext('2d'),ec=mk(2),exc=ec.getContext('2d',{willReadFrequently:true});
let ev=null;
function edraw(){if(!ev)return;const n=+ek.value,im=ein.createImageData(G,G);
  for(let i=0;i<G*G;i++){const v=ev[i]*255;im.data.set([v,v,v,255],i*4)}ein.putImageData(im,0,0);
  ec.width=ec.height=n;exc.imageSmoothingEnabled=true;exc.imageSmoothingQuality='high';exc.fillStyle='#000';exc.fillRect(0,0,n,n);exc.drawImage(ei,0,0,n,n);
  const d=exc.getImageData(0,0,n,n).data,s=112/n;el.fillStyle='#000';el.fillRect(0,0,112,112);
  for(let i=0;i<n*n;i++){const v=d[i*4];el.fillStyle='rgb('+v+','+v+','+v+')';el.fillRect((i%n)*s,((i/n)|0)*s,s-.5,s-.5)}
  eo.imageSmoothingEnabled=true;eo.imageSmoothingQuality='high';eo.drawImage(ec,0,0,n,n,0,0,G,G);
  const o=eo.getImageData(0,0,G,G).data;let e=0;for(let i=0;i<G*G;i++)e+=Math.abs(o[i*4]/255-ev[i]);
  edsvg(n)}
function edsvg(n){const S=24+n*3,cy=145,lx=380-S/2,ly=cy-S/2,
  im=(c,x,y,w,h)=>`<image href="${c.toDataURL()}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="none" style="image-rendering:pixelated"/><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" style="stroke:var(--line)"/>`;
  $('#edv').innerHTML=`<polygon points="150,75 ${lx},${ly} ${lx},${ly+S} 150,215" style="fill:var(--panel);stroke:var(--pos);stroke-width:2"/>`+
  `<polygon points="${lx+S},${ly} 610,75 610,215 ${lx+S},${ly+S}" style="fill:var(--panel);stroke:var(--acc);stroke-width:2"/>`+
  im(ei,10,75,140,140)+im($('#e-lat'),lx,ly,S,S)+im($('#e-out'),610,75,140,140)+
  `<text class="t" x="235" y="142" text-anchor="middle">CODIFICADOR</text><text class="s" x="235" y="162" text-anchor="middle">función F(·)</text>`+
  `<text class="t" x="525" y="142" text-anchor="middle">DECODIFICADOR</text><text class="s" x="525" y="162" text-anchor="middle">función G(·)</text>`+
  `<text x="80" y="240" text-anchor="middle">Imagen original</text><text class="s" x="80" y="258" text-anchor="middle">X</text>`+
  `<text x="380" y="240" text-anchor="middle">Imagen comprimida</text><text class="s" x="380" y="258" text-anchor="middle">F(X)</text>`+
  `<text x="680" y="240" text-anchor="middle">Imagen reconstruida</text><text class="s" x="680" y="258" text-anchor="middle">X′ = (G∘F)(X)</text>`}
for(let d=0;d<10;d++){const b=document.createElement('button');b.className='alt';b.textContent=d;b.setAttribute('aria-label','Dígito '+d);
  b.onclick=()=>{ev=sample(d)||ev;edraw()};$('#eddig').appendChild(b)}

ek.oninput=edraw;ev=sample(7);edraw();

/* ---------- Vectores y espacio vectorial ---------- */
const WD=[['perro',.18,.28,0],['gato',.26,.18,0],['caballo',.12,.42,0],['auto',.7,.25,1],['camión',.82,.38,1],['tren',.62,.4,1],['alegría',.3,.78,2],['tristeza',.46,.88,2],['miedo',.2,.9,2]],GC=['#f2a100','#5aa0ff','#6fcf97'];let vsel=0;
function vdraw(){const w=WD[vsel],s=[...w[0]].reduce((a,c)=>a*31+c.charCodeAt(0),7),
  v=Array.from({length:12},(_,k)=>k===0?w[1]*2-1:k===1?w[2]*2-1:Math.sin(s*(k+3))*.9);
  if($('#vb'))$('#vb').innerHTML=v.map(n=>`<div class="b"><i style="height:${Math.abs(n)*60}px;${n>0?'bottom:50%;background:var(--pos)':'top:50%;background:var(--neg)'}"></i><s>${n.toFixed(2)}</s></div>`).join('');
  if($('#vc'))$('#vc').textContent='«'+w[0]+'» es una lista de '+v.length+' números. En un LLM real son cientos o miles.';
  if($('#vw'))[...$('#vw').children].forEach((b,i)=>b.style.outline=i===vsel?'2px solid var(--acc)':'none');
  const P=WD.map(q=>[20+q[1]*560,20+q[2]*300]),
  nn=WD.map((q,i)=>[i,Math.hypot(q[1]-w[1],q[2]-w[2])]).filter(a=>a[0]!==vsel).sort((a,b)=>a[1]-b[1]).slice(0,2);let h='';
  nn.forEach(a=>h+=`<line x1="${P[vsel][0]}" y1="${P[vsel][1]}" x2="${P[a[0]][0]}" y2="${P[a[0]][1]}"/>`);
  WD.forEach((q,i)=>h+=`<circle data-i="${i}" cx="${P[i][0]}" cy="${P[i][1]}" r="${i===vsel?11:8}" fill="${GC[q[3]]}" style="stroke:${i===vsel?'var(--ink)':'none'};stroke-width:3"/><text x="${P[i][0]+15}" y="${P[i][1]+4}">${q[0]}</text>`);
  h+='<text class="ax" x="20" y="334"></text>';$('#vs').innerHTML=h;
  $('#vn')}
if($('#vw'))WD.forEach((q,i)=>{const b=document.createElement('button');b.className='alt';b.textContent=q[0];b.onclick=()=>{vsel=i;vdraw()};$('#vw').appendChild(b)});
$('#vs').onclick=e=>{const i=e.target.dataset.i;if(i!=null){vsel=+i;vdraw()}};vdraw();

/* ---------- RLHF vs RLAIF ---------- */
function run(ol,d){let i=0;const go=()=>{[...ol.children].forEach(l=>l.classList.remove('on'));if(i>=4)return;ol.children[i].classList.add('on');setTimeout(go,d[i++])};go()}

/* ---------- Menú: resalta la parte de la página donde estás ---------- */
const NAV={inicio:$('#t-inicio'),red:$('#t-red'),codificador:$('#t-codificador'),rlhf:$('#t-rlhf')},
  SEC=[['s-inicio','inicio'],['s-red','red'],['s-codificador','codificador'],['s-espacio','codificador'],['s-rlhf','rlhf']];
function spy(){let cur='inicio';SEC.forEach(([id,k])=>{if($('#'+id).getBoundingClientRect().top<=140)cur=k});
  for(const k in NAV){if(k===cur)NAV[k].setAttribute('aria-current','true');else NAV[k].removeAttribute('aria-current')}}
addEventListener('scroll',spy,{passive:true});spy();

/* ---------- Navbar: se oculta al bajar y reaparece al subir ---------- */
const navbar=document.querySelector('nav');
let lastScrollY=scrollY;
addEventListener('scroll',()=>{
  const y=scrollY;
  if(y<=10||y<lastScrollY)navbar.classList.remove('nav-hidden');
  else if(y>lastScrollY)navbar.classList.add('nav-hidden');
  lastScrollY=Math.max(y,0);
},{passive:true});

train();
