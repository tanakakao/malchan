let plotlySourcePromise = null;

export function safeScriptJson(value) {
  return JSON.stringify(value)
    .replaceAll("&", "\\u0026")
    .replaceAll("<", "\\u003c")
    .replaceAll(">", "\\u003e")
    .replaceAll("\u2028", "\\u2028")
    .replaceAll("\u2029", "\\u2029");
}

export function safeInlineScript(source) {
  return String(source || "").replace(/<\/script/gi, "<\\/script");
}

export async function loadPlotlySource() {
  if (!plotlySourcePromise) {
    plotlySourcePromise = import("plotly.js-dist-min/plotly.min.js?raw")
      .then((module) => module.default || "")
      .catch(() => "");
  }
  return plotlySourcePromise;
}

export const INTERACTIVE_REPORT_CSS = `
    [hidden]{display:none!important}
    .export-figure-card header{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
    .export-edit-button{min-height:28px;padding:5px 9px;border:1px solid color-mix(in srgb,var(--primary) 30%,var(--line));border-radius:8px;background:var(--primary-soft);color:var(--primary);font-size:9px;white-space:nowrap;cursor:pointer}
    .export-figure-open{position:relative;width:100%;min-height:0;padding:0;border:0;border-radius:0;background:#fff;box-shadow:none;cursor:zoom-in}
    .export-figure-open:hover{background:#fff;box-shadow:none}
    .export-figure-open img,.export-figure-static{display:block;width:100%;height:auto}
    .export-figure-open span{position:absolute;right:10px;bottom:10px;padding:6px 9px;border-radius:999px;background:rgba(32,26,26,.82);color:#fff;font-size:9px;font-weight:700;opacity:0;transition:opacity .15s}
    .export-figure-open:hover span,.export-figure-open:focus-visible span{opacity:1}
    body.report-modal-open{overflow:hidden}
    .report-figure-modal{position:fixed;inset:0;z-index:9999;display:grid;place-items:center;padding:18px;background:rgba(20,15,15,.72);backdrop-filter:blur(7px)}
    .report-figure-dialog{width:min(1480px,97vw);height:min(940px,96vh);display:grid;grid-template-rows:auto auto minmax(0,1fr);overflow:hidden;border:1px solid rgba(255,255,255,.18);border-radius:18px;background:#fff;box-shadow:0 28px 80px rgba(0,0,0,.45)}
    .report-figure-dialog>header{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:12px 16px;border-bottom:1px solid var(--line);background:#faf6f6}
    .report-figure-dialog h3{margin:0;font-size:16px}
    .report-modal-close{width:36px;height:36px;min-height:36px;padding:0;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--text);font-size:22px;cursor:pointer}
    .report-figure-editor{display:flex;align-items:end;gap:8px;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid var(--line);background:#fff}
    .report-figure-editor label{display:grid;gap:3px;color:var(--muted);font-size:9px;font-weight:700}
    .report-figure-editor input,.report-figure-editor select{width:112px;min-height:34px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--text);font-size:11px}
    .report-figure-editor select{width:100px}
    .report-figure-editor button{min-height:34px;padding:7px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--text);font-size:10px;font-weight:700;cursor:pointer}
    .report-figure-editor button.primary{border-color:var(--primary);background:var(--primary);color:#fff}
    .report-editor-status{min-width:180px;align-self:center;color:var(--muted);font-size:9px}
    .report-figure-stage{position:relative;min-height:0;overflow:hidden;background:#fff}
    .report-figure-plot{width:100%;height:100%;min-height:460px}
    .report-figure-fallback{width:100%;height:100%;object-fit:contain;background:#fff}
    .report-figure-help{position:absolute;right:12px;bottom:8px;padding:5px 8px;border-radius:8px;background:rgba(255,255,255,.88);color:var(--muted);font-size:9px;pointer-events:none}
    @media(max-width:850px){.report-figure-modal{padding:6px}.report-figure-dialog{width:99vw;height:98vh;border-radius:10px}.report-figure-editor{max-height:180px;overflow:auto}.report-figure-editor input{width:92px}}
    @media print{.export-edit-button,.export-figure-open span,.report-figure-modal{display:none!important}.export-figure-open{cursor:default}}
`;

export function interactiveModalHtml() {
  return `
  <div id="malchan-figure-modal" class="report-figure-modal" hidden>
    <section class="report-figure-dialog" role="dialog" aria-modal="true" aria-labelledby="malchan-figure-modal-title">
      <header>
        <h3 id="malchan-figure-modal-title">図の拡大・編集</h3>
        <button type="button" class="report-modal-close" id="malchan-figure-modal-close" aria-label="閉じる">×</button>
      </header>
      <div class="report-figure-editor" id="malchan-figure-editor">
        <label>X軸<select id="malchan-x-axis"></select></label>
        <label>X最小<input id="malchan-x-min" type="number" step="any" placeholder="自動"></label>
        <label>X最大<input id="malchan-x-max" type="number" step="any" placeholder="自動"></label>
        <label>Y軸<select id="malchan-y-axis"></select></label>
        <label>Y最小<input id="malchan-y-min" type="number" step="any" placeholder="自動"></label>
        <label>Y最大<input id="malchan-y-max" type="number" step="any" placeholder="自動"></label>
        <label>文字サイズ<input id="malchan-font-size" type="number" min="8" max="40" step="1" value="13"></label>
        <label>図の高さ<input id="malchan-plot-height" type="number" min="420" max="1400" step="20" value="700"></label>
        <button type="button" class="primary" id="malchan-apply-editor">反映</button>
        <button type="button" id="malchan-autoscale-editor">選択軸を自動</button>
        <button type="button" id="malchan-reset-editor">初期状態</button>
        <button type="button" id="malchan-download-editor">PNG保存</button>
        <span class="report-editor-status" id="malchan-editor-status"></span>
      </div>
      <div class="report-figure-stage">
        <div id="malchan-figure-plot" class="report-figure-plot"></div>
        <img id="malchan-figure-fallback" class="report-figure-fallback" alt="拡大図" hidden>
        <span class="report-figure-help">ドラッグで範囲選択、ホイールでズーム、ダブルクリックで自動範囲</span>
      </div>
    </section>
  </div>`;
}

export function interactiveRuntimeScript() {
  return `(function(){
const node=document.getElementById("malchan-interactive-figures");if(!node)return;
const registry=JSON.parse(node.textContent||"{}");
const $=(id)=>document.getElementById(id),modal=$("malchan-figure-modal"),title=$("malchan-figure-modal-title"),close=$("malchan-figure-modal-close"),editor=$("malchan-figure-editor"),plot=$("malchan-figure-plot"),fallback=$("malchan-figure-fallback"),xAxis=$("malchan-x-axis"),yAxis=$("malchan-y-axis"),xMin=$("malchan-x-min"),xMax=$("malchan-x-max"),yMin=$("malchan-y-min"),yMax=$("malchan-y-max"),font=$("malchan-font-size"),height=$("malchan-plot-height"),status=$("malchan-editor-status");
let active=null,original=null;const clone=(v)=>JSON.parse(JSON.stringify(v));
const axes=(layout,prefix)=>{const primary=prefix+"axis",keys=Object.keys(layout||{}).filter((key)=>new RegExp("^"+primary+"\\\\d*$").test(key));if(!keys.includes(primary))keys.unshift(primary);return [...new Set(keys)].sort((a,b)=>(Number(a.replace(primary,""))||1)-(Number(b.replace(primary,""))||1));};
const fill=(select,keys,prefix)=>{select.innerHTML="";keys.forEach((key)=>{const option=document.createElement("option");option.value=key;option.textContent=(prefix==="x"?"X軸":"Y軸")+(key.replace(prefix+"axis","")||"1");select.appendChild(option);});};
const setRange=(axis,min,max)=>{const range=plot.layout?.[axis]?.range||[];min.value=Number.isFinite(Number(range[0]))?String(range[0]):"";max.value=Number.isFinite(Number(range[1]))?String(range[1]):"";};
const sync=()=>{const layout=plot.layout||{};fill(xAxis,axes(layout,"x"),"x");fill(yAxis,axes(layout,"y"),"y");setRange(xAxis.value,xMin,xMax);setRange(yAxis.value,yMin,yMax);font.value=String(layout.font?.size||13);height.value=String(Math.round(layout.height||plot.clientHeight||700));};
const readRange=(min,max,label)=>{const a=min.value.trim(),b=max.value.trim();if(!a&&!b)return null;if(!a||!b)throw new Error(label+"は最小・最大の両方を入力してください。");const lo=Number(a),hi=Number(b);if(!Number.isFinite(lo)||!Number.isFinite(hi))throw new Error(label+"は数値で入力してください。");if(lo>=hi)throw new Error(label+"は最小値を最大値より小さくしてください。");return[lo,hi];};
const config={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:"png",scale:2}};
async function render(record){active=record;original=record.figure?clone(record.figure):null;title.textContent=record.title||"図の拡大・編集";status.textContent="";if(window.Plotly&&original){fallback.hidden=true;plot.hidden=false;editor.hidden=false;const layout={...clone(original.layout||{}),autosize:true,width:undefined,height:700,paper_bgcolor:"#fff",plot_bgcolor:"#fff"};await window.Plotly.react(plot,clone(original.data||[]),layout,config);sync();}else{plot.hidden=true;editor.hidden=true;fallback.hidden=false;fallback.src=record.image||"";}}
async function openModal(id){const record=registry[id];if(!record)return;modal.hidden=false;document.body.classList.add("report-modal-open");await render(record);close.focus();}
function closeModal(){modal.hidden=true;document.body.classList.remove("report-modal-open");status.textContent="";}
document.querySelectorAll("[data-open-report-figure]").forEach((button)=>button.addEventListener("click",()=>openModal(button.dataset.openReportFigure)));
close.addEventListener("click",closeModal);modal.addEventListener("click",(event)=>{if(event.target===modal)closeModal();});document.addEventListener("keydown",(event)=>{if(event.key==="Escape"&&!modal.hidden)closeModal();});xAxis.addEventListener("change",()=>setRange(xAxis.value,xMin,xMax));yAxis.addEventListener("change",()=>setRange(yAxis.value,yMin,yMax));
$("malchan-apply-editor").addEventListener("click",async()=>{try{const updates={},size=Math.max(8,Math.min(40,Number(font.value)||13)),h=Math.max(420,Math.min(1400,Number(height.value)||700));updates["font.size"]=size;updates["title.font.size"]=size+3;updates["legend.font.size"]=size;updates.height=h;[...axes(plot.layout||{},"x"),...axes(plot.layout||{},"y")].forEach((axis)=>{updates[axis+".tickfont.size"]=size;updates[axis+".title.font.size"]=size;});const xr=readRange(xMin,xMax,"X軸範囲"),yr=readRange(yMin,yMax,"Y軸範囲");updates[xAxis.value+".autorange"]=xr===null;updates[yAxis.value+".autorange"]=yr===null;if(xr)updates[xAxis.value+".range"]=xr;if(yr)updates[yAxis.value+".range"]=yr;await window.Plotly.relayout(plot,updates);status.textContent="変更を反映しました。";}catch(error){status.textContent=error.message||String(error);}});
$("malchan-autoscale-editor").addEventListener("click",async()=>{await window.Plotly.relayout(plot,{[xAxis.value+".autorange"]:true,[yAxis.value+".autorange"]:true});setRange(xAxis.value,xMin,xMax);setRange(yAxis.value,yMin,yMax);status.textContent="選択軸を自動範囲に戻しました。";});
$("malchan-reset-editor").addEventListener("click",async()=>{if(!original)return;const layout={...clone(original.layout||{}),autosize:true,width:undefined,height:700,paper_bgcolor:"#fff",plot_bgcolor:"#fff"};await window.Plotly.react(plot,clone(original.data||[]),layout,config);sync();status.textContent="初期状態に戻しました。";});
$("malchan-download-editor").addEventListener("click",()=>{if(!window.Plotly||!active)return;const name=String(active.title||"malchan-figure").normalize("NFKC").replace(/[\\\\/:*?\"<>|]+/g,"_").replace(/\\s+/g,"_");window.Plotly.downloadImage(plot,{format:"png",filename:name,width:Math.max(900,plot.clientWidth||1200),height:Number(height.value)||700,scale:2});});
})();`;
}
