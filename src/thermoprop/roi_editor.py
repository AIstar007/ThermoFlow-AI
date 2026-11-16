"""Interactive browser-side polygon/rectangle ROI editor using Streamlit Components v2."""
from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image


ROI_HTML = r"""
<div id="tp-roi-root">
  <div class="tp-roi-toolbar">
    <input id="tp-roi-name" type="text" value="ROI 1" aria-label="ROI name" />
    <input id="tp-roi-color" type="color" value="#4de2c5" aria-label="ROI colour" />
    <button id="tp-poly" class="active">＋ Polygon</button>
    <button id="tp-rect">▭ Rectangle</button>
    <button id="tp-finish">Finish shape</button>
    <button id="tp-full">Full frame</button>
    <button id="tp-undo">Undo</button>
    <button id="tp-clear" class="danger">Clear all</button>
  </div>
  <div class="tp-roi-status" id="tp-roi-status">Choose Polygon or Rectangle, then draw on the thermal image. Click “Finish shape” to save a polygon.</div>
  <div class="tp-roi-canvas-wrap"><canvas id="tp-roi-canvas"></canvas></div>
  <div class="tp-roi-footer">
    <span id="tp-roi-count">No named regions</span>
    <span>Coordinates are stored in source-image pixels</span>
  </div>
</div>
"""

ROI_CSS = r"""
#tp-roi-root { color:var(--st-text-color); font-family:var(--st-font); }
.tp-roi-toolbar { display:flex; flex-wrap:wrap; gap:7px; align-items:center; margin:0 0 8px; }
.tp-roi-toolbar input[type=text] { min-width:105px; max-width:180px; background:var(--st-secondary-background-color); color:var(--st-text-color); border:1px solid var(--st-border-color); border-radius:8px; padding:8px 10px; }
.tp-roi-toolbar input[type=color] { width:37px; height:35px; padding:2px; background:transparent; border:1px solid var(--st-border-color); border-radius:8px; }
.tp-roi-toolbar button { border:1px solid var(--st-border-color); background:var(--st-secondary-background-color); color:var(--st-text-color); border-radius:8px; padding:8px 11px; cursor:pointer; font-size:13px; transition:transform .15s,border-color .15s,background .15s; }
.tp-roi-toolbar button:hover { transform:translateY(-1px); border-color:var(--st-primary-color); }
.tp-roi-toolbar button.active { background:color-mix(in srgb,var(--st-primary-color) 18%,var(--st-secondary-background-color)); border-color:var(--st-primary-color); }
.tp-roi-toolbar button.danger { color:#ff9a9a; }
.tp-roi-status { color:var(--st-text-color); opacity:.76; font-size:12px; padding:3px 0 9px; }
.tp-roi-canvas-wrap { width:100%; overflow:hidden; background:#07111f; border:1px solid var(--st-border-color); border-radius:12px; display:flex; justify-content:center; align-items:center; padding:8px; }
#tp-roi-canvas { max-width:100%; height:auto; display:block; touch-action:none; cursor:crosshair; border-radius:5px; }
.tp-roi-footer { display:flex; justify-content:space-between; flex-wrap:wrap; color:var(--st-text-color); opacity:.58; font-size:11px; padding:7px 1px 0; }
"""

ROI_JS = r"""
export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement.querySelector('#tp-roi-root');
  if (!root) return;
  if (root.__tpROIEditor) { root.__tpROIEditor.update(data); return () => {}; }
  const canvas = root.querySelector('#tp-roi-canvas');
  const ctx = canvas.getContext('2d');
  const image = new Image();
  const nameInput = root.querySelector('#tp-roi-name');
  const colorInput = root.querySelector('#tp-roi-color');
  const status = root.querySelector('#tp-roi-status');
  const count = root.querySelector('#tp-roi-count');
  const polyBtn = root.querySelector('#tp-poly');
  const rectBtn = root.querySelector('#tp-rect');
  const finishBtn = root.querySelector('#tp-finish');
  const fullBtn = root.querySelector('#tp-full');
  const undoBtn = root.querySelector('#tp-undo');
  const clearBtn = root.querySelector('#tp-clear');

  let regions = Array.isArray(data?.regions) ? data.regions.map(r => ({...r})) : [];
  let mode = 'polygon';
  let points = [];
  let rectAnchor = null;
  let pointerNow = null;
  let sourceW = Number(data?.image_width || 1);
  let sourceH = Number(data?.image_height || 1);
  let scale = 1;
  let imageUrl = data?.image_data_url || '';

  function setStatus(text) { status.textContent = text; }
  function send() {
    setStateValue('regions', regions);
    count.textContent = regions.length ? `${regions.length} named ROI${regions.length===1?'':'s'} · ${regions.map(r=>r.name).join(' · ')}` : 'No named regions';
  }
  function setMode(next) {
    mode = next; points = []; rectAnchor = null; pointerNow = null;
    polyBtn.classList.toggle('active', mode==='polygon');
    rectBtn.classList.toggle('active', mode==='rectangle');
    setStatus(mode==='polygon' ? 'Polygon mode: click to add vertices, then Finish shape.' : 'Rectangle mode: click once for the first corner, then click the opposite corner.');
    draw();
  }
  function mapPoint(event) {
    const r = canvas.getBoundingClientRect();
    const cssX = Math.max(0, Math.min(r.width, event.clientX-r.left));
    const cssY = Math.max(0, Math.min(r.height, event.clientY-r.top));
    return {x: cssX / r.width * sourceW, y: cssY / r.height * sourceH};
  }
  function polygonPath(pts) {
    if (!pts || pts.length<1) return;
    ctx.beginPath(); ctx.moveTo(pts[0].x*scale, pts[0].y*scale);
    for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x*scale, pts[i].y*scale);
  }
  function draw() {
    if (!image.complete || !image.naturalWidth) return;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(image,0,0,canvas.width,canvas.height);
    for (const region of regions) {
      const pts=(region.points||[]).map(p=>({x:Number(p[0]),y:Number(p[1])}));
      if (pts.length<3) continue;
      polygonPath(pts); ctx.closePath();
      ctx.fillStyle=region.color||'#4de2c5'; ctx.globalAlpha=.15; ctx.fill(); ctx.globalAlpha=1;
      ctx.strokeStyle=region.color||'#4de2c5'; ctx.lineWidth=Math.max(2,canvas.width/500); ctx.stroke();
      const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
      const cx=xs.reduce((a,b)=>a+b,0)/xs.length*scale;
      const cy=ys.reduce((a,b)=>a+b,0)/ys.length*scale;
      ctx.font='bold 13px sans-serif'; ctx.lineWidth=3; ctx.strokeStyle='rgba(0,0,0,.7)';
      ctx.strokeText(region.name||'ROI',cx+4,cy-4); ctx.fillStyle=region.color||'#4de2c5'; ctx.fillText(region.name||'ROI',cx+4,cy-4);
    }
    if (points.length) {
      polygonPath(points); if (pointerNow) ctx.lineTo(pointerNow.x*scale,pointerNow.y*scale);
      ctx.strokeStyle=colorInput.value; ctx.lineWidth=2; ctx.setLineDash([6,4]); ctx.stroke(); ctx.setLineDash([]);
      points.forEach(p=>{ctx.beginPath();ctx.arc(p.x*scale,p.y*scale,4,0,Math.PI*2);ctx.fillStyle=colorInput.value;ctx.fill();});
    }
    if (rectAnchor && pointerNow) {
      ctx.strokeStyle=colorInput.value; ctx.lineWidth=2; ctx.setLineDash([6,4]);
      ctx.strokeRect(rectAnchor.x*scale,rectAnchor.y*scale,(pointerNow.x-rectAnchor.x)*scale,(pointerNow.y-rectAnchor.y)*scale); ctx.setLineDash([]);
    }
  }
  function addRegion(pts) {
    if (pts.length<3) { setStatus('A region needs at least three distinct vertices.'); return; }
    const name=(nameInput.value||'').trim() || `ROI ${regions.length+1}`;
    const colour=colorInput.value || '#4de2c5';
    const safePts=pts.map(p=>[Math.max(0,Math.min(sourceW-1,p.x)),Math.max(0,Math.min(sourceH-1,p.y))]);
    regions.push({id:`roi_${Date.now()}_${regions.length}`,name,color:colour,points:safePts});
    points=[]; rectAnchor=null; pointerNow=null;
    nameInput.value=`ROI ${regions.length+1}`;
    send(); setStatus(`Saved ${name}. Draw another region or continue to analysis.`); draw();
  }
  function onDown(event) {
    if (mode==='idle') return;
    event.preventDefault();
    const p=mapPoint(event);
    if (mode==='polygon') {
      points.push(p); pointerNow=null;
      setStatus(`${points.length} polygon vertices. Add more points, then Finish shape.`);
    } else if (mode==='rectangle') {
      if (!rectAnchor) { rectAnchor=p; pointerNow=p; setStatus('Select the opposite corner of the rectangle.'); }
      else {
        const a=rectAnchor; const b=p;
        addRegion([{x:a.x,y:a.y},{x:b.x,y:a.y},{x:b.x,y:b.y},{x:a.x,y:b.y}]);
      }
    }
    draw();
  }
  function onMove(event) { if (mode==='idle') return; pointerNow=mapPoint(event); draw(); }
  function onFinish() { if (mode==='polygon') addRegion(points); else if (rectAnchor) setStatus('Choose the opposite corner to finish the rectangle.'); }
  function onUndo() { if (mode==='polygon' && points.length) {points.pop(); draw();} else if (regions.length) {regions.pop();send();draw();setStatus('Removed the last saved ROI.');} }
  function onFull() { addRegion([{x:0,y:0},{x:sourceW-1,y:0},{x:sourceW-1,y:sourceH-1},{x:0,y:sourceH-1}]); }
  function onClear() { regions=[]; points=[]; rectAnchor=null; send(); draw(); setStatus('All regions cleared.'); }
  const onPoly = () => setMode('polygon');
  const onRect = () => setMode('rectangle');

  function loadImage(url) {
    if (!url) return;
    image.onload=()=>{
      const maxW=1120, maxH=540;
      scale=Math.min(1,maxW/sourceW,maxH/sourceH);
      canvas.width=Math.max(1,Math.round(sourceW*scale));
      canvas.height=Math.max(1,Math.round(sourceH*scale));
      draw();
    };
    image.src=url;
  }
  canvas.addEventListener('pointerdown',onDown);
  canvas.addEventListener('pointermove',onMove);
  polyBtn.addEventListener('click',onPoly);
  rectBtn.addEventListener('click',onRect);
  finishBtn.addEventListener('click',onFinish);
  fullBtn.addEventListener('click',onFull);
  undoBtn.addEventListener('click',onUndo);
  clearBtn.addEventListener('click',onClear);
  colorInput.addEventListener('input',draw);
  loadImage(imageUrl);
  count.textContent=regions.length ? `${regions.length} named ROI${regions.length===1?'':'s'} · ${regions.map(r=>r.name).join(' · ')}` : 'No named regions';
  setStatus('Polygon mode: click to add vertices, then Finish shape.');
  root.__tpROIEditor = { update(nextData) {
    if (Array.isArray(nextData?.regions)) regions = nextData.regions.map(r => ({...r}));
    sourceW = Number(nextData?.image_width || sourceW); sourceH = Number(nextData?.image_height || sourceH);
    const nextUrl = nextData?.image_data_url || imageUrl;
    if (nextUrl !== imageUrl) { imageUrl = nextUrl; loadImage(imageUrl); } else { draw(); }
    count.textContent=regions.length ? `${regions.length} named ROI${regions.length===1?'':'s'} · ${regions.map(r=>r.name).join(' · ')}` : 'No named regions';
  }};
  return () => {
    canvas.removeEventListener('pointerdown',onDown); canvas.removeEventListener('pointermove',onMove);
    polyBtn.removeEventListener('click',onPoly); rectBtn.removeEventListener('click',onRect);
    finishBtn.removeEventListener('click',onFinish); fullBtn.removeEventListener('click',onFull);
    undoBtn.removeEventListener('click',onUndo); clearBtn.removeEventListener('click',onClear);
    delete root.__tpROIEditor;
  };
}
"""


def thermal_image_data_url(frame: np.ndarray) -> str:
    """Create a compact false-colour PNG for the ROI editor (display only)."""
    arr = np.asarray(frame, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot display a frame with no finite pixels")
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0, 1)
    # A lightweight inferno-like ramp; numeric data remains unchanged in analysis.
    stops = np.array([[8, 7, 31], [75, 20, 90], [170, 45, 70], [235, 115, 35], [252, 220, 100]], dtype=float)
    x = norm * (len(stops) - 1)
    low = np.floor(x).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    frac = (x - low)[..., None]
    rgb = ((1 - frac) * stops[low] + frac * stops[high]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_REGISTERED_COMPONENT = None

def _get_component(st):
    global _REGISTERED_COMPONENT
    if _REGISTERED_COMPONENT is None:
        _REGISTERED_COMPONENT = st.components.v2.component(
            name="thermal_pattern_polygon_roi_editor", html=ROI_HTML, css=ROI_CSS, js=ROI_JS, isolate_styles=True
        )
    return _REGISTERED_COMPONENT



def extract_component_regions(result, default: list[dict] | None = None) -> list[dict]:
    """Normalize the Streamlit Components v2 return value to a region list.

    Components v2 returns its state object, while ``mount_roi_editor`` may
    already have unwrapped that state to the list stored under ``regions``.
    Supporting both shapes avoids silently dropping polygon edits on rerun.
    """
    fallback = default or []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        value = result.get("regions")
    else:
        value = getattr(result, "regions", None) if result is not None else None
    return value if isinstance(value, list) else fallback

def mount_roi_editor(st, frame: np.ndarray, regions: list[dict] | None, key: str) -> list[dict]:
    """Mount the Streamlit Components v2 polygon editor and return named ROIs.

    Streamlit 1.67+ is required. If components v2 is unavailable, the app uses
    an explicit full-frame ROI and asks the user to upgrade Streamlit.
    """
    initial = regions or []
    if not hasattr(st, "components") or not hasattr(st.components, "v2"):
        st.warning("Polygon ROI editing requires Streamlit Components v2 (Streamlit 1.67+). The analysis can still run using a full-frame ROI.")
        return initial
    component = _get_component(st)
    h, w = frame.shape
    result = component(
        data={"image_data_url": thermal_image_data_url(frame), "image_width": int(w),
              "image_height": int(h), "regions": initial},
        default={"regions": initial},
        on_regions_change=lambda: None,
        key=key,
        width="stretch",
    )
    if result is not None:
        value = getattr(result, "regions", None)
        if value is None and isinstance(result, dict):
            value = result.get("regions")
        if isinstance(value, list):
            return value
    return initial
