const $ = id => document.getElementById(id);
let state = {status: 'empty'}, pending = false, timer;
function controls() {
  const busy = pending || state.status === 'busy';
  $('load').disabled = busy || !$('scan').options.length;
  $('scan').disabled = busy;
  $('checkpoint').disabled = busy;
  $('run').disabled = busy || !state.slices || !$('checkpoint').options.length || state.scan_index !== Number($('scan').value);
  $('slice').disabled = busy || !state.slices;
  $('overlay').disabled = busy || !state.prediction;
  $('download').hidden = busy || !state.prediction;
}
function slice() {
  if (!state.slices) return;
  $('slice-label').textContent = `${Number($('slice').value) + 1} / ${state.slices}`;
  $('image').src = `/api/slice?z=${$('slice').value}&overlay=${$('overlay').checked ? 1 : 0}&v=${Date.now()}`;
  $('image').hidden = false;
  $('placeholder').hidden = true;
}
async function poll() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw Error('Connection failed.');
    state = await response.json();
    $('status').textContent = state.message;
    controls();
    if (state.status === 'busy') { timer = setTimeout(poll, 1000); return; }
    if (state.slices) {
      $('slice').max = state.slices - 1;
      if (Number($('slice').value) >= state.slices) $('slice').value = Math.floor(state.slices / 2);
      slice();
    }
    $('summary').textContent = state.prediction
      ? `${state.checkpoint} · Epoch ${state.epoch} · Threshold ${state.threshold} · ${state.voxels.toLocaleString()} predicted foreground voxels · ${state.seconds}s inference. No accuracy measured.` : '';
  } catch (error) {
    pending = false; controls();
    $('status').textContent = 'Cannot reach the local server. Check the terminal and reload this page.';
  }
}
async function action(name, choice) {
  clearTimeout(timer); pending = true; controls();
  try {
    const response = await fetch(`/api/${name}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({choice:Number(choice)})});
    const value = await response.json();
    if (!response.ok) throw Error(value.message || 'Request failed.');
    if (name === 'open') { $('image').hidden = true; $('placeholder').hidden = false; $('slice').value = 0; }
    pending = false; await poll();
  } catch (error) { pending = false; controls(); $('status').textContent = error.message; }
}
$('load').onclick = () => action('open', $('scan').value);
$('scan').onchange = () => { controls(); $('status').textContent = 'Click Load scan to view your selection.'; };
$('run').onclick = () => action('predict', $('checkpoint').value);
$('slice').oninput = slice;
$('overlay').onchange = slice;
async function init() {
  try {
    const response = await fetch('/api/catalog');
    if (!response.ok) throw Error();
    const catalog = await response.json();
    for (const [key, id] of [['scans', 'scan'], ['checkpoints', 'checkpoint']]) {
      catalog[key].forEach((label, index) => $(id).add(new Option(label, index)));
    }
    const best = catalog.checkpoints.indexOf('pilot6-hardneg-run1');
    if (best >= 0) $('checkpoint').value = best;
    await poll();
  } catch { $('status').textContent = 'Cannot load scans. Check the terminal and reload.'; }
}
init();
