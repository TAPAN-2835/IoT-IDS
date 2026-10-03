/* ═══════════════════════════════════════════════════════════════════════════
   IoT-IDS Dashboard — predict.js
   Live demo: pick a held-out test packet, classify it with the final CNN-GRU,
   and show the SHAP reasons behind the verdict.
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const lock = document.getElementById('predict-lock');
  const lockTitle = document.getElementById('predict-lock-title');
  const lockDesc = document.getElementById('predict-lock-desc');
  const form = document.getElementById('predict-form');
  const modelLine = document.getElementById('demo-model-line');
  const result = document.getElementById('predict-result');
  if (!lock || !form) return;

  const pct = (v) => (v * 100).toFixed(1) + '%';
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  async function getJSON(url, options) {
    const res = await fetch(url, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    return body;
  }

  async function waitForModel(attempt = 0) {
    try {
      const info = await getJSON('/api/demo/info');
      lock.style.display = 'none';
      form.style.display = 'flex';
      modelLine.innerHTML =
        `<strong>${esc(info.model)}</strong> · ${esc(info.dataset)} · ${info.parameters.toLocaleString()} parameters · ` +
        `${info.size_kb} KB<br><span class="demo-muted">Test set: macro-F1 ${info.test_macro_f1.toFixed(3)}, ` +
        `detects ${pct(info.test_detection_rate)} of attacks with ${pct(info.test_false_alarm_rate)} false alarms ` +
        `(threshold ${info.threshold.toFixed(2)}).</span>`;
      // Presentation shortcut: /#predict-demo scrolls here and runs one attack sample.
      if (location.hash === '#predict-demo') {
        document.getElementById('predict').scrollIntoView();
        form.querySelector('button[data-kind="attack"]').click();
      }
    } catch (err) {
      if (/HTTP 5\d\d/.test(err.message) && attempt < 40) {
        setTimeout(() => waitForModel(attempt + 1), 2000);   // still loading
        return;
      }
      lock.querySelector('.lock-icon').textContent = '⚠️';
      lockTitle.textContent = 'Live demo unavailable';
      lockDesc.textContent = err.message;
    }
  }

  function renderResult(r) {
    const maxAbs = Math.max(...r.top_reasons.map((x) => Math.abs(x.contribution)), 1e-9);
    const verdictClass = r.prediction === 'ATTACK' ? 'verdict-attack' : 'verdict-normal';
    const check = r.correct
      ? '<span class="demo-ok">✓ correct</span>'
      : '<span class="demo-miss">✗ wrong</span>';
    const missNote = (!r.correct && r.actual === 'ATTACK')
      ? '<p class="demo-note">Missed attack: about a third of attack packets in this dataset have exactly the same ' +
        'features as normal packets, so no packet-level model can catch them (see the detection-ceiling figure).</p>'
      : '';
    const rows = r.top_reasons.map((x) => {
      const width = Math.max(4, (Math.abs(x.contribution) / maxAbs) * 100);
      const dir = x.pushes_towards === 'attack' ? 'towards-attack' : 'towards-normal';
      return `<div class="reason-row">
          <div class="reason-name" title="${esc(x.feature)}">${esc(x.feature)} <span class="demo-muted">= ${esc(x.value)}</span></div>
          <div class="reason-bar-track"><div class="reason-bar ${dir}" style="width:${width}%"></div></div>
          <div class="reason-dir ${dir}">→ ${x.pushes_towards}</div>
        </div>`;
    }).join('');
    result.innerHTML = `
      <div class="demo-verdict-line">
        <span class="verdict ${verdictClass}">${r.prediction}</span>
        <span>attack probability <strong>${pct(r.probability_attack)}</strong></span>
        <span class="demo-muted">actual: ${r.actual}</span>
        ${check}
        <span class="demo-muted">${r.latency_ms.toFixed(1)} ms on CPU</span>
      </div>
      ${missNote}
      <div class="demo-reasons-title">Why? Top SHAP reasons for this packet</div>
      ${rows}
      <div class="demo-muted demo-footnote">Packet #${r.sample_id} from the held-out test set (never used in training).</div>`;
    result.classList.add('visible');
  }

  form.querySelectorAll('button[data-kind]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      form.querySelectorAll('button[data-kind]').forEach((b) => (b.disabled = true));
      result.classList.add('visible');
      result.innerHTML = '<span class="demo-muted">Classifying and explaining…</span>';
      try {
        const sample = await getJSON(`/api/demo/sample?kind=${btn.dataset.kind}`);
        const r = await getJSON('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_id: sample.sample_id }),
        });
        renderResult(r);
      } catch (err) {
        result.innerHTML = `<span class="demo-miss">Error: ${esc(err.message)}</span>`;
      } finally {
        form.querySelectorAll('button[data-kind]').forEach((b) => (b.disabled = false));
      }
    });
  });

  waitForModel();
})();
