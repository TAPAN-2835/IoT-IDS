/**
 * IoT-IDS Dashboard — app.js
 * Fetches experiment data from the FastAPI backend and renders all UI sections.
 */

'use strict';

// ──────────────────────────────────────────────────────────────────────────────
// Config & Helpers
// ──────────────────────────────────────────────────────────────────────────────

const API = {
  status:          '/api/status',
  experiments:     '/api/experiments',
  experiment:      (id)           => `/api/experiments/${id}`,
  image:           (expId, file)  => `/api/experiments/${expId}/images/${file}`,
  audit:           '/api/audit',
  labels:          '/api/labels',
};

const COLORS = {
  cyan:      '#00d4ff',
  green:     '#00ff9d',
  amber:     '#f59e0b',
  red:       '#f87171',
  purple:    '#a78bfa',
  muted:     '#4b5e78',
  textSec:   '#94a3b8',
};

const CHART_DEFAULTS = {
  color: COLORS.textSec,
  borderColor: 'rgba(0,212,255,0.15)',
  plugins: { legend: { labels: { color: COLORS.textSec, font: { family: 'Inter' } } } },
  scales: {
    x: { ticks: { color: COLORS.textSec }, grid: { color: 'rgba(255,255,255,0.04)' } },
    y: { ticks: { color: COLORS.textSec }, grid: { color: 'rgba(255,255,255,0.04)' } },
  },
};

/** Attack label mapping (key = string index) */
const LABEL_MAP = {};

/** Active Chart.js instances (for cleanup on re-render) */
const charts = {};

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

function fmt(val, decimals = 4) {
  if (val === null || val === undefined || val === '') return '—';
  const n = parseFloat(val);
  return isNaN(n) ? '—' : n.toFixed(decimals);
}
function fmtPct(val) {
  if (val === null || val === undefined) return '—';
  return (parseFloat(val) * 100).toFixed(2) + '%';
}
function fmtMs(val) {
  if (!val && val !== 0) return '—';
  return parseFloat(val).toFixed(4) + ' ms';
}
function fmtMB(val) {
  if (!val && val !== 0) return '—';
  return parseFloat(val).toFixed(1) + ' MB';
}
function fmtInt(val) {
  if (!val && val !== 0) return '—';
  return parseInt(val).toLocaleString();
}

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

/** Get a colour for an F1 score */
function f1Color(f1) {
  if (f1 === null || f1 === undefined) return COLORS.muted;
  const v = parseFloat(f1);
  if (v >= 0.95) return COLORS.green;
  if (v >= 0.75) return COLORS.cyan;
  if (v >= 0.50) return COLORS.amber;
  return COLORS.red;
}

// ──────────────────────────────────────────────────────────────────────────────
// Navbar scroll effect
// ──────────────────────────────────────────────────────────────────────────────

window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 10);
});

// ──────────────────────────────────────────────────────────────────────────────
// Counter animation
// ──────────────────────────────────────────────────────────────────────────────

function animateCounter(el) {
  const target = parseFloat(el.dataset.target);
  const fmt    = el.dataset.format;
  const dur    = 1600;
  const start  = performance.now();

  function tick(now) {
    const pct  = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - pct, 3); // ease-out-cubic
    const cur  = ease * target;

    if (fmt === 'int')      el.textContent = Math.round(cur).toLocaleString();
    else if (fmt === 'pct') el.textContent = cur.toFixed(2) + '%';
    else                    el.textContent = cur.toFixed(4);

    if (pct < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function initCounters() {
  document.querySelectorAll('[data-target]').forEach(el => {
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { animateCounter(el); obs.disconnect(); }
    }, { threshold: 0.3 });
    obs.observe(el);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Pipeline Stepper
// ──────────────────────────────────────────────────────────────────────────────

const PHASE_ICONS = { complete: '✅', in_progress: '⚡', pending: '⏳' };

function renderPipeline(phases) {
  const container = document.getElementById('pipeline-steps');
  if (!phases?.length) return;

  // Update nav status
  const inProg = phases.find(p => p.status === 'in_progress');
  document.getElementById('nav-phase-status').textContent =
    inProg ? `Phase ${inProg.id} in progress` : 'All phases complete';

  container.innerHTML = phases.map(p => `
    <div class="pipeline-step ${p.status}">
      <div class="step-circle">
        <span>${PHASE_ICONS[p.status] || '⏳'}</span>
      </div>
      <div class="step-name">${p.name}</div>
      <div class="step-desc">${p.description.substring(0, 90)}${p.description.length > 90 ? '…' : ''}</div>
      <span class="step-badge badge-${p.status}">${p.status.replace('_', ' ')}</span>
    </div>
  `).join('');
}

// ──────────────────────────────────────────────────────────────────────────────
// Key Metrics Cards
// ──────────────────────────────────────────────────────────────────────────────

function renderMetrics(experiments) {
  const completed = experiments.filter(e => e.status === 'complete');
  const total     = experiments.length;

  const bestAcc  = Math.max(...completed.map(e => e.accuracy || 0));
  const bestF1   = Math.max(...completed.map(e => e.macro_f1 || 0));
  const fastest  = Math.min(...completed.filter(e => e.inference_latency).map(e => e.inference_latency));

  const cards = [
    { icon: '🎯', val: (bestAcc * 100).toFixed(2) + '%', label: 'Best Accuracy', sub: 'E01 Random Forest Binary' },
    { icon: '📊', val: bestF1.toFixed(4),                label: 'Best Macro F1', sub: 'E02 Random Forest Multiclass' },
    { icon: '⚡', val: fastest ? fastest.toFixed(4) + ' ms' : '—', label: 'Fastest Inference',  sub: 'per sample, on test set' },
    { icon: '🧪', val: `${completed.length}/${total}`,   label: 'Experiments Done', sub: `${total - completed.length} pending / in progress` },
  ];

  document.getElementById('metrics-grid').innerHTML = cards.map(c => `
    <div class="metric-card">
      <div class="metric-icon">${c.icon}</div>
      <div class="metric-val">${c.val}</div>
      <div class="metric-label">${c.label}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>
  `).join('');
}

// ──────────────────────────────────────────────────────────────────────────────
// Experiment Table
// ──────────────────────────────────────────────────────────────────────────────

const STATUS_LABELS = { complete: '✅ Complete', in_progress: '⚡ Training', pending: '⏳ Pending' };

function renderExperimentTable(experiments) {
  const tbody = document.getElementById('exp-tbody');

  tbody.innerHTML = experiments.map(e => {
    const isClickable = e.status === 'complete' && e.has_artifacts;
    const accStr  = e.accuracy  !== null ? `<span class="val-good">${(e.accuracy * 100).toFixed(2)}%</span>`  : `<span class="val-muted">—</span>`;
    const f1Str   = e.macro_f1  !== null ? `<span class="val-good">${e.macro_f1.toFixed(4)}</span>`           : `<span class="val-muted">—</span>`;
    const latStr  = e.inference_latency !== null ? fmtMs(e.inference_latency) : `<span class="val-muted">—</span>`;
    const sizeStr = e.model_size_mb    !== null ? fmtMB(e.model_size_mb)     : `<span class="val-muted">—</span>`;

    return `
      <tr class="${isClickable ? 'clickable' : ''}" data-expid="${e.experiment_id}" data-status="${e.status}">
        <td><span class="exp-id">${e.experiment_id}</span></td>
        <td>${e.model}</td>
        <td>${e.task || '—'}</td>
        <td>${accStr}</td>
        <td>${f1Str}</td>
        <td>${latStr}</td>
        <td>${sizeStr}</td>
        <td><span class="badge st-${e.status}">${STATUS_LABELS[e.status] || e.status}</span></td>
      </tr>
    `;
  }).join('');

  // Row click → load detail
  tbody.querySelectorAll('tr.clickable').forEach(row => {
    row.addEventListener('click', () => {
      const expId = row.dataset.expid;
      // Toggle: if already active, close
      if (row.classList.contains('active-row')) {
        row.classList.remove('active-row');
        document.getElementById('detail-panel').style.display = 'none';
        return;
      }
      tbody.querySelectorAll('tr').forEach(r => r.classList.remove('active-row'));
      row.classList.add('active-row');
      loadExperimentDetail(expId);
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Experiment Detail Panel
// ──────────────────────────────────────────────────────────────────────────────

async function loadExperimentDetail(expId) {
  const panel = document.getElementById('detail-panel');
  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Show loading state
  document.getElementById('detail-title').textContent = expId;
  document.getElementById('detail-subtitle').textContent = 'Loading…';
  document.getElementById('detail-metrics').innerHTML = '<div class="skeleton" style="height:60px;width:100%;"></div>';

  const data = await fetchJSON(API.experiment(expId));
  if (!data) {
    document.getElementById('detail-subtitle').textContent = 'Failed to load experiment data.';
    return;
  }

  // Title
  document.getElementById('detail-title').textContent = expId.replace(/_/g, ' ');
  document.getElementById('detail-subtitle').textContent =
    `${data.model || ''} · ${data.dataset || 'Edge-IIoTset'} · Seed ${data.random_seed ?? 42}`;

  // Mini metric cards
  const metrics = [
    { label: 'Accuracy',      val: data.accuracy  !== undefined ? (data.accuracy * 100).toFixed(2) + '%' : '—' },
    { label: 'Macro F1',      val: fmt(data.macro_f1) },
    { label: 'Precision',     val: fmt(data.precision) },
    { label: 'Recall',        val: fmt(data.recall) },
    { label: 'FPR',           val: data.fpr !== null && data.fpr !== undefined ? fmt(data.fpr) : '—' },
    { label: 'FNR',           val: data.fnr !== null && data.fnr !== undefined ? fmt(data.fnr) : '—' },
    { label: 'Train Time',    val: data.training_time_s ? data.training_time_s.toFixed(1) + 's' : '—' },
    { label: 'Latency',       val: data.inference_latency_ms ? data.inference_latency_ms.toFixed(4) + ' ms' : '—' },
  ];
  document.getElementById('detail-metrics').innerHTML = metrics.map(m => `
    <div class="detail-metric">
      <div class="detail-metric-val">${m.val}</div>
      <div class="detail-metric-lbl">${m.label}</div>
    </div>
  `).join('');

  // Feature Importance Chart
  const fiCard = document.getElementById('fi-card');
  if (data.feature_importance?.length) {
    fiCard.style.display = 'block';
    renderFeatureImportanceChart(data.feature_importance.slice(0, 15));
  } else {
    fiCard.style.display = 'none';
  }

  // Per-class F1 (only for multiclass)
  const perclassCard = document.getElementById('perclass-card');
  if (data.classification_report?.length) {
    const classRows = data.classification_report.filter(r => {
      const k = Object.keys(r)[0];
      return !isNaN(parseInt(r[k]));
    });
    if (classRows.length > 0) {
      perclassCard.style.display = 'block';
      renderPerClassChart(classRows);
    } else {
      perclassCard.style.display = 'none';
    }
  } else {
    perclassCard.style.display = 'none';
  }

  // Confusion Matrix
  const cmSection = document.getElementById('cm-section');
  const cmImg     = document.getElementById('cm-img');
  if (data.images?.includes('confusion_matrix.png')) {
    cmSection.style.display = 'block';
    cmImg.src = API.image(expId, 'confusion_matrix.png');
    cmImg.onclick = () => openModal(cmImg.src);
  } else {
    cmSection.style.display = 'none';
  }
}

// Close detail panel
document.getElementById('close-detail').addEventListener('click', () => {
  document.getElementById('detail-panel').style.display = 'none';
  document.querySelectorAll('#exp-tbody tr').forEach(r => r.classList.remove('active-row'));
});

// ──────────────────────────────────────────────────────────────────────────────
// Feature Importance Chart
// ──────────────────────────────────────────────────────────────────────────────

function renderFeatureImportanceChart(features) {
  destroyChart('fi');
  const labels = features.map(f => {
    // Clean up sklearn feature names like "cat__tcp.len_0.0"
    return f.feature.replace(/^(cat__|num__)/, '').replace(/_\d+\.\d+$/, '');
  });
  const values = features.map(f => parseFloat(f.importance) || 0);

  const ctx = document.getElementById('fi-chart').getContext('2d');
  charts.fi = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance',
        data: values,
        backgroundColor: values.map((_, i) =>
          `rgba(0,212,255,${0.7 - i * 0.03})`),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(5)}` },
        },
      },
      scales: {
        x: {
          ticks: { color: COLORS.textSec, font: { size: 10 } },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: { color: COLORS.textSec, font: { size: 10 }, autoSkip: false },
          grid:  { display: false },
        },
      },
    },
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-Class F1 Chart
// ──────────────────────────────────────────────────────────────────────────────

function renderPerClassChart(classRows) {
  destroyChart('perclass');

  // The first key in each row is the class index
  const labels = classRows.map(r => {
    const idx = Object.values(r)[0];
    return LABEL_MAP[idx] || `Class ${idx}`;
  });
  const f1Vals = classRows.map(r => parseFloat(r['f1-score']) || 0);

  const ctx = document.getElementById('perclass-chart').getContext('2d');
  charts.perclass = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'F1 Score',
        data: f1Vals,
        backgroundColor: f1Vals.map(v => f1Color(v) + 'BB'),
        borderColor:     f1Vals.map(v => f1Color(v)),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` F1: ${ctx.parsed.x.toFixed(4)}` } },
      },
      scales: {
        x: {
          min: 0, max: 1,
          ticks: { color: COLORS.textSec, font: { size: 10 } },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: { color: COLORS.textSec, font: { size: 10 }, autoSkip: false },
          grid:  { display: false },
        },
      },
    },
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Experiment Comparison Bar Chart
// ──────────────────────────────────────────────────────────────────────────────

function renderComparisonChart(experiments) {
  destroyChart('comparison');

  const completed = experiments.filter(e => e.status === 'complete' && e.macro_f1 !== null);
  if (completed.length === 0) return;

  const labels  = completed.map(e => e.experiment_id.split('_').slice(0, 2).join('_'));
  const accVals = completed.map(e => e.accuracy ? (e.accuracy * 100) : 0);
  const f1Vals  = completed.map(e => e.macro_f1 ? (e.macro_f1 * 100) : 0);

  const ctx = document.getElementById('comparison-chart').getContext('2d');
  charts.comparison = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Accuracy (%)',
          data: accVals,
          backgroundColor: 'rgba(0,212,255,0.6)',
          borderColor: COLORS.cyan,
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: 'Macro F1 (%)',
          data: f1Vals,
          backgroundColor: 'rgba(0,255,157,0.5)',
          borderColor: COLORS.green,
          borderWidth: 1,
          borderRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: COLORS.textSec, font: { family: 'Inter', size: 12 } },
        },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` },
        },
      },
      scales: {
        x: {
          ticks: { color: COLORS.textSec, font: { size: 11 } },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: COLORS.textSec, callback: v => v + '%' },
          grid:  { color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Attack Class F1 Chart
// ──────────────────────────────────────────────────────────────────────────────

async function renderAttackSection() {
  const data = await fetchJSON(API.experiment('E02_random_forest_multiclass'));
  if (!data?.classification_report) return;

  const rows = data.classification_report.filter(r => {
    const firstVal = Object.values(r)[0];
    return !isNaN(parseInt(firstVal));
  });

  const labels  = rows.map(r => LABEL_MAP[Object.values(r)[0]] || `Class ${Object.values(r)[0]}`);
  const f1Vals  = rows.map(r => parseFloat(r['f1-score']) || 0);
  const supportVals = rows.map(r => parseInt(r.support) || 0);

  // Bar chart
  const ctx = document.getElementById('attack-f1-chart').getContext('2d');
  destroyChart('attackF1');
  charts.attackF1 = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'F1 Score',
        data: f1Vals,
        backgroundColor: f1Vals.map(v => f1Color(v) + 'BB'),
        borderColor:     f1Vals.map(v => f1Color(v)),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` F1: ${ctx.parsed.x.toFixed(4)}`,
            afterLabel: ctx => ` Support: ${supportVals[ctx.dataIndex].toLocaleString()}`,
          },
        },
      },
      scales: {
        x: {
          min: 0, max: 1,
          ticks: { color: COLORS.textSec },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: { color: COLORS.textSec, font: { size: 11 }, autoSkip: false },
          grid:  { display: false },
        },
      },
    },
  });

  // Legend with colored dots
  const legendEl = document.getElementById('attack-legend');
  const sorted = labels.map((l, i) => ({ label: l, f1: f1Vals[i], support: supportVals[i] }))
                       .sort((a, b) => b.f1 - a.f1);

  legendEl.innerHTML = sorted.map(item => `
    <div class="attack-item">
      <div class="attack-dot" style="background:${f1Color(item.f1)}"></div>
      <span class="attack-name">${item.label}</span>
      <span class="attack-f1" style="color:${f1Color(item.f1)}">${item.f1.toFixed(3)}</span>
    </div>
  `).join('');
}

// ──────────────────────────────────────────────────────────────────────────────
// Dataset Audit Section
// ──────────────────────────────────────────────────────────────────────────────

async function renderAuditSection() {
  const data = await fetchJSON(API.audit);
  if (!data) return;

  const grid = document.getElementById('audit-grid');
  const cards = [];

  // Dataset Summary Card
  if (data.dataset_summary) {
    const s = data.dataset_summary;
    cards.push(`
      <div class="audit-card">
        <div class="audit-card-title">📂 Dataset Summary</div>
        ${[
          ['File',      s.filename],
          ['Source',    'Edge-IIoTset (Kaggle)'],
          ['Rows',      parseInt(s.num_rows).toLocaleString()],
          ['Raw Columns', s.num_columns],
          ['File Size', ((s.file_size_bytes || 0) / 1e9).toFixed(2) + ' GB'],
          ['SHA-256',   (s.sha256 || '').substring(0, 12) + '…'],
        ].map(([k, v]) => `
          <div class="audit-stat-row">
            <span class="audit-stat-key">${k}</span>
            <span class="audit-stat-val">${v ?? '—'}</span>
          </div>
        `).join('')}
      </div>
    `);
  }

  // Feature Policy Card
  cards.push(`
    <div class="audit-card">
      <div class="audit-card-title">🛡️ Feature Policy</div>
      ${[
        ['Original Features',  63],
        ['Retained (Operational)', data.retained_features?.length ?? '—'],
        ['Dropped (Leakage)',  data.dropped_features?.length ?? '—'],
        ['Policy',             'Operational — no IPs, ports, timestamps'],
        ['OHE Strategy',       'Chunk-based, high-cardinality capped'],
        ['Seed',               42],
      ].map(([k, v]) => `
        <div class="audit-stat-row">
          <span class="audit-stat-key">${k}</span>
          <span class="audit-stat-val">${v}</span>
        </div>
      `).join('')}
    </div>
  `);

  // Leakage Candidates Card
  if (data.leakage_candidates?.length) {
    const items = data.leakage_candidates.filter(l => l.trim()).slice(0, 12);
    cards.push(`
      <div class="audit-card">
        <div class="audit-card-title">⚠️ Dropped Leakage Features</div>
        <ul class="leakage-list">
          ${items.map(l => `<li class="leakage-item">${l.trim()}</li>`).join('')}
          ${data.leakage_candidates.length > 12 ? `<li class="leakage-item" style="color:var(--text-muted)">…and ${data.leakage_candidates.length - 12} more</li>` : ''}
        </ul>
      </div>
    `);
  }

  // Attack Classes Card
  const labelEntries = Object.entries(LABEL_MAP);
  if (labelEntries.length) {
    cards.push(`
      <div class="audit-card">
        <div class="audit-card-title">🎭 Attack Classes (${labelEntries.length})</div>
        ${labelEntries.map(([idx, name]) => `
          <div class="audit-stat-row">
            <span class="audit-stat-key">${name}</span>
            <span class="audit-stat-val">${idx}</span>
          </div>
        `).join('')}
      </div>
    `);
  }

  grid.innerHTML = cards.join('');
}

// ──────────────────────────────────────────────────────────────────────────────
// Modal (zoom-in confusion matrix)
// ──────────────────────────────────────────────────────────────────────────────

function openModal(src) {
  const overlay = document.getElementById('modal-overlay');
  document.getElementById('modal-img').src = src;
  overlay.style.display = 'flex';
}

document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) closeModal();
});
document.getElementById('modal-close').addEventListener('click', closeModal);
function closeModal() { document.getElementById('modal-overlay').style.display = 'none'; }

// ──────────────────────────────────────────────────────────────────────────────
// Main Init
// ──────────────────────────────────────────────────────────────────────────────

async function init() {
  // Load labels first (needed by charts)
  const labels = await fetchJSON(API.labels);
  if (labels) Object.assign(LABEL_MAP, labels);

  // Parallel fetch
  const [statusData, expData] = await Promise.all([
    fetchJSON(API.status),
    fetchJSON(API.experiments),
  ]);

  // Render pipeline
  if (statusData?.phases) renderPipeline(statusData.phases);

  // Render experiment sections
  const experiments = expData?.experiments ?? [];
  if (experiments.length) {
    renderMetrics(experiments);
    renderExperimentTable(experiments);
    renderComparisonChart(experiments);
  }

  // Render attack analysis and audit (async, don't block)
  renderAttackSection();
  renderAuditSection();

  // Animated hero counters
  initCounters();
}

document.addEventListener('DOMContentLoaded', init);
