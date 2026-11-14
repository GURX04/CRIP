/**
 * CRIP Live Dashboard — app.js
 *
 * Vanilla ES2020 — no build step, no framework.
 * Depends only on uPlot (vendored) loaded before this script.
 *
 * Architecture:
 *   SSEClient → EventBus → Components (TopBar, Charts, InjectPanel, Stopwatch, Alerts, RCA, Eval)
 */
'use strict';

// ============================================================
// Constants
// ============================================================
const SERVICES = ['AmazonApiGateway', 'EC2 Data Transfer', 'DynamoDB', 'CloudWatch Logs', 'S3 PUT'];
const SVC_SHORT = { 'AmazonApiGateway': 'API GW', 'EC2 Data Transfer': 'EC2', 'DynamoDB': 'DDB', 'CloudWatch Logs': 'CWL', 'S3 PUT': 'S3' };
const SVC_COLOR = { 'AmazonApiGateway': '#0072B2', 'EC2 Data Transfer': '#E69F00', 'DynamoDB': '#009E73', 'CloudWatch Logs': '#CC79A7', 'S3 PUT': '#56B4E9' };
const MAX_POINTS = 720;   // per series in main chart
const CHART_HEIGHT_MAIN = 180;
const CHART_HEIGHT_SMALL = 100;

// ============================================================
// Utility helpers
// ============================================================
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const fmt2 = (n) => typeof n === 'number' ? n.toFixed(2) : '—';
const fmtMs = (ms) => ms < 1000 ? ms + 'ms' : (ms/1000).toFixed(1) + 's';
const fmtSimTs = (iso) => iso ? iso.slice(11, 16) : '—';
const isoToEpoch = (iso) => iso ? new Date(iso).getTime() / 1000 : 0;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function showToast(msg, type = 'info', durationMs = 4000) {
  const wrap = $('#toast-container');
  const div = document.createElement('div');
  div.className = `toast ${type}`;
  div.textContent = msg;
  wrap.appendChild(div);
  setTimeout(() => div.remove(), durationMs);
}

// ============================================================
// EventBus (simple pub/sub)
// ============================================================
class EventBus {
  constructor() { this._h = {}; }
  on(ev, fn) { (this._h[ev] = this._h[ev] || []).push(fn); }
  emit(ev, data) { (this._h[ev] || []).forEach(fn => fn(data)); }
}
const bus = new EventBus();

// ============================================================
// SSE Client (auto-reconnect with backoff)
// ============================================================
class SSEClient {
  constructor(url) {
    this._url = url;
    this._backoff = 1000;
    this._es = null;
    this._connect();
  }
  _connect() {
    if (this._es) this._es.close();
    this._es = new EventSource(this._url);
    this._es.onopen = () => {
      this._backoff = 1000;
      $('#disconnected-banner').classList.remove('visible');
      bus.emit('connected', {});
    };
    this._es.onerror = () => {
      $('#disconnected-banner').classList.add('visible');
      this._es.close();
      setTimeout(() => this._connect(), this._backoff);
      this._backoff = Math.min(this._backoff * 2, 30000);
    };
    ['tick','detection','injection','eval','state','heartbeat'].forEach(ev => {
      this._es.addEventListener(ev, e => {
        try { bus.emit(ev, JSON.parse(e.data)); }
        catch(err) { console.warn('SSE parse error', err); }
      });
    });
  }
}

// ============================================================
// Data store (rolling window per service)
// ============================================================
class DataStore {
  constructor() {
    this.reset();
  }
  reset() {
    this.runId = null;
    // per service: arrays of [epoch_s, cost, expected_cost]
    this.ticks = {};
    this.detections = [];
    this.injections = {};
    SERVICES.forEach(s => { this.ticks[s] = { ts: [], cost: [], exp: [] }; });
  }
  addTick(d) {
    const svc = d.service;
    if (!this.ticks[svc]) return;
    const t = this.ticks[svc];
    const ep = isoToEpoch(d.sim_ts);
    t.ts.push(ep);
    t.cost.push(d.cost ?? d.unblended_cost ?? 0);
    t.exp.push(d.expected_cost ?? 0);
    if (t.ts.length > MAX_POINTS) {
      t.ts.shift(); t.cost.shift(); t.exp.shift();
    }
  }
  addDetection(d) {
    this.detections.push(d);
    if (this.detections.length > 200) this.detections.shift();
  }
  addInjection(d) {
    this.injections[d.injection_id] = d;
  }
  totalCostSeries() {
    let refSvc = SERVICES[0];
    for (const s of SERVICES) {
      if ((this.ticks[s]?.ts?.length || 0) > (this.ticks[refSvc]?.ts?.length || 0)) refSvc = s;
    }
    const ref = this.ticks[refSvc];
    if (!ref || !ref.ts || ref.ts.length === 0) return { ts: [], cost: [], exp: [] };
    const n = ref.ts.length;
    const costs = new Float64Array(n);
    const exps  = new Float64Array(n);
    for (const s of SERVICES) {
      const t = this.ticks[s];
      if (!t || !t.ts) continue;
      const off = n - t.ts.length;
      for (let i = 0; i < t.ts.length; i++) {
        costs[off + i] += t.cost[i] || 0;
        exps[off + i]  += t.exp[i]  || 0;
      }
    }
    return {
      ts:   ref.ts,
      cost: Array.from(costs),
      exp:  Array.from(exps),
    };
  }
}
const store = new DataStore();

// ============================================================
// TopBar component
// ============================================================
class TopBar {
  constructor() {
    this._simTs = '—';
    this._runId = '—';
    this._paused = false;
    this._speed  = 5;
    this._bindControls();
    this._startHealthPolling();
  }
  _bindControls() {
    $('#btn-pause').addEventListener('click', async () => {
      const ep = this._paused ? '/api/control/resume' : '/api/control/pause';
      await apiFetch('POST', ep, {});
      this._paused = !this._paused;
      $('#btn-pause').textContent = this._paused ? '▶ Resume' : '⏸ Pause';
    });
    $('#speed-select').addEventListener('change', async (e) => {
      const v = parseFloat(e.target.value);
      await apiFetch('POST', '/api/control/speed', { sim_minutes_per_real_second: v });
      this._speed = v;
      showToast(`Speed set to ${v}x`, 'info');
    });
    $('#btn-reset').addEventListener('click', async () => {
      if (!confirm('Reset generator? This clears all state and starts a new run.')) return;
      const res = await apiFetch('POST', '/api/control/reset', {});
      if (res) showToast(`New run: ${res.run_id}`, 'success');
    });
    // Presentation mode
    document.addEventListener('keydown', e => {
      if (e.key === 'f' || e.key === 'F') {
        document.body.classList.toggle('present');
        showToast(document.body.classList.contains('present') ? 'Presentation mode ON' : 'Presentation mode OFF', 'info', 2000);
      }
    });
  }
  update(d) {
    if (d.sim_ts)  { this._simTs = d.sim_ts; $('#sim-time').textContent = 'Sim: ' + fmtSimTs(d.sim_ts); }
    if (d.run_id)  { this._runId = d.run_id; $('#run-id-label').textContent = 'Run: ' + d.run_id.slice(0,8); }
    if (typeof d.paused !== 'undefined') {
      this._paused = d.paused;
      $('#btn-pause').textContent = d.paused ? '▶ Resume' : '⏸ Pause';
    }
    if (d.sim_minutes_per_real_second) {
      this._speed = d.sim_minutes_per_real_second;
      $('#speed-select').value = d.sim_minutes_per_real_second;
    }
  }
  setChipStatus(id, ok) {
    const el = $('#chip-' + id);
    if (!el) return;
    el.className = 'chip ' + (ok ? 'ok' : 'err');
  }
  async _startHealthPolling() {
    const poll = async () => {
      try {
        const h = await apiFetch('GET', '/health', null);
        if (h) {
          this.setChipStatus('gen', h.generator?.ok);
          this.setChipStatus('det', h.detector?.ok);
          this.setChipStatus('redis', h.redis?.ok);
        }
      } catch(e) {}
      setTimeout(poll, 5000);
    };
    setTimeout(poll, 1000);
  }
}

// ============================================================
// Chart manager (uPlot)
// ============================================================
class ChartManager {
  constructor() {
    this._mainPlot = null;
    this._svcPlots = {};
    this._showGT   = true;    // show ground-truth injection regions
    this._markers  = [];      // recent anomaly markers (drawn on canvas)
    this._regions  = [];      // ground-truth injection regions (drawn on canvas)

    // Rendering throttle state
    this._dirtyServices = new Set();
    this._dirtyTotal = false;
    this._renderPending = false;

    this._initMain();
    this._initSmall();

    bus.on('tick', d => this._onTick(d));
    bus.on('detection', d => this._onDetection(d));
    bus.on('injection', d => this._onInjection(d));

    $('#gt-toggle').addEventListener('change', e => {
      this._showGT = e.target.checked;
      if (this._mainPlot) {
        this._mainPlot.setRegions(this._showGT ? this._regions : []);
      }
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => this._redrawAll(), 150);
    });
  }

  _uplotOpts(title, seriesLabels, colors, height, container) {
    return {
      title,
      width:  container.clientWidth || 600,
      height: height,
      cursor: { show: true },
      legend: { show: seriesLabels.length > 1 },
      axes: [
        { stroke: '#8b949e', grid: { stroke: '#21262d' },
          values: (u, vs) => vs.map(v => v ? new Date(v*1000).toISOString().slice(11,16) : '') },
        { stroke: '#8b949e', grid: { stroke: '#21262d' },
          values: (u, vs) => vs.map(v => v != null ? '$' + v.toFixed(4) : '') },
      ],
      series: [
        {},  // x (time)
        ...seriesLabels.map((label, i) => ({
          label, stroke: colors[i], width: 2, fill: i === 1 ? 'rgba(88,166,255,0.1)' : undefined,
        })),
      ],
    };
  }

  _initMain() {
    const wrap = $('#main-chart-wrap');
    if (!wrap || typeof uPlot === 'undefined') return;
    const opts = this._uplotOpts('Total Cost', ['Actual', 'Expected'], ['#F0E442', '#58a6ff'], CHART_HEIGHT_MAIN, wrap);
    this._mainPlot = new uPlot(opts, [[], [], []], wrap);
  }

  _initSmall() {
    SERVICES.forEach((svc, i) => {
      const wrap = $('#chart-' + i);
      if (!wrap || typeof uPlot === 'undefined') return;
      const color = SVC_COLOR[svc] || '#58a6ff';
      const opts = this._uplotOpts(SVC_SHORT[svc], ['Actual', 'Expected'], [color, '#58a6ff'], CHART_HEIGHT_SMALL, wrap);
      this._svcPlots[svc] = new uPlot(opts, [[], [], []], wrap);
    });
  }

  _onTick(d) {
    store.addTick(d);
    this._dirtyServices.add(d.service);
    this._dirtyTotal = true;
    this._scheduleRender();
  }

  _onDetection(d) {
    store.addDetection(d);
    const ep = isoToEpoch(d.sim_ts);
    const sev = d.severity || 50;
    const color = sev > 70 ? '#FF5252' : sev > 40 ? '#FFB300' : '#8b949e';
    this._markers.push({ ts: ep, color });
    if (this._markers.length > 80) this._markers.shift();

    this._dirtyTotal = true;
    this._scheduleRender();
  }

  _onInjection(d) {
    store.addInjection(d);
    const start = isoToEpoch(d.sim_start_ts);
    const end = d.sim_end_ts ? isoToEpoch(d.sim_end_ts) : null;
    const color = SVC_COLOR[d.service] || '#CC79A7';
    this._regions.push({ start, end, color });
    if (this._regions.length > 30) this._regions.shift();

    this._dirtyTotal = true;
    this._scheduleRender();
  }

  _scheduleRender() {
    if (this._renderPending) return;
    this._renderPending = true;
    requestAnimationFrame(() => {
      this._renderPending = false;
      this._flushRender();
    });
  }

  _flushRender() {
    // 1. Redraw only modified service small plots (at most once per frame)
    for (const svc of this._dirtyServices) {
      const p = this._svcPlots[svc];
      if (p) {
        const t = store.ticks[svc];
        p.setData([t.ts, t.cost, t.exp]);
      }
    }
    this._dirtyServices.clear();

    // 2. Redraw main plot with updated series + canvas markers/regions (at most once per frame)
    if (this._dirtyTotal && this._mainPlot) {
      this._dirtyTotal = false;
      const tot = store.totalCostSeries();
      this._mainPlot.setData([tot.ts, tot.cost, tot.exp]);
      this._mainPlot.setMarkers(this._markers);
      if (this._showGT) {
        this._mainPlot.setRegions(this._regions);
      }
    }
  }

  _redrawAll() {
    if (this._mainPlot) {
      const wrap = $('#main-chart-wrap');
      if (wrap) this._mainPlot.setSize({ width: wrap.clientWidth, height: CHART_HEIGHT_MAIN });
    }
    SERVICES.forEach((svc, i) => {
      const p = this._svcPlots[svc];
      const wrap = $('#chart-' + i);
      if (p && wrap) p.setSize({ width: wrap.clientWidth, height: CHART_HEIGHT_SMALL });
    });
  }

  loadHistory(h) {
    if (!h) return;
    h.ticks.forEach(t => store.addTick(t));
    h.detections.forEach(d => {
      store.addDetection(d);
      const ep = isoToEpoch(d.sim_ts);
      const sev = d.severity || 50;
      const color = sev > 70 ? '#FF5252' : sev > 40 ? '#FFB300' : '#8b949e';
      this._markers.push({ ts: ep, color });
    });
    h.injections.forEach(inj => {
      store.addInjection(inj);
      const start = isoToEpoch(inj.sim_start_ts);
      const end = inj.sim_end_ts ? isoToEpoch(inj.sim_end_ts) : null;
      const color = SVC_COLOR[inj.service] || '#CC79A7';
      this._regions.push({ start, end, color });
    });
    this._redrawAll();
    this._flushRender();
  }
}

// ============================================================
// InjectPanel
// ============================================================
class InjectPanel {
  constructor() {
    this._activeInjections = {};
    this._bindForm();
    this._bindPresets();
    this._pollActiveInjections();
    bus.on('injection', d => { this._activeInjections[d.injection_id] = d; this._renderList(); });
  }
  _bindForm() {
    ['mag-slider','dur-slider','ramp-slider'].forEach(id => {
      const el = $('#' + id);
      if (!el) return;
      el.addEventListener('input', () => {
        const out = $('#' + id + '-val');
        if (out) out.textContent = el.value + (id==='mag-slider' ? 'x' : 'm');
      });
    });
    $('#inject-btn').addEventListener('click', () => this._doInject());
  }
  _bindPresets() {
    const presets = [
      { label: '⚡ DDB Spike ×4', p: { type: 'spike', service: 'DynamoDB', magnitude: 4, duration_min: 30 } },
      { label: '💰 Price Hike',   p: { type: 'cost_only', service: 'DynamoDB', magnitude: 5, duration_min: 20 } },
      { label: '🌊 Slow Leak',    p: { type: 'slow_drift', service: 'AmazonApiGateway', magnitude: 30, duration_min: 60 } },
      { label: '🔥 All ×2',       p: { type: 'spike', service: 'all', magnitude: 2, duration_min: 15 } },
    ];
    $$('.preset-btn').forEach((btn, i) => {
      if (presets[i]) {
        btn.textContent = presets[i].label;
        btn.addEventListener('click', () => this._applyPreset(presets[i].p));
      }
    });
  }
  _applyPreset(p) {
    if ($('#inj-type'))  $('#inj-type').value  = p.type;
    if ($('#inj-svc'))   $('#inj-svc').value   = p.service;
    if ($('#mag-slider')) { $('#mag-slider').value = p.magnitude; $('#mag-slider-val').textContent = p.magnitude + 'x'; }
    if ($('#dur-slider')) { $('#dur-slider').value = p.duration_min; $('#dur-slider-val').textContent = p.duration_min + 'm'; }
    this._doInject(p);
  }
  async _doInject(override) {
    const body = override || {
      type:        $('#inj-type')?.value  || 'spike',
      service:     $('#inj-svc')?.value   || 'DynamoDB',
      magnitude:   parseFloat($('#mag-slider')?.value || 2),
      duration_min:parseFloat($('#dur-slider')?.value || 30),
      ramp_min:    parseFloat($('#ramp-slider')?.value || 0),
    };
    const res = await apiFetch('POST', '/api/inject', body);
    if (res?.injection_id) {
      showToast(`Injected: ${body.type} on ${body.service}`, 'success');
      bus.emit('inject_started', { ...body, ...res, wall_start_ms: Date.now() });
    }
  }
  async _pollActiveInjections() {
    const res = await apiFetch('GET', '/api/injections', null);
    if (res) {
      res.forEach(inj => { if (inj.status !== 'ended') this._activeInjections[inj.injection_id] = inj; });
      this._renderList();
    }
    setTimeout(() => this._pollActiveInjections(), 5000);
  }
  _renderList() {
    const list = $('#active-injections');
    if (!list) return;
    const active = Object.values(this._activeInjections).filter(i => i.status !== 'ended');
    list.innerHTML = active.length === 0
      ? '<div style="font-size:0.8em;color:var(--text-muted)">No active injections</div>'
      : active.map(i => `
          <div class="inj-item">
            <span>${i.type} ${SVC_SHORT[i.service]||i.service} ×${i.magnitude}</span>
            <button class="inj-end-btn" onclick="injectPanel.endInjection('${i.injection_id}')">End</button>
          </div>`).join('');
  }
  async endInjection(id) {
    await apiFetch('DELETE', '/api/inject/' + id, null);
    if (this._activeInjections[id]) this._activeInjections[id].status = 'ended';
    this._renderList();
    showToast('Injection ended', 'info');
  }
}

// ============================================================
// Latency Stopwatch
// ============================================================
class StopwatchPanel {
  constructor() {
    this._timer  = null;
    this._start  = null;
    this._active = null;   // { injection_id, service, wall_start_ms }
    this._history = [];
    this._timeout = 60000; // 60 real seconds
    bus.on('inject_started', d => this._onInjectStart(d));
    bus.on('detection',      d => this._onDetection(d));
  }
  _startTimer() {
    this._stopTimer();
    this._timer = setInterval(() => this._tick(), 100);
  }
  _stopTimer() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }
  _onInjectStart(d) {
    this._active = d;
    this._start  = d.wall_start_ms || Date.now();
    this._setDisplay(0, 'running');
    this._startTimer();
    // Auto-miss after timeout
    clearTimeout(this._missTimer);
    this._missTimer = setTimeout(() => {
      if (this._active && this._active.injection_id === d.injection_id) {
        this._stopTimer();
        this._setDisplay(null, 'missed');
        this._history.push({ ...d, detected: false, latency_wall_s: null, first_detector: '—' });
        this._active = null;
        evalPanel.updateHistory(this._history);
      }
    }, this._timeout);
  }
  _onDetection(det) {
    if (!this._active) return;
    // Check service match (or "all")
    if (det.service !== this._active.service && this._active.service !== 'all') return;
    this._stopTimer();
    const latency = (det.detect_wall_ts_ms - this._start) / 1000;
    const simMin  = latency * (parseFloat($('#speed-select')?.value || 5)) / 60;
    const firstDet = det.detector_votes?.find(v => v.is_anomaly)?.detector_name || '—';
    this._setDisplay(latency, 'frozen', simMin, firstDet, det.rca);
    clearTimeout(this._missTimer);
    this._history.push({
      ...this._active,
      detected: true,
      latency_wall_s: latency,
      latency_sim_min: simMin,
      first_detector: firstDet,
    });
    this._active = null;
    evalPanel.updateHistory(this._history);
  }
  _tick() {
    if (!this._active || !this._start) return;
    const elapsed = (Date.now() - this._start) / 1000;
    this._setDisplay(elapsed, 'running');
  }
  _setDisplay(seconds, state, simMin, detector, rca) {
    const timeEl = $('#sw-time');
    const detEl  = $('#sw-detector');
    const detailEl = $('#sw-detail');
    if (!timeEl) return;
    if (state === 'running') {
      timeEl.textContent = seconds !== null ? seconds.toFixed(1) + 's' : '—';
      timeEl.className = 'sw-time';
      if (detEl) detEl.textContent = '';
      if (detailEl) detailEl.textContent = 'Waiting for detection…';
    } else if (state === 'frozen') {
      timeEl.textContent = seconds.toFixed(1) + 's';
      timeEl.className = 'sw-time frozen';
      if (detailEl) detailEl.textContent = `≈ ${(simMin||0).toFixed(0)} sim-min`;
      if (detEl) detEl.textContent = 'by: ' + (detector || '—');
      if (rca) rcaPanel.show(rca);
    } else if (state === 'missed') {
      timeEl.textContent = 'MISSED';
      timeEl.className = 'sw-time missed';
      if (detailEl) detailEl.textContent = 'No detection within 60s';
      if (detEl) detEl.textContent = '';
    }
  }
}

// ============================================================
// Alert Feed
// ============================================================
class AlertFeed {
  constructor() {
    this._detections = [];
    this._renderPending = false;
    bus.on('detection', d => this._add(d));
  }
  _add(d) {
    this._detections.unshift(d);
    if (this._detections.length > 50) this._detections.pop();
    this._scheduleRender();
  }
  _scheduleRender() {
    if (this._renderPending) return;
    this._renderPending = true;
    requestAnimationFrame(() => {
      this._renderPending = false;
      this._render();
    });
  }
  _render() {
    const feed = $('#alert-feed');
    if (!feed) return;
    if (this._detections.length === 0) {
      feed.innerHTML = '<div style="font-size:0.8em;color:var(--text-muted);padding:8px 0">No alerts detected yet</div>';
      return;
    }
    feed.innerHTML = this._detections.slice(0, 20).map((d, i) => {
      const sev = d.severity || 0;
      const cls = sev > 70 ? 'sev-high' : sev > 40 ? 'sev-medium' : 'sev-low';
      const votes = (d.detector_votes || []).filter(v => v.is_anomaly).map(v => v.detector_name).join(', ');
      return `<div class="alert-item" onclick="alertFeed.clickAlert(${i})">
        <span class="sev-badge ${cls}">${sev.toFixed(0)}</span>
        <div>
          <div class="alert-svc">${d.service} <span class="alert-status-${d.status}">[${d.status}]</span></div>
          <div class="alert-meta">${fmtSimTs(d.sim_ts)} · ${votes || 'unknown'}</div>
        </div>
      </div>`;
    }).join('');
  }
  clickAlert(idx) {
    const d = this._detections[idx];
    if (d && d.rca) rcaPanel.show(d.rca);
  }
  loadHistory(detections) {
    this._detections = [...detections].reverse();
    this._scheduleRender();
  }
}

// ============================================================
// RCA Panel
// ============================================================
class RCAPanel {
  show(rca) {
    const panel = $('#rca-panel');
    if (!panel || !rca) return;
    panel.classList.add('visible');

    const classifications = {
      traffic_driven:        'Cost and volume both rose — this is real demand increase, not a configuration change.',
      price_driven:          'Cost rose but request volume did not. Likely a pricing or configuration change — not real demand.',
      volume_drop_cost_flat: 'Request volume fell but cost stayed flat — a potential zombie resource or billing anomaly.',
      unknown:               'Could not classify the anomaly driver confidently.',
    };
    const cls = rca.alert_type || rca.classification || 'unknown';
    $('#rca-classification').textContent = classifications[cls] || cls;
    $('#rca-type-label').textContent = cls.replace(/_/g, ' ').toUpperCase();

    // Service contribution bars
    const contrib = rca.service_contributions || rca.service_contribution || {};
    const bars = $('#rca-bars');
    if (bars) {
      bars.innerHTML = Object.entries(contrib).map(([svc, frac]) => `
        <div class="rca-bar-row">
          <span class="rca-bar-label">${SVC_SHORT[svc]||svc}</span>
          <div class="rca-bar-track"><div class="rca-bar-fill" style="width:${(frac*100).toFixed(1)}%"></div></div>
          <span class="rca-bar-val">${(frac*100).toFixed(1)}%</span>
        </div>`).join('');
    }

    // Metrics
    const now1k = rca.cost_per_1k_now || rca.cost_per_1k_requests_now || 0;
    const base1k = rca.cost_per_1k_baseline || 0;
    const pctCost = rca.pct_cost_change || 0;
    const pctVol  = rca.pct_volume_change || 0;
    if ($('#rca-cost-now'))    $('#rca-cost-now').textContent   = '$' + now1k.toFixed(4);
    if ($('#rca-cost-base'))   $('#rca-cost-base').textContent  = '$' + base1k.toFixed(4);
    if ($('#rca-pct-cost'))    $('#rca-pct-cost').textContent   = (pctCost >= 0 ? '+' : '') + pctCost.toFixed(1) + '%';
    if ($('#rca-pct-volume'))  $('#rca-pct-volume').textContent = (pctVol >= 0 ? '+' : '') + pctVol.toFixed(1) + '%';
  }
}

// ============================================================
// Eval Table
// ============================================================
class EvalPanel {
  constructor() {
    this._history = [];
    this._evalData = { precision: null, recall: null, f1: null, false_alarms_per_day: null };
    this._renderPending = false;
    bus.on('eval', d => this._onEval(d));
  }
  _onEval(d) {
    this._evalData = d;
    this._scheduleRender();
  }
  updateHistory(h) {
    this._history = h.slice(-10);
    this._scheduleRender();
  }
  _scheduleRender() {
    if (this._renderPending) return;
    this._renderPending = true;
    requestAnimationFrame(() => {
      this._renderPending = false;
      this._render();
    });
  }
  _render() {
    const d = this._evalData;
    const p = d.precision != null ? (d.precision*100).toFixed(1) + '%' : '—';
    const r = d.recall    != null ? (d.recall*100).toFixed(1) + '%' : '—';
    const f = d.f1        != null ? (d.f1*100).toFixed(1) + '%' : '—';
    const fa = d.false_alarms_per_day != null ? d.false_alarms_per_day.toFixed(2) : '—';
    const summary = $('#eval-summary');
    if (summary) summary.innerHTML = `
      <div class="eval-metric">P <span>${p}</span></div>
      <div class="eval-metric">R <span>${r}</span></div>
      <div class="eval-metric">F1 <span>${f}</span></div>
      <div class="eval-metric">FA/day <span>${fa}</span></div>`;

    const tbody = $('#eval-tbody');
    if (!tbody) return;
    tbody.innerHTML = this._history.slice().reverse().map(h => `
      <tr>
        <td>${(h.injection_id||'').slice(0,8)}</td>
        <td>${h.type||'—'}</td>
        <td>${SVC_SHORT[h.service]||h.service||'—'}</td>
        <td>${h.latency_wall_s != null ? h.latency_wall_s.toFixed(1)+'s' : '—'}</td>
        <td>${h.first_detector||'—'}</td>
        <td class="${h.detected ? 'badge-ok' : 'badge-miss'}">${h.detected ? '✓' : '✗'}</td>
      </tr>`).join('');
  }
}

// ============================================================
// API helper
// ============================================================
async function apiFetch(method, path, body) {
  try {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
      const err = await res.text().catch(() => res.statusText);
      showToast('API error: ' + err, 'error');
      return null;
    }
    return method !== 'DELETE' ? res.json() : null;
  } catch(e) {
    showToast('Network error: ' + e.message, 'error');
    return null;
  }
}

// ============================================================
// Replay mode detection
// ============================================================
function checkReplayMode() {
  const params = new URLSearchParams(location.search);
  if (params.has('replay')) {
    $('#replay-banner').classList.add('visible');
    $('#replay-banner').textContent = '⏺ REPLAY MODE — ' + (params.get('replay') || 'recorded session');
    document.title = '[REPLAY] ' + document.title;
  }
}

// ============================================================
// Bootstrap
// ============================================================
let injectPanel, stopwatchPanel, alertFeed, rcaPanel, evalPanel, chartManager, topBar;

document.addEventListener('DOMContentLoaded', async () => {
  checkReplayMode();

  topBar       = new TopBar();
  chartManager = new ChartManager();
  injectPanel  = new InjectPanel();
  stopwatchPanel = new StopwatchPanel();
  alertFeed    = new AlertFeed();
  rcaPanel     = new RCAPanel();
  evalPanel    = new EvalPanel();

  // Load history for chart restore
  const hist = await apiFetch('GET', '/history?minutes=60', null);
  if (hist) {
    chartManager.loadHistory(hist);
    alertFeed.loadHistory(hist.detections || []);
  }

  // Wire SSE bus to components
  bus.on('state', d => {
    topBar.update(d);
    if (d.new_run) {
      store.reset();
      showToast('New generator run started (run_id: ' + (d.run_id||'').slice(0,8) + ')', 'info', 6000);
    }
    if (d.replay) {
      $('#replay-banner').classList.add('visible');
    }
  });
  bus.on('tick', d => {}); // chart manager handles via bus.on('tick')

  // Start SSE
  new SSEClient('/stream');
});

