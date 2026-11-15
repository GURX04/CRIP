'use strict';

// ---------------------------------------------------------------------------
// Helpers & Constants
// ---------------------------------------------------------------------------
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

const SV = ['AmazonApiGateway', 'EC2 Data Transfer', 'DynamoDB', 'CloudWatch Logs', 'S3 PUT'];
const LB = {
  AmazonApiGateway: 'API GW',
  'EC2 Data Transfer': 'EC2',
  DynamoDB: 'DynamoDB',
  'CloudWatch Logs': 'CW Logs',
  'S3 PUT': 'S3'
};

const PRE = [
  { type: 'spike', service: 'DynamoDB', magnitude: 4, duration_min: 30, ramp_min: 0 },
  { type: 'cost_only', service: 'DynamoDB', magnitude: 5, duration_min: 20, ramp_min: 0 },
  { type: 'slow_drift', service: 'DynamoDB', magnitude: 3, duration_min: 60, ramp_min: 0 },
  { type: 'spike', service: 'all', magnitude: 2, duration_min: 30, ramp_min: 0 }
];

const S = {
  tk: {},
  tot: new Map(),
  det: new Map(),
  inj: new Map(),
  prop: new Map(),
  run: '',
  paused: false,
  range: 60,
  gt: false,
  sel: null,
  watch: null,
  last: null,
  saved: 0,
  results: 0,
  mode: 'DRY_RUN',
  kill: false,
  fs: 'all',
  fv: 'all',
  pf: 'Pending',
  ev: {},
  now: 0
};
SV.forEach(s => { S.tk[s] = []; });

const T = i => Date.parse(i) || 0;
const hm = t => new Date(t).toISOString().slice(11, 16);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
}[c]));

const $$$ = (o, k) => {
  for (const x of k) {
    if (o && o[x] != null) return o[x];
  }
};

const pct = v => v == null ? '—' : ((v <= 1 ? v * 100 : v).toFixed(1) + '%');
const usd = v => v == null || isNaN(v) ? '—' : '$' + (+v).toFixed(v < 1 ? 4 : 2);

const api = (m, u, b) => fetch(u, {
  method: m,
  headers: { 'Content-Type': 'application/json' },
  body: b ? JSON.stringify(b) : undefined
}).then(async r => {
  if (!r.ok) throw new Error((await r.text()).slice(0, 140));
  return r.json().catch(() => ({}));
});

const toast = t => {
  const e = $('#toast');
  if (!e) return;
  e.textContent = t;
  e.style.display = 'block';
  clearTimeout(toast.h);
  toast.h = setTimeout(() => { e.style.display = 'none'; }, 3500);
};

const act = (p, ok) => p.then(ok).catch(e => toast('Failed: ' + e.message));
const isPend = p => /pend|propos/i.test(p.status || '');
const isApp = p => /appl|approv|mitig/i.test(p.status || '');

const names = d => {
  const v = d.detector_votes;
  if (Array.isArray(v)) {
    return v.filter(x => x.is_anomaly || x.vote || x.fired)
      .map(x => x.detector_name || x.detector || x.name)
      .filter(Boolean);
  }
  return v && typeof v === 'object' ? Object.keys(v).filter(k => v[k]) : [];
};

let dirty = false;
const draw = () => {
  if (!dirty) {
    dirty = true;
    requestAnimationFrame(() => {
      dirty = false;
      render();
    });
  }
};

// ---------------------------------------------------------------------------
// Ingest & State Management
// ---------------------------------------------------------------------------
function clearAll() {
  SV.forEach(s => { S.tk[s] = []; });
  S.tot.clear();
  S.det.clear();
  S.inj.clear();
  S.prop.clear();
  S.watch = null;
  S.last = null;
}

function addTick(d) {
  if (!SV.includes(d.service)) return;
  if (S.run && d.run_id && d.run_id !== S.run) {
    clearAll();
    toast('New run started');
  }
  if (d.run_id) S.run = d.run_id;

  const t = T(d.sim_ts);
  const c = +d.cost || 0;
  const e = +(d.expected_cost ?? d.cost) || 0;
  const a = S.tk[d.service];

  a.push({ t, c, e });
  if (a.length > 1500) a.shift();

  const o = S.tot.get(t) || { c: 0, e: 0 };
  o.c += c;
  o.e += e;
  S.tot.set(t, o);
  if (S.tot.size > 1500) S.tot.delete(S.tot.keys().next().value);
  S.now = Math.max(S.now, t);
}

function addDet(d) {
  const id = d.detection_id;
  const t = T(d.sim_ts);
  const o = S.det.get(id);

  if (o) {
    o.end = Math.max(o.end, t);
    o.status = d.status || o.status;
    o.sev = d.severity ?? o.sev;
    o.rca = d.rca || o.rca;
  } else {
    const r = {
      id,
      svc: d.service,
      start: t,
      end: t,
      sev: d.severity,
      votes: names(d),
      rca: d.rca || {},
      status: d.status || 'open'
    };
    S.det.set(id, r);

    const w = S.watch;
    if (w && (w.svc === d.service || w.svc === 'all') && !S.last) {
      S.last = {
        lat: (Date.now() - w.t0) / 1000,
        svc: d.service,
        by: r.votes.join(' + ') || 'ensemble'
      };
      S.watch = null;
    }
  }
}

function addProp(p) {
  if (p && p.proposal_id) {
    S.prop.set(p.proposal_id, { ...(S.prop.get(p.proposal_id) || {}), ...p });
  }
}

function addInj(d) {
  if (d.injection_id) {
    S.inj.set(d.injection_id, { ...(S.inj.get(d.injection_id) || {}), ...d });
  }
}

function onState(d) {
  if (d.run_id && S.run && d.run_id !== S.run && d.new_run) {
    clearAll();
  }
  if (d.run_id) S.run = d.run_id;
  if (d.paused != null) S.paused = !!d.paused;
  (d.active_injections || d.injections || []).forEach(addInj);
}

// ---------------------------------------------------------------------------
// SVG Charts
// ---------------------------------------------------------------------------
function lineChart() {
  const all = Array.from(S.tot).slice(-Math.max(3, Math.round(S.range / 5)));
  if (all.length < 2) {
    return '<div class="empty">Waiting for data. Start the generator and detector.</div>';
  }

  const W = 900;
  const H = 300;
  const P = 250;
  const t0 = all[0][0];
  const t1 = all.at(-1)[0] || t0 + 1;
  const mx = Math.max(...all.map(([, v]) => Math.max(v.c, v.e))) * 1.1 || 1;

  const X = t => (t - t0) / (t1 - t0 || 1) * W;
  const Y = v => P - (v / mx) * (P - 20);
  const pl = k => all.map(([t, v], i) => (i ? 'L' : 'M') + X(t).toFixed(1) + ',' + Y(v[k]).toFixed(1)).join('');

  let g = '';
  for (let i = 0; i < 4; i++) {
    g += `<line class="gl" x1="0" x2="${W}" y1="${40 + i * 70}" y2="${40 + i * 70}"/>`;
  }

  if (S.gt) {
    S.inj.forEach(j => {
      const a = T(j.sim_start_ts || j.start_sim_ts);
      const b = T(j.sim_end_ts || j.end_sim_ts) || t1;
      if (a && b > t0 && a < t1) {
        g += `<rect x="${X(Math.max(a, t0))}" y="0" width="${Math.max(2, X(Math.min(b, t1)) - X(Math.max(a, t0)))}" height="${P}" fill="var(--crit)" opacity=".1"/>`;
      }
    });
  }

  g += `<path d="${pl('e')}" fill="none" stroke="var(--c1)" stroke-opacity=".18" stroke-width="30"/>
        <path d="${pl('e')}" fill="none" stroke="var(--mut)" stroke-width="2" stroke-dasharray="6 5"/>
        <path d="${pl('c')}" fill="none" stroke="var(--c1)" stroke-width="3"/>`;

  S.det.forEach(d => {
    if (d.start < t0 || d.start > t1) return;
    const n = all.reduce((a, b) => Math.abs(b[0] - d.start) < Math.abs(a[0] - d.start) ? b : a);
    g += `<circle cx="${X(n[0])}" cy="${Y(n[1].c)}" r="7" fill="var(--crit)" stroke="var(--surf)" stroke-width="2">
            <title>${esc(LB[d.svc] || d.svc)} · severity ${Math.round(d.sev || 0)}</title>
          </circle>`;
  });

  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Total cost over time">
    ${g}
    <text x="0" y="284" class="ax">${hm(t0)}</text>
    <text x="${W / 2 - 20}" y="284" class="ax">${hm((t0 + t1) / 2)}</text>
    <text x="${W - 40}" y="284" class="ax">${hm(t1)}</text>
  </svg>`;
}

function stackChart() {
  const n = 6;
  const cols = ['--c1', '--c2', '--c3', '--c4', '--c5'];
  const L = SV.map(s => S.tk[s].slice(-n));
  if (!L[0] || !L[0].length) return '<div class="empty">No data yet.</div>';

  const m = L[0].length;
  const tot = i => L.reduce((a, l) => a + ((l[l.length - m + i] || {}).c || 0), 0);
  const mx = Math.max(...Array.from({ length: m }, (_, i) => tot(i))) || 1;

  let g = '<line class="gl" x1="0" x2="520" y1="250" y2="250"/>';
  for (let i = 0; i < m; i++) {
    let y = 250;
    L.forEach((l, k) => {
      const h = (((l[l.length - m + i] || {}).c || 0) / mx) * 220;
      y -= h;
      g += `<rect x="${20 + i * 82}" y="${y}" width="56" height="${h}" fill="var(${cols[k]})"/>`;
    });
    g += `<text x="${20 + i * 82}" y="272" class="ax">${hm((L[0][L[0].length - m + i] || {}).t || 0)}</text>`;
  }

  return `<svg viewBox="0 0 520 280" role="img" aria-label="Stacked cost by service">${g}</svg>`;
}

function timeline() {
  const t1 = S.now;
  const t0 = t1 - S.range * 60000;
  if (!t1) return '<div class="empty">No data yet.</div>';

  const X = t => 110 + (Math.min(Math.max(t, t0), t1) - t0) / (t1 - t0 || 1) * 770;
  let g = '';

  SV.forEach((s, i) => {
    const y = 40 + i * 46;
    g += `<text x="0" y="${y + 4}" class="ax">${LB[s]}</text>
          <line class="gl" x1="110" x2="880" y1="${y + 16}" y2="${y + 16}"/>`;
  });

  S.det.forEach(d => {
    const i = SV.indexOf(d.svc);
    if (i < 0 || d.end < t0) return;
    const y = 40 + i * 46;
    const x = X(d.start);
    const w = Math.max(8, X(Math.max(d.end, d.start)) - x);
    const open = d.status !== 'closed';
    const hi = (d.sev || 0) >= 60;

    g += `<g data-act="sel" data-id="${esc(d.id)}" style="cursor:pointer">
            <rect x="${x}" y="${y + 2}" width="${w}" height="28" rx="6" fill="${open ? (hi ? 'var(--crit)' : 'var(--c3)') : 'var(--bd2)'}"/>
            <text x="${x + 8}" y="${y + 20}" style="font:600 12px system-ui;fill:${open && hi ? 'var(--onc)' : open ? '#1B222B' : 'var(--tx)'}">
              ${Math.round(d.sev || 0)}${open ? ' · open' : ' · resolved'}
            </text>
          </g>`;
  });

  return `<svg viewBox="0 0 900 270" role="img" aria-label="Incident timeline">
    <text x="110" y="262" class="ax">${hm(t0)}</text>
    <text x="840" y="262" class="ax">${hm(t1)}</text>
    ${g}
  </svg>
  <p class="muted" style="margin:4px 0 0;font-size:12px">Bars run from detection to resolution. Select a bar to open its incident.</p>`;
}

// ---------------------------------------------------------------------------
// Views & Incident Cards
// ---------------------------------------------------------------------------
const WHY = {
  traffic_driven: 'Cost and volume both rose. This looks like real demand, not a configuration change.',
  price_driven: 'Cost per request rose while volume stayed flat. This looks like a pricing or configuration change. Throttling will not help.',
  volume_drop_cost_flat: 'Volume dropped but cost stayed flat. Possible orphaned provisioned capacity or fixed charges.'
};

function incCard(d, full) {
  const c = d.rca && (d.rca.classification || d.rca.alert_type);
  const p = [...S.prop.values()].find(x => x.detection_id === d.id);
  const hi = (d.sev || 0) >= 60;

  const btn = p && isPend(p)
    ? `<button class="btn ok" data-act="approve" data-id="${esc(p.proposal_id)}">Approve</button>
       <button class="btn" data-act="reject" data-id="${esc(p.proposal_id)}">Reject</button>`
    : '';

  const acts = c === 'price_driven'
    ? '<span class="muted">Recommended: investigate the configuration change.</span>'
    : btn;

  return `<div class="inc${S.sel === d.id ? ' sel' : ''}" data-act="sel" data-id="${esc(d.id)}">
    <div class="row">
      <span class="sev${hi ? '' : ' lo'}">${Math.round(d.sev || 0)}</span>
      <b>${esc(LB[d.svc] || d.svc)}</b>
      <span class="sp"></span>
      <span class="tag">${esc((c || d.status || '').replace(/_/g, ' '))}</span>
    </div>
    ${full ? `<div class="note">${esc(WHY[c] || 'Waiting for root-cause analysis.')}</div>` : ''}
    <div class="row" style="margin-top:8px">${acts}</div>
  </div>`;
}

const set = (id, h) => {
  const e = document.getElementById(id);
  if (e && e.innerHTML !== h) e.innerHTML = h;
};

const txt = (id, v) => {
  const e = document.getElementById(id);
  if (e && e.textContent !== String(v)) e.textContent = v;
};

const chips = (id, items, cur, act) => {
  set(id, items.map(i => `<button class="chip${i == cur ? ' on' : ''}" data-act="${act}" data-v="${i}">${i}</button>`).join(' '));
};

function render() {
  const pg = (location.hash || '#overview').slice(1);
  $$('.page').forEach(e => e.classList.toggle('on', e.id === 'p-' + pg));
  $$('.nav').forEach(a => a.classList.toggle('on', a.hash === '#' + pg));
  txt('title', pg[0].toUpperCase() + pg.slice(1));

  const lt = S.now;
  const tc = [...S.tot.values()].at(-1);
  const open = [...S.det.values()].filter(d => d.status !== 'closed');
  const pend = [...S.prop.values()].filter(isPend);

  txt('clock', 'Sim ' + (lt ? hm(lt) : '—'));
  txt('pause', S.paused ? 'Resume' : 'Pause');
  txt('kill', 'Kill switch: ' + (S.kill ? 'ON' : 'off'));
  if ($('#kill')) $('#kill').classList.toggle('bad', S.kill);
  if ($('#m-dry')) $('#m-dry').classList.toggle('on', S.mode === 'DRY_RUN');
  if ($('#m-enf')) $('#m-enf').classList.toggle('on', S.mode === 'ENFORCE');
  txt('run', S.run ? 'Run ' + S.run.slice(0, 8) : '');

  document.querySelectorAll('[data-range]').forEach(e => {
    const h = [15, 60, 360, 1440].map(m =>
      `<button class="chip${m === S.range ? ' on' : ''}" data-act="range" data-v="${m}">${m < 60 ? m + 'm' : (m / 60) + 'h'}</button>`
    ).join(' ');
    if (e.innerHTML !== h) e.innerHTML = h;
  });

  if (pg === 'overview') {
    txt('k-cost', tc ? usd(tc.c) : '—');
    txt('k-open', open.length);
    txt('k-open2', open.length ? open.map(d => LB[d.svc] || d.svc).join(', ') : 'none');

    const w = S.watch;
    const l = S.last;
    txt('k-lat', l ? l.lat.toFixed(1) + ' s' : w ? ((Date.now() - w.t0) / 1000).toFixed(0) + ' s…' : '—');
    txt('k-by', l ? l.svc + ' · ' + l.by : w ? 'waiting for detection' : 'inject an anomaly to measure');
    txt('k-sav', '$' + pend.reduce((a, p) => a + (+p.estimated_savings_hourly || 0), 0).toFixed(2) + '/hr');

    set('c-cost', lineChart());
    set('o-inc', open.length ? open.slice(-3).reverse().map(d => incCard(d, 1)).join('') : '<div class="empty">No open incidents.</div>');

    const q = S.ev;
    txt('q-p', pct($$$(q, ['precision', 'running_precision', 'p'])));
    txt('q-r', pct($$$(q, ['recall', 'running_recall', 'r'])));
    txt('q-f', pct($$$(q, ['f1', 'running_f1'])));
    txt('q-a', $$$(q, ['fa_per_day', 'false_alarms_per_day', 'fa_day'])?.toFixed?.(2) ?? '—');

    set('active', [...S.inj.values()]
      .filter(j => !j.ended && !j.sim_end_reached && j.status !== 'ended')
      .slice(-4)
      .map(j => `<div class="row">${esc(j.type)} · ${esc(LB[j.service] || j.service)} <button class="chip" data-act="end" data-id="${esc(j.injection_id)}">End now</button></div>`)
      .join('') || 'No active injections.');
  }

  if (pg === 'monitoring') {
    set('c-stack', stackChart());
    set('leg', SV.map((s, i) => `<span><i style="background:var(--c${i + 1})"></i>${LB[s]}</span>`).join(''));
    set('tiles', SV.map(s => {
      const a = S.tk[s].at(-1);
      const dv = a && a.e ? ((a.c / a.e) - 1) * 100 : 0;
      const w = Math.abs(dv) > 50;
      return `<div class="tile${w ? ' warn' : ''}">
        <span class="muted">${LB[s]}</span>
        <b style="display:block;font-size:20px">${a ? usd(a.c) : '—'}</b>
        <span style="color:var(${w ? '--crit' : '--ok'});font-weight:600">
          ${a ? (w ? (dv > 0 ? '+' : '') + dv.toFixed(0) + '% vs expected' : 'Normal') : 'No data'}
        </span>
      </div>`;
    }).join(''));
  }

  if (pg === 'intelligence') {
    set('c-tl', timeline());
    const d = S.det.get(S.sel) || [...S.det.values()].at(-1);
    set('i-det', d ? incCard(d, 1) : '<div class="empty">No incidents yet. Inject an anomaly to see one here.</div>');

    const r = (d && d.rca) || {};
    set('i-why', d ? `
      <div class="row">
        <div class="stat"><span class="muted">Cost / 1k now</span><b>${usd(r.cost_per_1k_now)}</b></div>
        <div class="stat"><span class="muted">Baseline</span><b>${usd(r.cost_per_1k_baseline)}</b></div>
      </div>
      <p style="margin:12px 0 6px"><b>Detectors that voted</b></p>
      <p class="muted" style="margin:0">${esc(d.votes.join(', ') || '—')}</p>` : '<div class="empty">Select an incident.</div>');
  }

  if (pg === 'remediation') {
    const P = [...S.prop.values()].reverse();
    txt('r-mode', S.mode === 'ENFORCE' ? 'Enforce' : 'Dry run');
    txt('r-pend', pend.length);
    txt('r-app', P.filter(isApp).length);
    txt('r-sav', '$' + S.saved.toFixed(2));
    chips('r-f', ['Pending', 'Applied', 'Rejected', 'All'], S.pf, 'pf');

    const F = P.filter(p => S.pf === 'All' || (S.pf === 'Pending' ? isPend(p) : S.pf === 'Applied' ? isApp(p) : /reject|block/i.test(p.status || '')));

    set('r-tab', F.length ? `
      <table>
        <tr>
          <th>Time</th>
          <th>Service</th>
          <th>Action</th>
          <th>Est. savings</th>
          <th>Risk</th>
          <th>Status</th>
          <th></th>
        </tr>
        ${F.map(p => `
          <tr>
            <td>${p.sim_ts ? hm(T(p.sim_ts)) : '—'}</td>
            <td><b>${esc(LB[p.service] || p.service)}</b></td>
            <td>${esc(String(p.action || '').replace(/_/g, ' '))}</td>
            <td>${usd(p.estimated_savings_hourly)}/hr</td>
            <td>${pct(p.estimated_risk)}</td>
            <td><span class="tag">${esc(p.status)}</span></td>
            <td>
              ${isPend(p) ? `
                <button class="btn ok" data-act="approve" data-id="${esc(p.proposal_id)}">Approve</button>
                <button class="btn" data-act="reject" data-id="${esc(p.proposal_id)}">Reject</button>
              ` : ''}
            </td>
          </tr>
        `).join('')}
      </table>
    ` : '<div class="empty">No proposals in this view.</div>');
  }

  if (pg === 'history') {
    chips('h-s', ['all', 'open', 'resolved'], S.fs, 'fs');
    chips('h-v', ['all', ...SV.map(s => LB[s])], S.fv === 'all' ? 'all' : LB[S.fv], 'fv');

    const R = [...S.det.values()]
      .filter(d => (S.fs === 'all' || (S.fs === 'open') === (d.status !== 'closed')) && (S.fv === 'all' || d.svc === S.fv))
      .sort((a, b) => b.start - a.start);

    set('h-tab', R.length ? `
      <table>
        <tr>
          <th>Detected</th>
          <th>Service</th>
          <th>Severity</th>
          <th>Cause</th>
          <th>Lifecycle</th>
        </tr>
        ${R.map(d => {
          const p = [...S.prop.values()].find(x => x.detection_id === d.id);
          const n = 1 + (p ? 1 : 0) + (p && isApp(p) ? 1 : 0) + (d.status === 'closed' ? 1 : 0);
          const rcaCause = (d.rca && (d.rca.classification || d.rca.alert_type)) || '—';
          return `
            <tr>
              <td>${hm(d.start)}</td>
              <td><b>${esc(LB[d.svc] || d.svc)}</b></td>
              <td><span class="sev${(d.sev || 0) >= 60 ? '' : ' lo'}">${Math.round(d.sev || 0)}</span></td>
              <td>${esc(rcaCause.replace(/_/g, ' '))}</td>
              <td>
                <span class="stp">
                  ${[0, 1, 2, 3].map(k => `<i class="${k < n ? 'on' : ''}"></i>`).join('')}
                </span>
                ${['Detected', 'Proposed', 'Mitigated', 'Resolved'][n - 1]}
              </td>
            </tr>
          `;
        }).join('')}
      </table>
    ` : '<div class="empty">No incidents match these filters.</div>');
  }
}

// ---------------------------------------------------------------------------
// Actions & Interactions
// ---------------------------------------------------------------------------
document.addEventListener('click', e => {
  const b = e.target.closest('[data-act]');
  if (!b) return;

  const a = b.dataset.act;
  const id = b.dataset.id;
  const v = b.dataset.v;

  if (a === 'sel') {
    S.sel = id;
    if (location.hash !== '#intelligence') location.hash = '#intelligence';
    draw();
  } else if (a === 'range') {
    S.range = +v;
    draw();
  } else if (a === 'pf') {
    S.pf = v;
    draw();
  } else if (a === 'fs') {
    S.fs = v;
    draw();
  } else if (a === 'fv') {
    S.fv = v === 'all' ? 'all' : SV.find(s => LB[s] === v);
    draw();
  } else if (a === 'theme') {
    const d = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = d;
    localStorage.setItem('crip-theme', d);
    txt('theme', d === 'dark' ? 'Light' : 'Dark');
  } else if (a === 'pause') {
    act(api('POST', S.paused ? '/api/control/resume' : '/api/control/pause'), () => {
      S.paused = !S.paused;
      draw();
    });
  } else if (a === 'reset') {
    if (confirm('Reset the simulation?')) {
      act(api('POST', '/api/control/reset', {}), () => {
        clearAll();
        draw();
      });
    }
  } else if (a === 'mode') {
    if (v === 'ENFORCE' && !confirm('Enable ENFORCE mode? Real mitigations will be applied to traffic.')) {
      return;
    }
    act(api('POST', '/api/remediation/mode', { mode: v }), () => {
      S.mode = v;
      draw();
    });
  } else if (a === 'kill') {
    act(api('POST', '/api/remediation/kill-switch', { enabled: !S.kill }), () => {
      S.kill = !S.kill;
      draw();
    });
  } else if (a === 'preset' || a === 'inject') {
    const body = a === 'preset'
      ? PRE[+b.dataset.i]
      : {
          type: $('#i-type').value,
          service: $('#i-svc').value,
          magnitude: +$('#i-mag').value,
          duration_min: +$('#i-dur').value,
          ramp_min: +$('#i-ramp').value
        };

    act(api('POST', '/api/inject', body), r => {
      S.last = null;
      S.watch = { t0: Date.now(), svc: body.service };
      if (r && r.injection_id) addInj({ ...body, ...r });
      toast('Injected: ' + body.type);

      setTimeout(() => {
        if (S.watch && !S.last) {
          S.watch = null;
          S.last = { lat: NaN, svc: body.service, by: 'missed' };
          txt('k-lat', 'Missed');
          draw();
        }
      }, 60000);
      draw();
    });
  } else if (a === 'end') {
    act(api('DELETE', '/api/inject/' + id), () => {
      const j = S.inj.get(id);
      if (j) j.ended = true;
      draw();
    });
  } else if (a === 'approve' || a === 'reject') {
    act(api('POST', `/api/remediation/proposals/${id}/${a}`), () => {
      const p = S.prop.get(id);
      if (p) p.status = a === 'approve' ? 'approved' : 'rejected';
      draw();
    });
  }
});

// Setup input bindings
if ($('#speed')) {
  $('#speed').onchange = e => act(api('POST', '/api/control/speed', { sim_minutes_per_real_second: +e.target.value }), () => {});
}

if ($('#gt')) {
  $('#gt').onchange = e => {
    S.gt = e.target.checked;
    draw();
  };
}

[['mag', '×'], ['dur', ' min'], ['ramp', ' min']].forEach(([k, u]) => {
  const i = $('#i-' + k);
  if (i) {
    i.oninput = () => txt('v-' + k, (k === 'mag' ? (+i.value).toFixed(1) : i.value) + u);
  }
});

if ($('#i-svc')) {
  $('#i-svc').innerHTML = ['all', ...SV].map(s =>
    `<option value="${s}"${s === 'DynamoDB' ? ' selected' : ''}>${LB[s] || s}</option>`
  ).join('');
}

window.addEventListener('hashchange', draw);

// Initial theme setup
{
  const t = localStorage.getItem('crip-theme') || (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = t;
  txt('theme', t === 'dark' ? 'Light' : 'Dark');
}

// ---------------------------------------------------------------------------
// Realtime Stream Connection (SSE)
// ---------------------------------------------------------------------------
function connect() {
  const es = new EventSource('/stream');
  const on = (n, f) => es.addEventListener(n, m => {
    try {
      f(JSON.parse(m.data));
      draw();
    } catch (x) {}
  });

  es.onopen = () => {
    const b = $('#banner');
    if (b) b.style.display = 'none';
  };

  es.onerror = () => {
    const b = $('#banner');
    if (b) b.style.display = 'block';
  };

  on('tick', addTick);
  on('detection', addDet);
  on('injection', addInj);
  on('eval', d => { S.ev = d.summary || d; });
  on('state', onState);
  on('remediation_proposal', addProp);
  on('remediation_result', d => {
    S.results++;
    S.saved += +(d.savings ?? d.net_benefit ?? 0) || 0;
  });
  on('mitigation', () => {});
  on('heartbeat', () => {});
}

async function health() {
  try {
    const h = await (await fetch('/health')).json();
    [['redis', 'd-redis'], ['generator', 'd-gen'], ['detector', 'd-det']].forEach(([k, id]) => {
      const e = document.getElementById(id);
      if (e) {
        e.className = 'dot ' + (h[k].ok ? 'ok' : 'bad');
        e.title = h[k].detail || '';
      }
    });
  } catch (x) {
    ['d-redis', 'd-gen', 'd-det'].forEach(id => {
      const e = document.getElementById(id);
      if (e) e.className = 'dot bad';
    });
  }
}

async function boot() {
  try {
    const h = await (await fetch('/history?minutes=1440')).json();
    (h.ticks || []).forEach(addTick);
    (h.detections || []).forEach(addDet);
    (h.injections || []).forEach(addInj);
    (h.proposals || []).forEach(addProp);
  } catch (x) {}

  try {
    const r = await (await fetch('/api/remediation/state')).json();
    S.mode = r.mode || 'DRY_RUN';
    S.kill = !!r.kill_switch;
  } catch (x) {}

  try {
    onState(await (await fetch('/api/state')).json());
  } catch (x) {}

  connect();
  health();
  setInterval(health, 3000);
  setInterval(() => {
    if (S.watch) draw();
  }, 1000);

  draw();
}

boot();
