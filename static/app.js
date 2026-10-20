/* PoiMae front-end. No frameworks, no CDNs, no analytics.
 *
 * Two privacy rules the code keeps:
 *   1. the image never leaves this origin - it is POSTed to the same localhost
 *      server that serves this page, and that server never writes it to disk;
 *   2. the collection-day schedule lives in localStorage only.
 */
'use strict';

const DAYS_JA = ['月', '火', '水', '木', '金', '土', '日'];
const $ = (sel) => document.querySelector(sel);

const state = {
  municipalities: [],
  current: null,
  mode: 'photo',
  weekSchedule: {},
};

const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ------------------------------------------------------------------ setup */

async function boot() {
  const res = await fetch('/api/municipalities');
  const data = await res.json();
  state.municipalities = data.municipalities;

  const select = $('#municipality');
  select.innerHTML = state.municipalities
    .map((m) => {
      const kind = m.data_kind === 'official' ? '' : ' ※サンプル';
      return `<option value="${esc(m.id)}">${esc(m.display_name)}（${esc(m.prefecture)}）${kind}</option>`;
    })
    .join('');

  const saved = localStorage.getItem('poimae.municipality');
  if (saved && state.municipalities.some((m) => m.id === saved)) select.value = saved;

  select.addEventListener('change', () => onMunicipality());
  onMunicipality();

  $('.tabs').addEventListener('click', (event) => {
    const tab = event.target.closest('.tab');
    if (tab) setMode(tab.dataset.mode);
  });

  $('#file').addEventListener('change', onFile);
  document.querySelectorAll('[data-sample]').forEach((button) => {
    button.addEventListener('click', async () => {
      button.setAttribute('aria-busy', 'true');
      try {
        await loadSample(button.dataset.sample);
      } finally {
        button.removeAttribute('aria-busy');
      }
    });
  });
  $('#search-btn').addEventListener('click', onTextSearch);
  $('#query').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') onTextSearch();
  });

  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach((name) =>
    drop.addEventListener(name, (e) => { e.preventDefault(); drop.classList.add('hot'); }));
  ['dragleave', 'drop'].forEach((name) =>
    drop.addEventListener(name, () => drop.classList.remove('hot')));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleImage(file);
  });

  applyDeepLink(URLSearchParams ? new URLSearchParams(location.search) : null);
}

/* Deep links exist so the demo (and the screenshot script) can land straight on
 * an answered state: `?demo=pet_bottle` runs the photo flow, `?q=乾電池` runs
 * the name flow. Nothing here changes how an answer is computed. */
function applyDeepLink(params) {
  if (!params) return;
  const q = params.get('q');
  const demo = params.get('demo');
  if (demo) {
    setMode('photo');
    loadSample(demo).catch((err) => showError(`見本を読み込めませんでした: ${err.message}`));
  } else if (q) {
    setMode('text');
    $('#query').value = q;
    searchText(q);
  }
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('.tab').forEach((t) => {
    const on = t.dataset.mode === mode;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  $('#mode-photo').hidden = mode !== 'photo';
  $('#mode-text').hidden = mode !== 'text';
}

function currentMunicipality() {
  return state.municipalities.find((m) => m.id === state.current) || null;
}

function onMunicipality() {
  state.current = $('#municipality').value;
  localStorage.setItem('poimae.municipality', state.current);
  const m = currentMunicipality();
  if (!m) return;

  const badge = $('#source-badge');
  badge.textContent = m.data_kind === 'official' ? '公式データ' : 'サンプル（ダミー）';
  badge.className = 'badge ' + (m.data_kind === 'official' ? 'official' : 'dummy');

  const source = m.source || {};
  const url = source.url || '';
  $('#source-note').innerHTML = m.data_kind === 'official'
    ? `${esc(m.display_name)}の公式ごみ分別辞典から取り込み（${esc(m.item_count)}品目 / 取得日 ${esc(source.retrieved_at || '不明')}）。
       <a href="${esc(url)}" target="_blank" rel="noopener">出典を開く</a>`
    : `${esc(m.display_name)}は<b>サンプル（ダミー）</b>です。実在の制度ではありません。` +
      `${esc(source.disclaimer || '')}`;

  $('#cal-link').href = source.calendar_url || url || '#';
  $('#about-counts').textContent =
    `読み込み済み: ${state.municipalities.map((x) => x.display_name + '(' + x.item_count + '品目)').join(' / ')}`;

  restoreWeek();
  renderWeekConfig();
  refreshWeek();
}

/* ---------------------------------------------------------------- results */

function showResult(html) {
  $('#result-panel').hidden = false;
  $('#answer').innerHTML = html;
  // 'nearest' keeps the page still when the answer is already on screen - it
  // only moves when the reader would otherwise miss the result.
  $('#result-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
  showResult(`<div class="error">${esc(message)}</div>`);
}

const WARN_ICON = {
  cross_city: '⚠️',
  conditional: '📝',
  oversized: '🚚',
  dummy: '🧪',
};

function renderAnswer(data, features) {
  const lookup = data.lookup || {};
  const found = lookup.found;
  let html = '';

  if (found) {
    html += `
      <div class="verdict">
        <div class="q">${esc(data.question || '')}</div>
        <div class="item">${esc(lookup.item || '')}</div>
        <div class="bin">${esc(lookup.category || '—')}</div>
        ${lookup.fee && lookup.fee !== '0' && lookup.fee !== '0円'
          ? `<div class="fee">手数料 ${esc(lookup.fee)}</div>` : ''}
        <div class="fee">${esc(data.municipality)} の区分</div>
      </div>`;
  } else {
    html += `<div class="error">${esc(data.question || '見つかりませんでした')}</div>`;
  }

  if (lookup.note) {
    html += `<div class="note"><b>公式の注記：</b>${esc(lookup.note)}</div>`;
  }

  (data.warnings || []).forEach((w) => {
    html += `<div class="warn ${esc(w.code)}">
      <span class="icon">${WARN_ICON[w.code] || '•'}</span>
      <span>${esc(w.text)}${w.note ? `<br><span class="muted small">${esc(w.note)}</span>` : ''}</span>
    </div>`;
  });

  if ((data.other_cities || []).length) {
    html += `<div class="section-label">ほかの自治体では</div><div class="chips">`;
    data.other_cities.forEach((o) => {
      html += `<span class="chip">${esc(o.municipality)}: <b>${esc(o.category)}</b></span>`;
    });
    html += `</div>`;
  }

  if ((data.shortlist || []).length) {
    html += `<div class="section-label">候補（近いものを選んでください）</div><div class="candidates">`;
    data.shortlist.forEach((c, index) => {
      html += `<button class="cand" data-item="${esc(c.item)}">
        <span>
          <span class="c-name">${esc(c.item)}</span><br>
          <span class="c-meta">「${esc(c.via_keyword)}」から ${esc(c.match)}・一致度 ${(c.score).toFixed(2)}</span>
        </span>
        <span class="c-bin">${esc(c.category)}</span>
      </button>`;
    });
    html += `</div>`;
  }

  if ((data.material_evidence || []).length) {
    html += `<div class="section-label">写真から読み取った材質</div><div class="chips">`;
    data.material_evidence.slice(0, 3).forEach((e) => {
      html += `<span class="chip">${esc(e.label)} <b>${(e.score * 100).toFixed(0)}%</b></span>`;
    });
    html += `</div>`;
    const top = data.material_evidence[0];
    if (top && top.reasons?.length) {
      html += `<p class="muted small">根拠: ${top.reasons.map(esc).join(' / ')}</p>`;
    }
    html += `<div class="meter"><i style="width:${Math.round((top?.score || 0) * 100)}%"></i></div>`;
  }

  if (features) {
    html += `<div class="section-label">画像の特徴量（OpenCVで算出）</div><div class="chips">`;
    Object.entries(features).forEach(([k, v]) => {
      html += `<span class="chip">${esc(k)} <b>${typeof v === 'number' ? v : esc(v)}</b></span>`;
    });
    html += `</div>`;
  }

  showResult(html);

  document.querySelectorAll('.cand').forEach((button) => {
    button.addEventListener('click', () => searchText(button.dataset.item));
  });
}

/* ------------------------------------------------------------------ photo */

/* A bundled sample lets someone try the pipeline before they have a photo in
 * hand. It is fetched as bytes and wrapped in a File, so it travels the exact
 * same path as a real upload - no shortcut that could hide a bug. */
async function loadSample(stem) {
  const res = await fetch(`/static/samples/${stem}.png`);
  if (!res.ok) throw new Error(`sample ${stem} not found`);
  const blob = await res.blob();
  await handleImage(new File([blob], `${stem}.png`, { type: blob.type || 'image/png' }));
}

function onFile(event) {
  const file = event.target.files?.[0];
  if (file) handleImage(file);
}

async function handleImage(file) {
  const wrap = $('#preview-wrap');
  const img = $('#preview');
  wrap.hidden = false;
  $('#preview-meta').textContent = `${file.name} — ${(file.size / 1024).toFixed(0)} KB`;
  img.src = URL.createObjectURL(file);

  showResult(`<span class="spinner"></span>写真を解析しています（この端末の中だけで処理します）`);

  const form = new FormData();
  form.append('image', file);
  try {
    const res = await fetch(`/api/photo?municipality=${encodeURIComponent(state.current)}`, {
      method: 'POST',
      body: form,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderAnswer(data, data.features);
  } catch (err) {
    showError(`解析できませんでした: ${err.message}`);
  }
}

/* ------------------------------------------------------------------- text */

async function onTextSearch() {
  const query = $('#query').value.trim();
  if (query) searchText(query);
}

async function searchText(query) {
  showResult(`<span class="spinner"></span>「${esc(query)}」を調べています`);
  try {
    const url = `/api/lookup?municipality=${encodeURIComponent(state.current)}&q=${encodeURIComponent(query)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderAnswer(data, null);
  } catch (err) {
    showError(`検索できませんでした: ${err.message}`);
  }
}

/* ------------------------------------------------------------------- week */

function weekKey() {
  return `poimae.week.${state.current}`;
}

function restoreWeek() {
  try {
    state.weekSchedule = JSON.parse(localStorage.getItem(weekKey()) || '{}');
  } catch {
    state.weekSchedule = {};
  }
}

function renderWeekConfig() {
  const m = currentMunicipality();
  if (!m) return;
  const host = $('#week-config');
  host.innerHTML = '';
  m.categories.forEach(({ label }) => {
    const category = label;
    const selected = new Set(state.weekSchedule[category] || []);
    const row = document.createElement('div');
    row.className = 'week-row';
    row.innerHTML = `<span class="cat">${esc(category)}</span><span class="days"></span>`;
    const days = row.querySelector('.days');
    DAYS_JA.forEach((label, index) => {
      const weekday = index + 1;
      const button = document.createElement('button');
      button.className = 'day' + (selected.has(weekday) ? ' on' : '');
      button.textContent = label;
      button.setAttribute('aria-pressed', selected.has(weekday) ? 'true' : 'false');
      button.addEventListener('click', () => {
        const set = new Set(state.weekSchedule[category] || []);
        if (set.has(weekday)) set.delete(weekday); else set.add(weekday);
        state.weekSchedule[category] = [...set].sort();
        localStorage.setItem(weekKey(), JSON.stringify(state.weekSchedule));
        button.classList.toggle('on', set.has(weekday));
        button.setAttribute('aria-pressed', set.has(weekday) ? 'true' : 'false');
        refreshWeek();
      });
      days.appendChild(button);
    });
    host.appendChild(row);
  });
}

async function refreshWeek() {
  const out = $('#week-out');
  if (!Object.values(state.weekSchedule).some((d) => d.length)) {
    out.innerHTML = `<p class="muted small">曜日を選ぶと、この先1週間の予定が出ます。</p>`;
    return;
  }
  try {
    const res = await fetch('/api/week', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ municipality: state.current, schedule: state.weekSchedule }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    out.innerHTML = `<div class="today-line">${esc(data.headline)}</div>
      <div class="due-list">${data.due.map((d) => `
        <div class="due ${d.in_days === 0 ? 'today' : ''}">
          <span class="when">${esc(d.label)}（${esc(d.weekday)}）</span>
          <span class="what">${d.categories.map(esc).join(' / ')}</span>
        </div>`).join('') || '<span class="muted small">この先の予定はありません</span>'}
      </div>`;
  } catch (err) {
    out.innerHTML = `<div class="error">${esc(err.message)}</div>`;
  }
}

boot().catch((err) => showError(`起動に失敗しました: ${err.message}`));
