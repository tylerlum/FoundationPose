// Renders the result tables and charts from results.json (published beside the page).
(function () {
  const TAKES = ['pick_lift', 'occlusion_tour', 'fast_wave', 'heldout_1', 'heldout_2', 'heldout_4', 'heldout_5', 'heldout_6'];
  const el = (tag, attrs = {}, kids = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'text') e.textContent = v; else if (k === 'html') e.innerHTML = v; else e.setAttribute(k, v);
    }
    for (const c of [].concat(kids)) if (c) e.appendChild(c);
    return e;
  };
  const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN;
  const runs = (byTake) => TAKES.map((t) => byTake && byTake[t]).filter((r) => r && r.ok2cm !== undefined);
  const fmt = (x, d = 0) => (Number.isFinite(x) ? x.toFixed(d) : '–');
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svg = (tag, attrs) => { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };

  function summaryRow(name, byTake, note) {
    const r = runs(byTake);
    const tr = el('tr');
    tr.appendChild(el('th', { scope: 'row', text: name }));
    if (!r.length) { tr.appendChild(el('td', { colspan: 5, class: 'muted', text: 'not run' })); return tr; }
    const ok = mean(r.map((x) => x.ok2cm)), lost = mean(r.map((x) => x.lost_pct));
    const rec = r.filter((x) => x.recovered).length;
    const cell = (v, d, cls) => el('td', { class: 'num ' + (cls || ''), text: v });
    tr.appendChild(barCell(ok, 'good'));
    tr.appendChild(barCell(lost, 'bad'));
    tr.appendChild(cell(`${rec}/${r.length}`, 0, rec === r.length ? 'pos' : ''));
    tr.appendChild(cell(fmt(mean(r.map((x) => x.cube_med)), 1) + ' mm'));
    tr.appendChild(el('td', { class: 'muted small', text: note || '' }));
    return tr;
  }
  function barCell(v, kind) {
    const td = el('td', { class: 'barcell' });
    const wrap = el('div', { class: 'bar ' + kind });
    const fill = el('span', { style: `width:${Math.max(0, Math.min(100, v))}%` });
    wrap.appendChild(fill);
    td.appendChild(el('div', { class: 'barrow' }, [wrap, el('span', { class: 'num', text: fmt(v, 0) + '%' })]));
    return td;
  }
  function table(head, rows) {
    const t = el('table');
    const thead = el('thead'); const hr = el('tr');
    head.forEach((h) => hr.appendChild(el('th', { scope: 'col', text: h })));
    thead.appendChild(hr); t.appendChild(thead);
    const tb = el('tbody'); rows.forEach((r) => tb.appendChild(r)); t.appendChild(tb);
    return el('div', { class: 'tablewrap' }, [t]);
  }

  function perTakeTable(cfgs, data) {
    const head = ['take'].concat(cfgs.map((c) => c[0]));
    const rows = TAKES.map((t) => {
      const tr = el('tr'); tr.appendChild(el('th', { scope: 'row', text: t }));
      cfgs.forEach(([, key]) => {
        const r = data[key] && data[key][t];
        if (!r || r.ok2cm === undefined) { tr.appendChild(el('td', { class: 'muted', text: '–' })); return; }
        const cls = r.ok2cm >= 70 ? 'pos' : r.ok2cm < 30 ? 'neg' : '';
        tr.appendChild(el('td', { class: 'num ' + cls, title: `lost ${r.lost_pct}% · ends ${r.end_mm} mm off`, text: `${fmt(r.ok2cm, 0)}%${r.recovered ? '' : ' ✗'}` }));
      });
      return tr;
    });
    return table(head, rows);
  }

  function curveChart(container, data, cfgs) {
    const sel = el('select', { id: 'curve-take', 'aria-label': 'Take' });
    TAKES.forEach((t) => sel.appendChild(el('option', { value: t, text: t })));
    sel.value = 'occlusion_tour';
    const legend = el('div', { class: 'legend' });
    const box = el('div', { class: 'chartbox' });
    container.appendChild(el('div', { class: 'chartctl' }, [el('label', { for: 'curve-take', text: 'Take ' }), sel, legend]));
    container.appendChild(box);
    const colors = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)'];
    function draw() {
      box.innerHTML = '';
      legend.innerHTML = '';
      const W = 900, H = 260, L = 46, B = 28, T = 10, Rr = 10;
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Brush pose error over time' });
      const ymax = 200; // mm, clipped
      let n = 0;
      cfgs.forEach(([, key]) => { const r = data[key] && data[key][sel.value]; if (r && r.curve) n = Math.max(n, r.curve.length); });
      const x = (i) => L + (i / Math.max(n - 1, 1)) * (W - L - Rr);
      const y = (v) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - B);
      [0, 20, 50, 100, 200].forEach((v) => {
        s.appendChild(svg('line', { x1: L, x2: W - Rr, y1: y(v), y2: y(v), class: v === 20 ? 'grid ok' : 'grid' }));
        const tx = svg('text', { x: L - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'axis' }); tx.textContent = v; s.appendChild(tx);
      });
      for (let sec = 0; sec * 30 < n; sec += 2) {
        const tx = svg('text', { x: x(sec * 30), y: H - 8, 'text-anchor': 'middle', class: 'axis' }); tx.textContent = sec + ' s'; s.appendChild(tx);
      }
      const yl = svg('text', { x: 12, y: T + 8, class: 'axis' }); yl.textContent = 'mm'; s.appendChild(yl);
      cfgs.forEach(([name, key], k) => {
        const r = data[key] && data[key][sel.value];
        if (!r || !r.curve) return;
        const d = r.curve.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
        s.appendChild(svg('path', { d, fill: 'none', stroke: colors[k % 4], 'stroke-width': 1.8, 'stroke-linejoin': 'round' }));
        legend.appendChild(el('span', { class: 'key' }, [el('i', { style: `background:${colors[k % 4]}` }), document.createTextNode(`${name} (${fmt(r.ok2cm, 0)}% < 2 cm)`)]));
      });
      box.appendChild(s);
    }
    sel.addEventListener('change', draw);
    draw();
  }

  function render(R) {
    const main = R.main;
    const m = document.getElementById('main-table');
    if (m) {
      const rows = [
        ['Old FoundationPose (plain)', 'plain', 'no SAM2'],
        ['v4 (live yesterday, old prompt)', 'v4 (old prompt)', ''],
        ['v4 + new SAM2 prompt', 'v4 + new prompt', 'recommended live now'],
        ['v4 + new SAM2 prompt (repeat)', 'v4 + new prompt (repeat)', ''],
        ['v8 (old prompt)', 'v8 (old prompt)', 'colour check: sim only'],
        ['v8 (old prompt, repeat)', 'v8 (old prompt, repeat)', ''],
        ['v8 + new prompt', 'v8 + new prompt', 'colour check: sim only'],
        ['v8 + new prompt (repeat)', 'v8 + new prompt (repeat)', ''],
        ['v8 + new prompt + filter', 'v8 + filter', 'constant-velocity start pose'],
        ['v8 + new prompt + filter (repeat)', 'v8 + filter (repeat)', ''],
      ].map(([n, k, note]) => summaryRow(n, main[k], note));
      m.appendChild(table(['config', 'frames within 2 cm', 'frames lost (> 5 cm)', 'recovered by end', 'cube error', ''], rows));
    }
    const pt = document.getElementById('per-take');
    if (pt) pt.appendChild(perTakeTable([['plain', 'plain'], ['v4 old', 'v4 (old prompt)'], ['v4 new', 'v4 + new prompt'], ['v8 new', 'v8 + new prompt'], ['v8 new +filter', 'v8 + filter']], main));
    const cc = document.getElementById('curves');
    if (cc) curveChart(cc, main, [['plain', 'plain'], ['v4 old prompt', 'v4 (old prompt)'], ['v4 new prompt', 'v4 + new prompt'], ['v8 new prompt', 'v8 + new prompt']]);
    const cam = document.getElementById('cam-table');
    if (cam && R.cams) {
      const names = { zed2_172707: 'ZED2 current, front-on', zed2_161129: 'ZED2 previous, diagonal', zed1: 'ZED1, left side' };
      const rows = [];
      Object.entries(R.cams).forEach(([c, d]) => {
        const occ = mean(Object.values(d._occluded_pct || {}));
        const px = mean(Object.values(d._brush_px_med || {}));
        [['old FP', 'plain'], ['v8', 'v8'], ['v8 repeat', 'v8b']].forEach(([n, k]) => {
          const r = runs(d[k]);
          const tr = el('tr');
          tr.appendChild(el('th', { scope: 'row', text: `${names[c] || c} · ${n}` }));
          if (!r.length) { tr.appendChild(el('td', { colspan: 4, class: 'muted', text: 'not run' })); rows.push(tr); return; }
          tr.appendChild(barCell(mean(r.map((x) => x.ok2cm)), 'good'));
          tr.appendChild(barCell(mean(r.map((x) => x.lost_pct)), 'bad'));
          tr.appendChild(el('td', { class: 'num', text: `${r.filter((x) => x.recovered).length}/${r.length}` }));
          tr.appendChild(el('td', { class: 'num muted', text: `${fmt(occ, 0)}% · ${fmt(px / 1000, 1)}k px` }));
          rows.push(tr);
        });
      });
      cam.appendChild(table(['camera · tracker', 'frames within 2 cm', 'frames lost', 'recovered', 'brush < 60% visible · median brush size'], rows));
    }
    const two = document.getElementById('twocam-table');
    if (two && R.twocam) {
      const rows = [];
      Object.entries(R.twocam).forEach(([pair, d]) => {
        Object.entries(d).forEach(([n, s]) => {
          const tr = el('tr');
          tr.appendChild(el('th', { scope: 'row', text: `${pair} · ${n}` }));
          tr.appendChild(barCell(s.ok2cm, 'good'));
          tr.appendChild(barCell(s.lost_pct, 'bad'));
          tr.appendChild(el('td', { class: 'num muted', text: s.n }));
          rows.push(tr);
        });
      });
      two.appendChild(table(['pair · choice', 'frames within 2 cm', 'frames lost', 'runs'], rows));
    }
  }

  function realChart(J) {
    const host = document.getElementById('real-chart');
    if (!host || !J) return;
    host.classList.remove('needs-data');
    const notes = { '20261118_164617': 'hand covers brush ~1.8 s; pose slides to the table after ~2.7 s',
      '20261118_173827': 'pose latches onto the orange KUKA link at ~4.4 s',
      '20261118_173734': 'picked up ~2 s', '20261118_173639': 'picked up ~1.7 s' };
    const colors = { mesh_colour: 'var(--c2)', appearance: 'var(--c3)' };
    const names = { mesh_colour: 'mesh-colour error (v8)', appearance: 'learned-appearance distance' };
    const legend = el('div', { class: 'legend' });
    Object.keys(colors).forEach((k) => legend.appendChild(el('span', { class: 'key' }, [el('i', { style: `background:${colors[k]}` }), document.createTextNode(names[k])])));
    host.appendChild(el('div', { class: 'chartctl' }, [el('strong', { text: 'Real takes, checks over time (no saved depth, so occlusion is not excluded)' }), legend]));
    const grid = el('div', { class: 'grid2' });
    Object.keys(J.mesh_colour || {}).forEach((take) => {
      const W = 440, H = 170, L = 34, B = 24, T = 8, Rr = 8;
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Checks over time for ${take}` });
      const tmax = Math.max(...Object.values(J).map((d) => (d[take] || []).reduce((m, p) => Math.max(m, p[0]), 0)), 1);
      const x = (t) => L + (t / tmax) * (W - L - Rr), y = (v) => T + (1 - Math.min(v, 0.8) / 0.8) * (H - T - B);
      [0, 0.2, 0.4, 0.6, 0.8].forEach((v) => {
        s.appendChild(svg('line', { x1: L, x2: W - Rr, y1: y(v), y2: y(v), class: 'grid' }));
        const tx = svg('text', { x: L - 4, y: y(v) + 4, 'text-anchor': 'end', class: 'axis' }); tx.textContent = v.toFixed(1); s.appendChild(tx);
      });
      for (let t = 0; t <= tmax; t += 1) { const tx = svg('text', { x: x(t), y: H - 6, 'text-anchor': 'middle', class: 'axis' }); tx.textContent = t + 's'; s.appendChild(tx); }
      Object.keys(colors).forEach((k) => {
        const pts = (J[k] || {})[take]; if (!pts) return;
        s.appendChild(svg('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(''), fill: 'none', stroke: colors[k], 'stroke-width': 1.8 }));
      });
      grid.appendChild(el('figure', {}, [el('div', { class: 'chartbox' }, [s]), el('figcaption', { text: `${take}: ${notes[take] || ''}` })]));
    });
    host.appendChild(grid);
  }
  fetch('real_checks.json').then((r) => r.json()).then(realChart).catch(() => {});

  fetch('results.json').then((r) => r.json()).then(render).catch((e) => {
    document.querySelectorAll('.needs-data').forEach((n) => { n.textContent = 'Could not load results.json: ' + e; });
  });
})();
