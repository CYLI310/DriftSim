/* DriftSim dataset GUI: edits a batch spec, validates it live, previews it, runs and lists datasets.
 * Plain JavaScript, no build step and no network access beyond this local server. */
"use strict";
(() => {
  // ------------------------------------------------------------------ small utilities
  const $ = (sel, el = document) => el.querySelector(sel);
  function h(tag, attrs = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "value") el.value = v;
      else if (k === "checked") el.checked = !!v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) if (kid !== null && kid !== undefined && kid !== false)
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return el;
  }
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const r6 = (x) => Number(Number(x).toPrecision(6));
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const fmtInt = (n) => (n == null ? "–" : Math.round(n).toLocaleString("en-US"));
  const fmtNum = (v) => (typeof v === "number" ? String(r6(v)) : String(v));
  function fmtBytes(b) {
    if (b == null) return "–";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
    while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
    return `${b >= 100 || i === 0 ? Math.round(b) : b.toFixed(1)} ${u[i]}`;
  }
  function fmtDur(s) {
    if (s == null || !isFinite(s)) return "–";
    if (s < 1) return "< 1 s";
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
    return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
  }
  const fmtDate = (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "–");
  function toast(msg, kind = "") {
    const el = h("div", { class: `toast ${kind}` }, msg);
    $("#toasts").append(el);
    setTimeout(() => el.remove(), kind === "bad" ? 7000 : 4000);
  }
  async function api(method, path, body) {
    const opt = { method, headers: {} };
    if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    const res = await fetch(path, opt);
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.error) || `${res.status} ${res.statusText}`);
      err.errors = (data && data.errors) || [];
      throw err;
    }
    return data;
  }
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* private mode */ } },
  };

  // ------------------------------------------------------------------ state
  const S = {
    cat: null, groups: {}, fields: {}, defaultSpec: null, cpu: 1, root: "", spec: null,
    section: "batch", search: "", advanced: !!store.get("driftsim.advanced"),
    val: { errors: [], warnings: [], estimate: null, pending: false }, rate: 40000,
    jobs: [], datasets: [], openFiles: {}, examples: [], online: false, jobState: {},
  };
  const DRAFT_KEY = "driftsim.spec.v1";
  const STARTUP_S = 2;          // worker processes start in about 2 s

  const SECTIONS = [
    { id: "batch", label: "Batch" },
    { id: "vehicle", label: "Chassis", group: "vehicle" },
    { id: "drivetrain", label: "Motor & drivetrain", group: "drivetrain" },
    { id: "actuators", label: "Steering & latency", group: "actuators" },
    { id: "tire", label: "Tires", group: "tire" },
    { id: "surface", label: "Surface", group: "surface" },
    { id: "condition", label: "Tire condition", group: "condition" },
    { id: "init", label: "Starting state", group: "init" },
    { id: "maneuver", label: "Driver inputs" },
    { id: "export", label: "Export" },
    { id: "sim", label: "Simulation", group: "sim" },
    { sep: true },
    { id: "runs", label: "Runs & datasets" },
  ];
  const INTRO = {
    vehicle: "Mass, size and weight distribution of the car.",
    drivetrain: "Layout, gearing, motor, battery, ESC and differentials.",
    actuators: "Steering limits and servo, control latency and the optional steering gyro.",
    tire: "Pick one or several compounds; episodes draw from the selection. Coefficients below apply to every selected compound.",
    surface: "Pick one or several surfaces; episodes draw from the selection. Properties below apply to every selected surface.",
    condition: "Tire wear, wetness, dust and starting temperature. Use per wheel for uneven tires.",
    init: "How each episode starts.",
    sim: "Time steps and numerical settings. The defaults are converged; change with care.",
  };
  const DIST_LABEL = {
    fixed: "Fixed", uniform: "Uniform", normal: "Normal", loguniform: "Log-uniform",
    choice: "Pick from list", sweep: "Sweep list (grid)", linspace: "Sweep range (grid)", bernoulli: "Random on/off",
  };
  const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#ff9da7"];

  // ------------------------------------------------------------------ field helpers
  const isManeuverKey = (k) => k.startsWith("maneuver.");
  function sectionOf(key) {
    if (isManeuverKey(key)) return "maneuver";
    if (key.startsWith("export") || key.startsWith("run")) return "export";
    const g = key.split(".")[0];
    return SECTIONS.some((s) => s.id === g) ? g : "batch";
  }
  function maneuverFields() {
    const m = S.cat.maneuvers[S.spec.maneuver.type];
    return (m ? m.params : []).map((p) => ({
      key: `maneuver.${p.name}`, name: p.name, default: p.default, unit: p.unit, desc: p.desc,
      type: "float", level: "basic", min: p.min, max: p.max,
    }));
  }
  function field(key) {
    if (isManeuverKey(key)) return maneuverFields().find((f) => f.key === key);
    return S.fields[key];
  }
  function getDist(key) {
    if (isManeuverKey(key)) return S.spec.maneuver.params[key.slice(9)];
    return S.spec.params[key];
  }
  function primaryChoice(gid) {
    const d = S.spec.params[gid];
    if (!d) return S.groups[gid].default_choice;
    return d.dist === "fixed" ? d.value : (d.values || [])[0];
  }
  function fieldDefault(key) {
    const f = field(key);
    if (!f) return undefined;
    for (const gid of ["tire", "surface"]) {
      if (key.startsWith(gid + ".")) {
        const cd = S.groups[gid].choice_defaults[primaryChoice(gid)];
        if (cd && key in cd) return cd[key];
      }
    }
    return f.default;
  }
  const sameValue = (a, b) => (typeof a === "number" && typeof b === "number" ? Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b)) : a === b);
  function isNoop(key, d) {
    if (!d) return true;
    if (d.dist !== "fixed" || (d.mode || "set") !== "set" || d.per_wheel) return false;
    if (key === "tire" || key === "surface") return d.value === S.groups[key].default_choice;
    return sameValue(d.value, fieldDefault(key));
  }
  function setDist(key, d) {
    const target = isManeuverKey(key) ? S.spec.maneuver.params : S.spec.params;
    const name = isManeuverKey(key) ? key.slice(9) : key;
    if (!d || isNoop(key, d)) delete target[name]; else target[name] = d;
    changed();
  }
  const isModified = (key) => getDist(key) !== undefined;
  function keyErrors(key) {
    return S.val.errors.filter((e) => e.startsWith(key + ":") || e.includes(`'${key}'`) ||
      (isManeuverKey(key) && e.includes(`parameter '${key.slice(9)}'`)));
  }
  function errorKey(msg) {
    const m = msg.match(/^([a-z_]+\.[A-Za-z0-9_]+|tire|surface):/) || msg.match(/'([a-z_]+\.[A-Za-z0-9_]+)'/);
    return m ? m[1] : null;
  }
  function distOptions(f) {
    if (f.type === "bool") return ["fixed", "bernoulli", "sweep"];
    if (f.type === "choice") return ["fixed", "choice", "sweep"];
    return ["fixed", "uniform", "normal", "loguniform", "choice", "sweep", "linspace"];
  }
  function freshDist(f, kind, cur) {
    const mode = cur && cur.mode === "scale" ? "scale" : undefined;
    let base = mode ? 1 : (cur && cur.dist === "fixed" && typeof cur.value === "number" ? cur.value : fieldDefault(f.key));
    if (f.type === "bool") {
      if (kind === "fixed") return { dist: "fixed", value: !!(cur && cur.value !== undefined ? cur.value : base) };
      if (kind === "bernoulli") return { dist: "bernoulli", p: 0.5 };
      return { dist: "sweep", values: [false, true] };
    }
    if (f.type === "choice") {
      const v = cur && cur.value !== undefined ? cur.value : base;
      if (kind === "fixed") return { dist: "fixed", value: v };
      const vals = [...new Set([v, ...f.choices])].slice(0, 2);
      return { dist: kind, values: vals };
    }
    if (typeof base !== "number" || !isFinite(base)) base = 0;
    const sp = base === 0 ? 1 : Math.abs(base) * 0.1;
    let d;
    switch (kind) {
      case "fixed": d = { dist: "fixed", value: r6(base) }; break;
      case "uniform": d = { dist: "uniform", low: r6(base - sp), high: r6(base + sp) }; break;
      case "normal": d = { dist: "normal", mean: r6(base), std: r6(sp / 2) }; break;
      case "loguniform": d = base > 0 ? { dist: "loguniform", low: r6(base * 0.8), high: r6(base * 1.25) }
        : { dist: "loguniform", low: 0.1, high: 1 }; break;
      case "linspace": d = { dist: "linspace", low: r6(base - sp), high: r6(base + sp), num: 3 }; break;
      default: d = { dist: kind, values: [r6(base - sp), r6(base), r6(base + sp)] };
    }
    if (mode) d.mode = mode;
    if (cur && cur.per_wheel) d.per_wheel = true;
    return d;
  }

  // ------------------------------------------------------------------ spec lifecycle
  function mergeSpec(loaded) {
    const base = clone(S.defaultSpec);
    const s = Object.assign(base, clone(loaded || {}));
    s.params = s.params || {};
    s.maneuver = Object.assign({ type: "drift_schedule", params: {} }, s.maneuver || {});
    s.maneuver.params = s.maneuver.params || {};
    s.export = Object.assign(clone(S.defaultSpec.export), s.export || {});
    s.run = Object.assign(clone(S.defaultSpec.run), s.run || {});
    return s;
  }
  function loadSpec(spec, label) {
    S.spec = mergeSpec(spec);
    changed(true);
    if (label) toast(`Loaded ${label}`, "ok");
  }
  function changed(full = false) {
    store.set(DRAFT_KEY, S.spec);
    S.val.pending = true;
    validate();
    if (full) render(); else { renderSidebar(); renderSummary(); markRows(); }
  }
  const validate = debounce(async () => {
    const sent = JSON.stringify(S.spec);
    try {
      const r = await api("POST", "/api/validate", { spec: S.spec });
      if (JSON.stringify(S.spec) !== sent) return;      // a newer edit is on its way
      S.val = { errors: r.errors, warnings: r.warnings, estimate: r.estimate, pending: false };
      S.rate = r.rate || S.rate;
    } catch (e) {
      S.val = { errors: [e.message], warnings: [], estimate: null, pending: false };
    }
    renderSidebar(); renderSummary(); markRows();
  }, 220);

  // ------------------------------------------------------------------ rendering: shell
  function render() { renderSidebar(); renderContent(); renderSummary(); }

  function sectionCount(id) {
    const p = S.spec.params;
    if (id === "batch") return ["name", "episodes", "duration_s", "seed"].filter((k) => S.spec[k] !== S.defaultSpec[k]).length;
    if (id === "maneuver") return Object.keys(S.spec.maneuver.params).length + (S.spec.maneuver.type !== S.defaultSpec.maneuver.type ? 1 : 0);
    if (id === "export") {
      const e = S.spec.export, d = S.defaultSpec.export;
      return Object.keys(d).filter((k) => JSON.stringify(e[k]) !== JSON.stringify(d[k])).length + (S.spec.run.workers !== S.defaultSpec.run.workers ? 1 : 0);
    }
    if (id === "runs") return S.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
    return Object.keys(p).filter((k) => k === id || k.startsWith(id + ".")).length;
  }
  function sectionHasError(id) { return S.val.errors.some((e) => { const k = errorKey(e); return k ? sectionOf(k) === id : id === "batch"; }); }

  function renderSidebar() {
    const nav = $("#sidebar");
    if (!nav.children.length) {
      for (const s of SECTIONS) {
        if (s.sep) { nav.append(h("div", { class: "nav-sep" })); continue; }
        nav.append(h("button", { class: "nav-item", "data-id": s.id, "aria-label": s.label,
          onclick: () => { S.section = s.id; S.search = ""; $("#search").value = ""; render(); window.scrollTo({ top: 0 }); } },
        h("span", {}, s.label), h("span", { class: "count", hidden: true })));
      }
    }
    for (const btn of nav.querySelectorAll(".nav-item")) {
      const id = btn.dataset.id, n = sectionCount(id), err = sectionHasError(id);
      btn.classList.toggle("active", S.section === id && !S.search);
      const badge = btn.querySelector(".count");
      badge.hidden = !err && !n;
      badge.classList.toggle("err", err);
      badge.textContent = err ? "!" : String(n);
      badge.title = err ? "has problems" : `${n} changed`;
    }
  }

  function renderContent() {
    const c = $("#content");
    if (S.search) return c.replaceChildren(renderSearch());
    const sec = S.section;
    let node;
    if (sec === "batch") node = renderBatch();
    else if (sec === "tire" || sec === "surface") node = renderChoiceGroup(sec);
    else if (sec === "maneuver") node = renderManeuver();
    else if (sec === "export") node = renderExport();
    else if (sec === "runs") node = renderRuns();
    else node = renderGroup(sec);
    c.replaceChildren(node);
  }

  // ------------------------------------------------------------------ parameter rows
  function numInput(value, onValue, { placeholder = "", allowEmpty = false, width, title, label } = {}) {
    const el = h("input", { type: "number", step: "any", value: value ?? "", placeholder, title, "aria-label": label || title });
    if (width) el.style.width = width;
    el.addEventListener("input", () => {
      const t = el.value.trim();
      if (t === "" && allowEmpty) { el.classList.remove("invalid"); onValue(null); return; }
      const v = Number(t);
      if (t === "" || !isFinite(v)) { el.classList.add("invalid"); return; }
      el.classList.remove("invalid");
      onValue(v);
    });
    return el;
  }
  function listInput(values, f, onValues) {
    const el = h("input", { type: "text", class: "list-input", value: (values || []).map(fmtNum).join(", "),
      placeholder: "values separated by commas" });
    el.addEventListener("input", () => {
      const parts = el.value.split(/[,\s]+/).filter(Boolean);
      const nums = parts.map(Number);
      if (!parts.length || nums.some((x) => !isFinite(x))) { el.classList.add("invalid"); return; }
      el.classList.remove("invalid");
      onValues(nums);
    });
    return el;
  }
  function chipSelect(options, selected, onChange, labelOf = (x) => String(x)) {
    const set = new Set(selected);
    return h("div", { class: "chips" }, options.map((o) => h("button", {
      class: `chip small ${set.has(o) ? "on" : ""}`, type: "button",
      onclick: (ev) => {
        if (set.has(o)) { if (set.size > 1) set.delete(o); } else set.add(o);
        ev.currentTarget.classList.toggle("on", set.has(o));
        onChange(options.filter((x) => set.has(x)));
      },
    }, labelOf(o))));
  }

  function paramRow(f) {
    const key = f.key;
    const def = fieldDefault(key);
    let cur = clone(getDist(key) || { dist: "fixed", value: def });
    const row = h("div", { class: "row", "data-key": key });
    const rerow = () => { const n = paramRow(f); row.replaceWith(n); markRow(n); };
    const commit = () => { setDist(key, clone(cur)); };
    const unit = f.unit && f.unit !== "-" ? f.unit : "";
    const scale = cur.mode === "scale";

    const left = [
      h("div", { class: "row-name" },
        h("span", { class: "pname", title: key }, f.name),
        unit && !scale ? h("span", { class: "unit" }, unit) : null,
        scale ? h("span", { class: "unit" }, "× default") : null,
        f.structural ? h("span", { class: "badge struct", title: "Varying this splits the batch into separate groups" }, "structural") : null),
      h("div", { class: "desc" }, f.desc || "", def !== undefined && def !== null ? h("span", {}, ` · default ${typeof def === "boolean" ? (def ? "on" : "off") : fmtNum(def)}`) : null),
    ];
    const ctl = h("div", { class: "row-ctl" });
    const sel = h("select", { "aria-label": `${f.name} distribution`,
      onchange: () => { cur = freshDist(f, sel.value, cur); commit(); rerow(); } },
    distOptions(f).map((k) => h("option", { value: k, selected: k === cur.dist }, DIST_LABEL[k])));
    ctl.append(sel);

    const set = (k) => (v) => { cur[k] = v; commit(); };
    const ph = scale ? "1" : "";
    if (f.type === "bool") {
      if (cur.dist === "fixed") {
        const inp = h("input", { type: "checkbox", checked: !!cur.value, onchange: () => { cur.value = inp.checked; commit(); } });
        ctl.append(h("label", { class: "switch", title: "on / off" }, inp, h("span")), h("span", { class: "lbl" }, cur.value ? "on" : "off"));
      } else if (cur.dist === "bernoulli") {
        ctl.append(h("span", { class: "lbl" }, "chance on"), numInput(cur.p, set("p"), { width: "80px" }));
      } else {
        ctl.append(h("span", { class: "lbl" }, "half the grid on, half off"));
      }
    } else if (f.type === "choice") {
      if (cur.dist === "fixed") {
        const s = h("select", { onchange: () => { cur.value = s.value; commit(); } },
          f.choices.map((c) => h("option", { value: c, selected: c === cur.value }, c)));
        ctl.append(s);
      } else {
        ctl.append(chipSelect(f.choices, cur.values, (vals) => { cur.values = vals; commit(); }));
      }
    } else {
      const allowEmpty = f.type === "optional_float";
      switch (cur.dist) {
        case "fixed":
          ctl.append(numInput(cur.value, (v) => { cur.value = v; commit(); }, { placeholder: allowEmpty ? "auto" : ph, allowEmpty, label: `${f.name} value` }));
          break;
        case "uniform": case "loguniform":
          ctl.append(numInput(cur.low, set("low"), { label: `${f.name} low` }), h("span", { class: "sep" }, "to"), numInput(cur.high, set("high"), { label: `${f.name} high` }));
          break;
        case "normal":
          ctl.append(h("span", { class: "lbl" }, "mean"), numInput(cur.mean, set("mean")), h("span", { class: "lbl" }, "std"), numInput(cur.std, set("std")),
            h("span", { class: "lbl" }, "clip"),
            numInput(cur.low, (v) => { if (v === null) delete cur.low; else cur.low = v; commit(); }, { placeholder: "min", allowEmpty: true, width: "76px" }),
            numInput(cur.high, (v) => { if (v === null) delete cur.high; else cur.high = v; commit(); }, { placeholder: "max", allowEmpty: true, width: "76px" }));
          break;
        case "linspace":
          ctl.append(numInput(cur.low, set("low")), h("span", { class: "sep" }, "to"), numInput(cur.high, set("high")),
            h("span", { class: "lbl" }, "points"), numInput(cur.num, (v) => { cur.num = Math.max(1, Math.round(v)); commit(); }, { width: "64px" }));
          break;
        default:
          ctl.append(listInput(cur.values, f, (vals) => { cur.values = vals; commit(); }));
      }
      ctl.append(h("button", { class: `opt-btn ${scale ? "on" : ""}`, type: "button",
        title: "Multiply the default instead of replacing it (e.g. ±5 % grip on every surface)",
        onclick: () => { cur = freshDist(f, cur.dist, { ...cur, mode: scale ? undefined : "scale", dist: "fixed", value: scale ? def : 1 });
          if (scale) delete cur.mode; commit(); rerow(); } }, "× default"));
    }
    if (f.per_wheel) {
      ctl.append(h("button", { class: `opt-btn ${cur.per_wheel ? "on" : ""}`, type: "button",
        title: "Draw each wheel separately (e.g. uneven wear)",
        onclick: () => { if (cur.per_wheel) delete cur.per_wheel; else cur.per_wheel = true; commit(); rerow(); } }, "per wheel"));
    }
    ctl.append(h("button", { class: "btn btn-ghost icon reset", type: "button", title: "Reset to default",
      onclick: () => { setDist(key, null); rerow(); } }, "↺"));
    row.append(...left, ctl, h("div", { class: "row-err" }));
    return row;
  }
  function markRow(row) {
    const key = row.dataset.key;
    const errs = keyErrors(key);
    row.classList.toggle("modified", isModified(key));
    row.classList.toggle("error", errs.length > 0);
    const e = row.querySelector(".row-err");
    if (e) e.textContent = errs.map((m) => m.replace(new RegExp(`^${key.replace(".", "\\.")}:\\s*`), "")).join(" · ");
  }
  function markRows() { document.querySelectorAll(".row[data-key]").forEach(markRow); }

  function rowsFor(fields) {
    const basic = fields.filter((f) => f.level !== "advanced");
    const adv = fields.filter((f) => f.level === "advanced");
    const out = basic.map(paramRow);
    const modAdv = adv.filter((f) => isModified(f.key));
    if (S.advanced) out.push(...(adv.length ? [h("div", { class: "subhead" }, "Advanced"), ...adv.map(paramRow)] : []));
    else if (modAdv.length) out.push(h("div", { class: "subhead" }, "Advanced (changed)"), ...modAdv.map(paramRow));
    if (!S.advanced && adv.length) out.push(h("p", { class: "muted small" },
      `${adv.length} advanced settings hidden. `, h("a", { href: "#", onclick: (e) => { e.preventDefault(); $("#advanced").click(); } }, "Show them")));
    return out;
  }

  // ------------------------------------------------------------------ sections
  function renderGroup(gid) {
    const g = S.groups[gid];
    return h("div", {}, h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h2", {}, g.label)),
      h("p", { class: "section-intro" }, INTRO[gid] || ""),
      rowsFor(g.fields)));
  }

  function renderChoiceGroup(gid) {
    const g = S.groups[gid];
    const d = S.spec.params[gid];
    const selected = !d ? [g.default_choice] : d.dist === "fixed" ? [d.value] : d.values;
    const mode = d && d.dist === "sweep" ? "sweep" : "choice";
    const update = (vals, m) => setDist(gid, vals.length === 1 ? { dist: "fixed", value: vals[0] } : { dist: m, values: vals });
    const chips = h("div", { class: "chips" }, g.choices.map((c) => h("button", {
      class: `chip ${selected.includes(c) ? "on" : ""}`, type: "button",
      onclick: () => {
        let vals = selected.includes(c) ? selected.filter((x) => x !== c) : [...selected, c];
        if (!vals.length) vals = [c];
        vals = g.choices.filter((x) => vals.includes(x));
        update(vals, mode); renderContent();
      },
    }, g.colors ? h("span", { class: "swatch", style: `background:${g.colors[c]}` }) : null, c.replaceAll("_", " "))));
    const modeSeg = selected.length > 1 ? h("div", { class: "seg", style: "margin-top:12px" },
      h("button", { class: mode === "choice" ? "on" : "", onclick: () => { update(selected, "choice"); renderContent(); } }, "Random per episode"),
      h("button", { class: mode === "sweep" ? "on" : "", onclick: () => { update(selected, "sweep"); renderContent(); } }, "Sweep through all (grid)")) : null;
    const prim = primaryChoice(gid);
    return h("div", {},
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, g.label), h("span", { class: "muted small" }, `${selected.length} of ${g.choices.length} selected`)),
        h("p", { class: "section-intro" }, INTRO[gid]), chips, modeSeg),
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, gid === "tire" ? "Compound coefficients" : "Surface properties"),
          h("span", { class: "muted small" }, `defaults shown for ${prim.replaceAll("_", " ")}`)),
        h("p", { class: "section-intro" }, "Set a value to override it for every selected choice, or use “× default” to scale each choice's own value."),
        rowsFor(g.fields)));
  }

  function renderBatch() {
    const sp = S.spec;
    const setTop = (k) => (v) => { sp[k] = k === "episodes" || k === "seed" ? Math.max(0, Math.round(v)) : v; changed(); };
    const quick = (k, vals, fmt) => h("div", { class: "chips" }, vals.map((v) => h("button", {
      class: `chip small ${sp[k] === v ? "on" : ""}`, type: "button", onclick: () => { sp[k] = v; changed(true); } }, fmt(v))));
    const name = h("input", { type: "text", value: sp.name, style: "width: 260px",
      oninput: () => { sp.name = name.value; changed(); } });
    const est = S.val.estimate;
    return h("div", {},
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, "Batch")),
        h("p", { class: "section-intro" }, "How many episodes to simulate and for how long. Every episode draws its own values for the variables you randomize in the other sections."),
        h("div", { class: "form-grid" },
          h("label", {}, "Name"), name,
          h("label", {}, "Episodes"), h("div", { class: "ctl" }, numInput(sp.episodes, setTop("episodes"), { width: "120px" }),
            quick("episodes", [100, 1000, 10000, 100000], fmtInt)),
          h("label", {}, "Episode length"), h("div", { class: "ctl" }, numInput(sp.duration_s, setTop("duration_s")), h("span", { class: "lbl" }, "seconds"),
            quick("duration_s", [2, 4, 8, 15], (v) => `${v} s`)),
          h("label", {}, "Seed"), h("div", { class: "ctl" }, numInput(sp.seed, setTop("seed")),
            h("button", { class: "btn small", type: "button", onclick: () => { sp.seed = Math.floor(Math.random() * 1e6); changed(true); } }, "New seed"),
            h("span", { class: "lbl" }, "the whole dataset is reproducible from the spec and this seed")))),
      h("div", { class: "card" },
        h("h3", {}, "Quick start"),
        h("ol", { class: "muted", style: "margin: 0; padding-left: 18px" },
          h("li", {}, "Pick tires, surfaces and driver inputs in the sections on the left."),
          h("li", {}, "For anything you want to vary, change “Fixed” to Uniform, Normal, Pick from list or a Sweep."),
          h("li", {}, "Check the summary on the right, press Preview to see a few episodes, then Start."),
          h("li", {}, "Finished datasets appear under Runs & datasets, ready to download.")),
        est ? h("p", { class: "muted small" }, `Each episode is ${fmtInt(est.control_steps)} control steps and ${fmtInt(est.records_per_episode)} recorded rows of ${est.columns} columns.`) : null));
  }

  function renderManeuver() {
    const cur = S.spec.maneuver.type;
    const cards = h("div", { class: "man-grid" }, Object.entries(S.cat.maneuvers).map(([k, m]) => h("button", {
      class: `man-card ${k === cur ? "on" : ""}`, type: "button",
      onclick: () => { if (k !== cur) { S.spec.maneuver = { type: k, params: {} }; changed(true); } },
    }, h("b", {}, m.label), h("span", {}, m.desc))));
    return h("div", {},
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, "Driver inputs")),
        h("p", { class: "section-intro" }, "The steering and throttle each episode receives (open loop, 50 Hz). Every parameter below can be randomized or swept too."),
        cards,
        h("div", { class: "subhead" }, `${S.cat.maneuvers[cur].label} parameters`),
        maneuverFields().map(paramRow)));
  }

  function renderExport() {
    const e = S.spec.export;
    const setE = (k, v) => { e[k] = v; changed(); };
    const fmtCards = h("div", { class: "grid-2" }, S.cat.all_formats.map((f) => {
      const avail = S.cat.formats.includes(f);
      const on = e.formats.includes(f);
      const desc = { npz: "NumPy arrays per shard (fast, compact). Recommended.", csv: "Long table, one row per episode and time step (about 2.5x larger).",
        parquet: avail ? "Columnar table for pandas / Spark." : "Needs pyarrow: pip install pyarrow" }[f];
      return h("label", { class: `check-card ${on ? "on" : ""}`, style: avail ? "" : "opacity:.55" },
        h("input", { type: "checkbox", checked: on, disabled: !avail, onchange: (ev) => {
          const set = new Set(e.formats); ev.target.checked ? set.add(f) : set.delete(f);
          if (!set.size) { ev.target.checked = true; return; }
          setE("formats", S.cat.all_formats.filter((x) => set.has(x))); renderContent(); } }),
        h("div", {}, h("b", {}, f.toUpperCase()), h("div", { class: "muted small" }, desc)));
    }));
    const sigCards = h("div", { class: "grid-2" }, S.cat.signals.map((g) => {
      const on = e.signals.includes(g.key);
      return h("label", { class: `check-card ${on ? "on" : ""}` },
        h("input", { type: "checkbox", checked: on, onchange: (ev) => {
          const set = new Set(e.signals); ev.target.checked ? set.add(g.key) : set.delete(g.key);
          if (!set.size) { ev.target.checked = true; return; }
          setE("signals", S.cat.signals.map((x) => x.key).filter((k) => set.has(k))); renderContent(); } }),
        h("div", {}, h("b", {}, g.label), g.default ? h("span", { class: "badge", style: "margin-left:6px" }, "default") : null,
          h("div", { class: "names" }, g.signals.map((s) => s.name + (s.per_wheel ? "[4]" : "")).join("  "))));
    }));
    const sw = (k) => { const i = h("input", { type: "checkbox", checked: !!e[k], onchange: () => setE(k, i.checked) }); return h("label", { class: "switch" }, i, h("span")); };
    const out = h("input", { type: "text", value: e.out_dir, style: "width: 260px", oninput: () => setE("out_dir", out.value || "exports") });
    const workers = h("select", { onchange: () => { S.spec.run.workers = Number(workers.value); changed(); } },
      h("option", { value: 0, selected: S.spec.run.workers === 0 }, `Automatic (${Math.max(1, S.cpu - 1)})`),
      Array.from({ length: S.cpu }, (_, i) => h("option", { value: i + 1, selected: S.spec.run.workers === i + 1 }, String(i + 1))));
    const run = S.spec.run, devs = S.devices || { cpu: true }, dev = run.device || "cpu";
    const setRun = (k, v) => { run[k] = v; changed(); renderContent(); };
    const devSel = h("select", { "aria-label": "Compute device", onchange: () => {
        if (devSel.value === "mps") run.precision = "float32";
        setRun("device", devSel.value); } },
      h("option", { value: "cpu", selected: dev === "cpu" }, "CPU: NumPy float64, all cores"),
      h("option", { value: "mps", selected: dev === "mps", disabled: !devs.mps }, `Apple GPU (MPS), float32${devs.mps ? "" : " (not available)"}`),
      h("option", { value: "cuda", selected: dev === "cuda", disabled: !devs.cuda }, `NVIDIA GPU (CUDA)${devs.cuda ? "" : " (not available)"}`),
      h("option", { value: "auto", selected: dev === "auto" }, `Best available (${devs.cuda ? "CUDA" : devs.mps ? "MPS" : "CPU"})`));
    const gpuOn = dev !== "cpu" && (dev !== "auto" || devs.cuda || devs.mps);
    const isMps = dev === "mps" || (dev === "auto" && !devs.cuda && devs.mps);
    const precSel = h("select", { "aria-label": "Precision", onchange: () => setRun("precision", precSel.value) },
      h("option", { value: "float32", selected: run.precision !== "float64" }, "float32 (fast)"),
      h("option", { value: "float64", selected: run.precision === "float64", disabled: isMps }, `float64${isMps ? " (not on Apple GPUs)" : ""}`));
    const comp = h("input", { type: "checkbox", checked: !!run.compile, onchange: () => setRun("compile", comp.checked) });
    const cdt = 0.02, pdt = 0.001, phys = e.record_rate === "physics";
    const recRate = h("select", { "aria-label": "Record rate", onchange: () => { setE("record_rate", recRate.value); renderContent(); } },
      h("option", { value: "control", selected: !phys }, `Every control step (${r6(1 / cdt)} Hz)`),
      h("option", { value: "physics", selected: phys }, `Every physics step (${r6(1 / pdt)} Hz)`));
    return h("div", {},
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "File formats")), fmtCards),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Signals"),
        h("span", { class: "muted small" }, "[4] = one column per wheel (FL, FR, RL, RR)")), sigCards),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Output options")),
        h("div", { class: "form-grid" },
          h("label", {}, "Record at"), h("div", { class: "ctl" }, recRate,
            h("span", { class: "lbl" }, phys ? "motor, steering and wheels every 1 ms; files about 20x larger" : "the rate the controller acts at")),
          h("label", {}, "Record every"), h("div", { class: "ctl" }, numInput(e.decimation, (v) => setE("decimation", Math.max(1, Math.round(v))), { width: "70px" }),
            h("span", { class: "lbl" }, `${phys ? "physics" : "control"} steps (${r6(1 / ((phys ? pdt : cdt) * Math.max(1, e.decimation)))} Hz)`)),
          h("label", {}, "Single precision"), h("div", { class: "ctl" }, sw("float32"), h("span", { class: "lbl" }, "float32 halves the size")),
          h("label", {}, "Compress"), h("div", { class: "ctl" }, sw("compress"), h("span", { class: "lbl" }, "smaller files, slower to write")),
          h("label", {}, "Episodes per file"), h("div", { class: "ctl" }, numInput(e.shard_episodes, (v) => setE("shard_episodes", Math.max(1, Math.round(v))), { width: "90px" }),
            h("span", { class: "lbl" }, "maximum; smaller shards are used to keep all cores busy")),
          h("label", {}, "Output folder"), h("div", { class: "ctl" }, out, h("span", { class: "lbl" }, `relative to the repository · now ${S.root}`)),
          h("label", {}, "Worker processes"), h("div", { class: "ctl" }, workers, gpuOn ? h("span", { class: "lbl" }, "not used on a GPU") : null),
          h("label", {}, "Compute device"), h("div", { class: "ctl" }, devSel,
            h("span", { class: "lbl" }, gpuOn ? "pays off from a few thousand episodes; float32 results match the CPU statistically, not episode by episode"
              : "the float64 reference; every episode is exactly reproducible")),
          gpuOn ? h("label", {}, "Precision") : null, gpuOn ? h("div", { class: "ctl" }, precSel) : null,
          gpuOn && !isMps ? h("label", {}, "Fuse kernels") : null,
          gpuOn && !isMps ? h("div", { class: "ctl" }, h("label", { class: "switch" }, comp, h("span")),
            h("span", { class: "lbl" }, "torch.compile on NVIDIA GPUs: slower start, faster after")) : null)));
  }

  function renderSearch() {
    const q = S.search.toLowerCase();
    const groups = [];
    for (const s of SECTIONS) {
      if (!s.group && s.id !== "maneuver") continue;
      const fields = s.id === "maneuver" ? maneuverFields() : S.groups[s.group].fields;
      const hits = fields.filter((f) => `${f.key} ${f.name} ${f.desc}`.toLowerCase().includes(q));
      if (hits.length) groups.push(h("div", { class: "subhead" }, s.label), ...hits.map(paramRow));
    }
    return h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, `Variables matching “${S.search}”`)),
      groups.length ? groups : h("div", { class: "empty" }, "No variable matches. Try another word, e.g. grip, mass, latency, wear."));
  }

  // ------------------------------------------------------------------ runs & datasets
  function jobCard(j, compact = false) {
    const p = j.progress || {};
    const frac = j.status === "complete" ? 1 : p.fraction || 0;
    const badge = { queued: "badge", running: "badge run", complete: "badge ok", failed: "badge bad", cancelled: "badge" }[j.status] || "badge";
    return h("div", { class: "job" },
      h("div", { class: "job-head" }, h("b", {}, j.name), h("span", { class: badge }, j.status)),
      j.status === "running" || j.status === "queued"
        ? h("div", { class: `progress ${j.status === "queued" || !p.fraction ? "indet" : ""}` }, h("div", { style: `width:${(100 * frac).toFixed(1)}%` })) : null,
      h("div", { class: "job-meta" },
        j.status === "running" ? `${fmtInt(p.done)} / ${fmtInt(j.episodes)} episodes · ${p.episodes_per_s ? p.episodes_per_s.toFixed(1) + " /s" : "starting"} · ETA ${fmtDur(p.eta_s)}`
          : j.status === "queued" ? "waiting for the previous job"
          : j.summary ? `${fmtInt(j.summary.episodes_written)} episodes in ${fmtDur(j.summary.seconds)} · ${fmtBytes(j.summary.bytes)}${j.error ? " · " + j.error : ""}`
          : j.error || ""),
      !compact && j.warnings && j.warnings.length ? h("div", { class: "job-meta" }, "⚠ " + j.warnings.join(" · ")) : null,
      (j.status === "running" || j.status === "queued") ? h("div", { style: "margin-top:8px" },
        h("button", { class: "btn small btn-danger", onclick: () => cancelJob(j.id) }, "Cancel")) : null,
      j.dataset && j.status !== "running" && j.status !== "queued" ? h("div", { class: "row-actions", style: "margin-top:8px" },
        h("a", { class: "btn small", href: `/api/datasets/${encodeURIComponent(j.dataset)}/zip` }, "Download .zip"),
        h("button", { class: "btn small", onclick: () => reveal(j.dataset) }, "Open folder")) : null);
  }
  function runsJobsContent() {
    return [h("div", { class: "card-head" }, h("h2", {}, "Runs this session")),
      ...(S.jobs.length ? S.jobs.map((j) => jobCard(j)) : [h("div", { class: "empty" }, "No runs yet. Press Start in the summary panel.")])];
  }
  function renderRuns() {
    const rows = S.datasets.map((d) => {
      const st = d.stats || {};
      const badge = { complete: "badge ok", running: "badge run", failed: "badge bad", interrupted: "badge bad", cancelled: "badge" }[d.status] || "badge";
      const filesBox = h("div", { class: `files ${S.openFiles[d.id] ? "open" : ""}`, id: `files-${d.id}` });
      if (S.openFiles[d.id]) fillFiles(filesBox, d.id);
      return h("tr", {},
        h("td", {}, h("b", {}, d.name), h("div", { class: "muted small mono" }, d.id), filesBox),
        h("td", {}, fmtDate(d.created)),
        h("td", {}, h("span", { class: badge }, d.status)),
        h("td", { class: "num" }, `${fmtInt(d.episodes_written)}${d.episodes_written !== d.episodes ? " / " + fmtInt(d.episodes) : ""}`),
        h("td", { class: "num" }, fmtBytes(d.bytes)),
        h("td", { class: "num" }, st.drift_fraction != null ? `${Math.round(100 * st.drift_fraction)} %` : "–"),
        h("td", { class: "num" }, st.spun_fraction != null ? `${Math.round(100 * st.spun_fraction)} %` : "–"),
        h("td", {}, h("div", { class: "row-actions" },
          h("a", { class: "btn small", href: `/api/datasets/${encodeURIComponent(d.id)}/zip`, title: "Download everything as a zip" }, "Zip"),
          h("button", { class: "btn small", onclick: () => { S.openFiles[d.id] = !S.openFiles[d.id]; renderContent(); } }, S.openFiles[d.id] ? "Hide files" : "Files"),
          h("button", { class: "btn small", onclick: () => reveal(d.id), title: "Open in Finder / file manager" }, "Folder"),
          h("button", { class: "btn small", onclick: () => loadDatasetSpec(d.id), title: "Load this dataset's spec into the editor" }, "Load spec"))));
    });
    return h("div", {},
      h("div", { class: "card", id: "runs-jobs" }, runsJobsContent()),
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, "Datasets"),
          h("div", { class: "row-actions" }, h("span", { class: "muted small mono" }, S.root), h("button", { class: "btn small", onclick: loadDatasets }, "Refresh"))),
        rows.length ? h("table", { class: "data" },
          h("thead", {}, h("tr", {}, h("th", {}, "dataset"), h("th", {}, "created"), h("th", {}, "status"), h("th", { class: "num" }, "episodes"),
            h("th", { class: "num" }, "size"), h("th", { class: "num", title: "episodes with at least 1 s of drift" }, "drifting"),
            h("th", { class: "num", title: "episodes that spun out" }, "spun"), h("th", {}, ""))),
          h("tbody", {}, rows)) : h("div", { class: "empty" }, "No datasets in this folder yet.")));
  }
  async function fillFiles(box, id) {
    try {
      const d = await api("GET", `/api/datasets/${encodeURIComponent(id)}`);
      box.replaceChildren(h("table", { class: "data" }, h("tbody", {}, d.files.map((f) => h("tr", {},
        h("td", {}, h("a", { href: `/api/datasets/${encodeURIComponent(id)}/files/${encodeURIComponent(f.name)}` }, f.name)),
        h("td", { class: "num" }, fmtBytes(f.bytes)))))));
    } catch (e) { box.textContent = e.message; }
  }
  async function reveal(id) {
    try { await api("POST", `/api/datasets/${encodeURIComponent(id)}/reveal`, {}); } catch (e) { toast(e.message, "bad"); }
  }
  async function loadDatasetSpec(id) {
    try { const d = await api("GET", `/api/datasets/${encodeURIComponent(id)}`); loadSpec(d.manifest.spec, `spec of ${id}`); S.section = "batch"; render(); }
    catch (e) { toast(e.message, "bad"); }
  }
  async function loadDatasets() {
    try {
      const r = await api("GET", "/api/datasets");
      S.root = r.root;
      const sig = JSON.stringify(r.datasets);
      if (sig === S.datasetsSig) return;
      S.datasetsSig = sig; S.datasets = r.datasets;
      if (S.section === "runs" && !S.search) renderContent();
    }
    catch (e) { /* offline: the status line shows it */ }
  }
  async function cancelJob(id) {
    try { await api("POST", `/api/jobs/${id}/cancel`, {}); toast("Cancelling… shards already written are kept."); pollJobs(); }
    catch (e) { toast(e.message, "bad"); }
  }

  // ------------------------------------------------------------------ summary panel
  function variedKeys() {
    return [...Object.keys(S.spec.params), ...Object.keys(S.spec.maneuver.params).map((k) => `maneuver.${k}`)];
  }
  function gridSize() {
    let g = 1;
    for (const d of [...Object.values(S.spec.params), ...Object.values(S.spec.maneuver.params)]) {
      if (d.dist === "sweep") g *= (d.values || []).length || 1;
      if (d.dist === "linspace") g *= Math.max(1, d.num || 1);
    }
    return g;
  }
  function goTo(key) {
    if (!key) { S.section = "batch"; render(); return; }
    S.section = sectionOf(key); S.search = ""; $("#search").value = "";
    const f = field(key);
    if (f && f.level === "advanced" && !S.advanced) { S.advanced = true; $("#advanced").checked = true; }
    render();
    const row = document.querySelector(`.row[data-key="${CSS.escape(key)}"]`);
    if (row) { row.scrollIntoView({ block: "center", behavior: "smooth" }); row.classList.add("flash"); }
  }
  function ensureSummary() {
    if (S.sum) return S.sum;
    const stat = (k) => { const v = h("div", { class: "v" }), kk = h("div", { class: "k" }, k); return { box: h("div", { class: "stat" }, v, kk), v, k: kk }; };
    const el = {
      name: h("div", { class: "spec-name" }), status: h("div"), warn: h("div"), varied: h("div"),
      episodes: stat("episodes"), each: stat("each"), bytes: stat("output (approx.)"), time: stat("run time (approx.)"),
      steps: stat("control steps"), grid: stat("variables changed"),
      preview: h("button", { class: "btn block", onclick: openPreview }, "Preview"),
      start: h("button", { class: "btn btn-primary block", onclick: startJob }, "Start batch ▶"),
    };
    const card = h("div", { class: "card" }, el.name,
      h("div", { class: "stats" }, [el.episodes, el.each, el.bytes, el.time, el.steps, el.grid].map((s) => s.box)),
      el.status, el.warn, el.varied,
      h("div", { class: "actions", style: "margin-top:14px" }, el.preview, el.start));
    $("#summary").replaceChildren(card, h("div", { id: "summary-job" }));
    S.sum = el;
    return el;
  }
  function renderSummary() {
    const el = ensureSummary();
    const sp = S.spec, v = S.val, est = v.estimate;
    const running = S.jobs.filter((j) => j.status === "running" || j.status === "queued");
    const time = est ? STARTUP_S + est.total_steps / (S.rate || 40000) : null;
    const grid = gridSize();
    const varied = variedKeys();
    el.name.textContent = sp.name || "batch";
    el.episodes.v.textContent = fmtInt(sp.episodes);
    el.each.v.textContent = `${fmtNum(sp.duration_s)} s`;
    el.bytes.v.textContent = est ? fmtBytes(est.bytes) : "–";
    el.time.v.textContent = time != null ? fmtDur(time) : "–";
    el.steps.v.textContent = est ? fmtInt(est.total_steps) : "–";
    el.grid.v.textContent = grid > 1 ? fmtInt(grid) : String(varied.length);
    el.grid.k.textContent = grid > 1 ? "grid points" : "variables changed";
    if (v.pending) el.status.replaceChildren(h("div", { class: "status-line" }, h("span", { class: "spinner" }), "checking…"));
    else if (v.errors.length) el.status.replaceChildren(h("div", { class: "status-line bad" }, h("div", {},
      h("b", {}, `${v.errors.length} problem${v.errors.length > 1 ? "s" : ""} to fix`),
      v.errors.slice(0, 6).map((e) => h("div", { class: "issue", onclick: () => goTo(errorKey(e)) }, e)))));
    else el.status.replaceChildren(h("div", { class: "status-line ok" }, "✓ Ready to run"));
    el.warn.replaceChildren(!v.pending && v.warnings.length
      ? h("div", { class: "status-line warn" }, h("div", {}, v.warnings.map((w) => h("div", {}, "⚠ " + w)))) : "");
    el.varied.replaceChildren(varied.length ? h("div", {}, h("div", { class: "subhead", style: "margin-top:4px" }, "Changed"),
      h("div", { class: "varied" }, varied.map((k) => h("button", { class: "chip small", onclick: () => goTo(k) }, k)))) : "");
    const blocked = v.errors.length > 0;
    el.preview.disabled = blocked;
    el.start.disabled = blocked;
    el.start.textContent = running.length ? "Queue batch" : "Start batch ▶";
    renderJobPanel();
  }
  function renderJobPanel() {
    const box = document.getElementById("summary-job");
    if (!box) return;
    const running = S.jobs.filter((j) => j.status === "running" || j.status === "queued");
    box.replaceChildren(running.length || S.jobs.length ? h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h3", {}, "Current run"), h("a", { href: "#", onclick: (e) => { e.preventDefault(); S.section = "runs"; S.search = ""; $("#search").value = ""; render(); window.scrollTo({ top: 0 }); } }, "All runs →")),
      (running.length ? running : S.jobs.slice(0, 1)).map((j) => jobCard(j, true))) : "");
  }

  // ------------------------------------------------------------------ jobs
  async function startJob() {
    const est = S.val.estimate;
    const time = est ? STARTUP_S + est.total_steps / (S.rate || 40000) : 0;
    if ((est && est.bytes > 5e9) || time > 1800) {
      if (!confirm(`This batch will take about ${fmtDur(time)} and write about ${fmtBytes(est.bytes)}. Start it?`)) return;
    }
    try {
      const j = await api("POST", "/api/jobs", { spec: S.spec });
      toast(`Started “${j.name}”`, "ok");
      S.jobState[j.id] = j.status;
      await pollJobs();
    } catch (e) {
      toast(e.errors && e.errors.length ? `${e.message}: ${e.errors[0]}` : e.message, "bad");
    }
  }
  async function pollJobs() {
    try {
      const r = await api("GET", "/api/jobs");
      S.rate = r.rate || S.rate;
      if (!S.online) setOnline(true);
      const sig = JSON.stringify(r.jobs);
      if (sig === S.jobsSig) return;              // nothing changed: leave the page alone
      const activeBefore = S.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
      S.jobsSig = sig;
      S.jobs = r.jobs;
      let finished = false;
      for (const j of S.jobs) {
        const prev = S.jobState[j.id];
        if (prev && prev !== j.status && ["complete", "failed", "cancelled"].includes(j.status)) {
          finished = true;
          if (j.status === "complete") toast(`“${j.name}” finished: ${fmtInt(j.summary.episodes_written)} episodes, ${fmtBytes(j.summary.bytes)}`, "ok");
          else toast(`“${j.name}” ${j.status}${j.error ? ": " + j.error : ""}`, j.status === "failed" ? "bad" : "");
        }
        S.jobState[j.id] = j.status;
      }
      if (finished) await loadDatasets();
      renderJobPanel();
      const activeNow = S.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
      if (activeNow !== activeBefore) { renderSidebar(); renderSummary(); }
      const rj = document.getElementById("runs-jobs");
      if (rj) rj.replaceChildren(...runsJobsContent());
    } catch (e) { setOnline(false); }
  }
  function setOnline(ok) {
    S.online = ok;
    const el = $("#server-status");
    el.classList.toggle("bad", !ok);
    el.textContent = ok ? `connected · ${S.cpu} CPU cores · exports to ${S.root}` : "disconnected: is the server still running?";
  }

  // ------------------------------------------------------------------ preview + charts
  async function openPreview() {
    const dlg = $("#preview-dialog");
    if (!dlg.open) dlg.showModal();
    const body = $("#preview-body");
    body.replaceChildren(h("div", { class: "loading" }, h("span", { class: "spinner" }), "Simulating…"));
    const n = Number($("#preview-n").value);
    try {
      const r = await api("POST", "/api/preview", { spec: S.spec, n });
      renderPreview(r);
    } catch (e) {
      body.replaceChildren(h("div", { class: "error-text" }, e.message, (e.errors || []).map((x) => h("div", {}, x))));
    }
  }
  function renderPreview(r) {
    const eps = r.episodes;
    const body = $("#preview-body");
    const color = (i) => PALETTE[i % PALETTE.length];
    const label = (e) => `#${e.row.episode_id} ${e.row.tire.replaceAll("_", " ")} / ${e.row.surface.replaceAll("_", " ")}`;
    const charts = [
      { title: "Path (m)", xy: true, x: "x", y: "y" },
      { title: "Speed (m/s)", y: "speed" },
      { title: "Sideslip (deg), drift band ±20..80", y: "beta_deg", hlines: [20, -20, 80, -80] },
      { title: "Yaw rate (deg/s)", y: "yaw_rate_deg_s" },
      { title: "Steering command", y: "steer_cmd" },
      { title: "Throttle command", y: "throttle_cmd" },
    ];
    const canvases = charts.map(() => h("canvas"));
    const varied = variedKeys().filter((k) => k !== "tire" && k !== "surface");
    const fmtCell = (v) => Array.isArray(v) ? v.map(fmtNum).join(" ") : typeof v === "boolean" ? (v ? "on" : "off") : v == null ? "–" : fmtNum(v);
    body.replaceChildren(
      h("p", { class: "muted small", style: "margin-top:0" }, `First ${eps.length} episodes of this spec, simulated in ${r.seconds.toFixed(1)} s. Nothing was written.`),
      h("div", { class: "legend" }, eps.map((e, i) => h("span", {}, h("i", { style: `background:${color(i)}` }), label(e)))),
      h("div", { class: "charts" }, charts.map((c, i) => h("div", { class: "chart" }, h("h4", {}, c.title), canvases[i]))),
      h("h3", { style: "margin-top:16px" }, "Episodes"),
      h("div", { style: "overflow-x:auto" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "tire"), h("th", {}, "surface"), varied.map((k) => h("th", { class: "mono" }, k)),
          h("th", { class: "num" }, "max |sideslip|"), h("th", { class: "num" }, "drift time"), h("th", {}, "spun"), h("th", { class: "num" }, "max speed"))),
        h("tbody", {}, eps.map((e, i) => h("tr", {},
          h("td", {}, h("span", { style: `color:${color(i)}; font-weight:700` }, e.row.episode_id)),
          h("td", {}, e.row.tire), h("td", {}, e.row.surface),
          varied.map((k) => h("td", { class: "mono" }, fmtCell(e.row[k]))),
          h("td", { class: "num" }, `${(+e.row.max_abs_beta_deg).toFixed(1)}°`),
          h("td", { class: "num" }, `${(+e.row.drift_time_s).toFixed(2)} s`),
          h("td", {}, e.row.spun ? h("span", { class: "badge bad" }, "yes") : "no"),
          h("td", { class: "num" }, `${(+e.row.max_speed).toFixed(2)} m/s`)))))));
    const draw = () => charts.forEach((c, i) => drawChart(canvases[i], eps.map((e, k) => ({
      x: c.xy ? e[c.x] : e.t, y: e[c.y], color: color(k) })), { equal: !!c.xy, hlines: c.hlines, xLabel: c.xy ? "x (m)" : "t (s)" }));
    requestAnimationFrame(draw);
    S.redrawPreview = draw;
  }
  function niceTicks(lo, hi, n = 5) {
    if (!isFinite(lo) || !isFinite(hi)) return [];
    if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : +v.toPrecision(10));
    return out;
  }
  function drawChart(cv, series, { equal = false, hlines = [], xLabel = "" } = {}) {
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const cText = css.getPropertyValue("--muted").trim(), cGrid = css.getPropertyValue("--border").trim();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of series) for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i], y = s.y[i];
      if (x == null || y == null) continue;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (!isFinite(x0)) { ctx.fillStyle = cText; ctx.fillText("no finite data", 10, 20); return; }
    if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
    if (x1 - x0 < 1e-9) { x0 -= 1; x1 += 1; }
    const py = 0.06 * (y1 - y0); y0 -= py; y1 += py;
    const m = { l: 46, r: 10, t: 8, b: 26 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    if (equal) {
      const sx = (x1 - x0) / pw, sy = (y1 - y0) / ph, s = Math.max(sx, sy);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - (s * pw) / 2; x1 = cx + (s * pw) / 2; y0 = cy - (s * ph) / 2; y1 = cy + (s * ph) / 2;
    }
    const X = (x) => m.l + ((x - x0) / (x1 - x0)) * pw, Y = (y) => m.t + (1 - (y - y0) / (y1 - y0)) * ph;
    ctx.font = "11px system-ui, sans-serif"; ctx.lineWidth = 1;
    ctx.strokeStyle = cGrid; ctx.fillStyle = cText;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of niceTicks(y0, y1)) { ctx.beginPath(); ctx.moveTo(m.l, Y(t)); ctx.lineTo(W - m.r, Y(t)); ctx.stroke(); ctx.fillText(fmtNum(t), m.l - 6, Y(t)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(x0, x1, 6)) { ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), H - m.b); ctx.stroke(); ctx.fillText(fmtNum(t), X(t), H - m.b + 5); }
    ctx.textAlign = "right"; ctx.fillText(xLabel, W - m.r, H - 12);
    ctx.setLineDash([4, 4]);
    for (const hl of hlines) if (hl > y0 && hl < y1) { ctx.beginPath(); ctx.moveTo(m.l, Y(hl)); ctx.lineTo(W - m.r, Y(hl)); ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
    ctx.lineWidth = 1.6; ctx.lineJoin = "round";
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.beginPath();
      let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], y = s.y[i];
        if (x == null || y == null) { pen = false; continue; }
        if (pen) ctx.lineTo(X(x), Y(y)); else { ctx.moveTo(X(x), Y(y)); pen = true; }
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ top bar actions
  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const a = h("a", { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function setupTopbar() {
    const menu = $("#examples-menu");
    $("#examples-btn").addEventListener("click", (ev) => { ev.stopPropagation(); menu.classList.toggle("open"); $("#examples-btn").setAttribute("aria-expanded", menu.classList.contains("open")); });
    document.addEventListener("click", () => menu.classList.remove("open"));
    $("#import-btn").addEventListener("click", () => $("#import-file").click());
    $("#import-file").addEventListener("change", async (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      try {
        let spec = JSON.parse(await f.text());
        if (spec.spec && spec.format_version) spec = spec.spec;
        loadSpec(spec, f.name);
      } catch (e) { toast(`Could not read ${f.name}: ${e.message}`, "bad"); }
      ev.target.value = "";
    });
    $("#export-btn").addEventListener("click", () => download(`${S.spec.name || "batch"}.json`, JSON.stringify(S.spec, null, 2)));
    $("#reset-btn").addEventListener("click", () => { if (confirm("Discard all changes and start from the defaults?")) { loadSpec(S.defaultSpec, "defaults"); S.section = "batch"; render(); } });
    $("#json-btn").addEventListener("click", () => {
      $("#json-text").value = JSON.stringify(S.spec, null, 2); $("#json-error").textContent = "";
      $("#json-dialog").showModal();
    });
    $("#json-apply").addEventListener("click", () => {
      try { loadSpec(JSON.parse($("#json-text").value), "JSON"); $("#json-dialog").close(); }
      catch (e) { $("#json-error").textContent = e.message; }
    });
    $("#json-copy").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText($("#json-text").value); toast("Copied", "ok"); } catch (_) { $("#json-text").select(); }
    });
    document.querySelectorAll("dialog [data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));
    $("#preview-rerun").addEventListener("click", openPreview);
    $("#preview-n").addEventListener("change", openPreview);
    window.addEventListener("resize", debounce(() => { if ($("#preview-dialog").open && S.redrawPreview) S.redrawPreview(); }, 120));
    const search = $("#search");
    search.addEventListener("input", debounce(() => { S.search = search.value.trim(); renderSidebar(); renderContent(); }, 120));
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) { ev.preventDefault(); search.focus(); }
    });
    const adv = $("#advanced");
    adv.checked = S.advanced;
    adv.addEventListener("change", () => { S.advanced = adv.checked; store.set("driftsim.advanced", S.advanced); renderContent(); });
  }
  async function loadExamples() {
    try {
      const r = await api("GET", "/api/examples");
      S.examples = r.examples;
      const list = $("#examples-list");
      list.replaceChildren(...(r.examples.length ? r.examples.map((ex) => h("button", { class: "menu-item", role: "menuitem",
        onclick: () => { loadSpec(ex.spec, ex.file); S.section = "batch"; render(); } },
      ex.spec.name || ex.file,
      h("small", {}, `${fmtInt(ex.spec.episodes)} episodes × ${ex.spec.duration_s} s · ${((ex.spec.maneuver || {}).type || "").replaceAll("_", " ")}`)))
        : [h("div", { class: "menu-item muted" }, "No examples found (examples/specs/)")]));
    } catch (_) { /* optional */ }
  }

  // ------------------------------------------------------------------ start
  async function init() {
    setupTopbar();
    try {
      const r = await api("GET", "/api/catalog");
      S.cat = r.catalog; S.defaultSpec = r.default_spec; S.cpu = r.cpu_count; S.rate = r.rate; S.root = r.export_root;
      S.devices = r.devices || { cpu: true };
      for (const g of S.cat.groups) { S.groups[g.id] = g; for (const f of g.fields) S.fields[f.key] = { ...f, group: g.id }; }
      setOnline(true);
    } catch (e) {
      $("#content").replaceChildren(h("div", { class: "card error-text" }, `Could not reach the server: ${e.message}`));
      setOnline(false);
      return;
    }
    const draft = store.get(DRAFT_KEY);
    S.spec = mergeSpec(draft || S.defaultSpec);
    render();
    changed();
    loadExamples();
    loadDatasets();
    pollJobs();
    setInterval(pollJobs, 1000);
    setInterval(() => { if (S.section === "runs") loadDatasets(); }, 5000);
  }
  init();
})();
