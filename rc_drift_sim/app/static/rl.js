/* DriftSim GUI: the reinforcement-learning pages.
 *   RL training - every environment, reward, PPO and compute setting, validated live; Start.
 *   RL runs     - all training runs; for the selected one: live learning curves against the reference
 *                 controllers, a timeline of policy snapshots recorded during training, and an animated
 *                 top-down viewer of any snapshot (or a fresh episode of the latest policy) with charts.
 * Uses the helpers app.js exposes as window.DriftSim. Plain JavaScript, no build step. */
"use strict";
(() => {
  const D = window.DriftSim;
  const { S, h, $, api, toast, store, clone, r6, debounce, fmtInt, fmtNum, fmtDur, fmtDate, PALETTE, numInput, drawChart } = D;
  const KEY = "driftsim.rl.v1";
  const R = (S.rl = {
    available: null, reason: "", cat: null, devices: { cpu: true }, root: "",
    cfg: null, val: { errors: [], iterations: null, samples: null, pending: false },
    runs: [], runsSig: "", view: null, detail: null, hist: [], snapName: null, snap: null, follow: true,
    play: { t: 0, playing: false, speed: 1, last: 0, raf: 0 }, rows: {}, dash: null, sum: null,
  });
  const PARTS = ["env", "noise", "reward", "ppo", "run"];
  const TASK_TEXT = {
    hold: "Hold a steady drift: sideslip near the target, either direction, with the car rotating into the slide, at the target speed. Position does not matter (a donut).",
    track: "Drive round a circle at the target speed, with a bonus for drifting into the turn. Straying too far from the line ends the episode.",
  };
  const isActive = (r) => r && (r.status === "running" || r.status === "queued");

  // ------------------------------------------------------------------ settings
  function defaults() {
    const d = R.cat.defaults;
    return { name: "drift", env: clone(d.env), noise: clone(d.noise), reward: clone(d.reward), ppo: clone(d.ppo), run: clone(d.run),
      randomize_mode: "none", randomize: {} };
  }
  function mergeCfg(c) {
    const out = defaults();
    if (!c || typeof c !== "object") return out;
    if (typeof c.name === "string") out.name = c.name;
    for (const part of PARTS) {
      const g = (c[part] && typeof c[part] === "object") ? c[part] : {};
      for (const f of R.cat[part]) if (g[f.key] !== undefined) out[part][f.key] = g[f.key];
    }
    if (["none", "dataset", "custom"].includes(c.randomize_mode)) out.randomize_mode = c.randomize_mode;
    if (c.randomize && typeof c.randomize === "object") {
      out.randomize = clone(c.randomize);
      if (!c.randomize_mode && Object.keys(c.randomize).length) out.randomize_mode = "custom";
    }
    if (out.randomize_mode === "custom" && !Object.keys(out.randomize).length) out.randomize_mode = "none";
    return out;
  }
  function randomizeDict() {
    if (R.cfg.randomize_mode === "dataset") return clone(S.spec ? S.spec.params : {});
    if (R.cfg.randomize_mode === "custom") return clone(R.cfg.randomize);
    return {};
  }
  function body() {
    const c = R.cfg;
    return { name: c.name, env: c.env, noise: c.noise, reward: c.reward, ppo: c.ppo, run: c.run,
      randomize_mode: c.randomize_mode, randomize: randomizeDict() };
  }
  function changedCount() {
    if (!R.cat || !R.cfg) return 0;
    let n = 0;
    for (const part of PARTS) for (const f of R.cat[part]) if (JSON.stringify(R.cfg[part][f.key]) !== JSON.stringify(f.default)) n++;
    return n + (R.cfg.randomize_mode !== "none" ? 1 : 0);
  }
  function save() { store.set(KEY, R.cfg); }
  function setVal(part, key, v, rerender = false) {
    R.cfg[part][key] = v;
    save(); validate(); R.val.pending = true; updateSum();
    if (rerender) D.renderContent();
    D.renderSidebar();
  }
  const validate = debounce(async () => {
    const sent = JSON.stringify(body());
    try {
      const r = await api("POST", "/api/rl/validate", JSON.parse(sent));
      if (JSON.stringify(body()) !== sent) return;
      R.val = { errors: r.errors, iterations: r.iterations, samples: r.samples_per_iteration, pending: false };
    } catch (e) { R.val = { errors: [e.message], iterations: null, samples: null, pending: false }; }
    updateSum(); D.renderSidebar();
  }, 250);

  // ------------------------------------------------------------------ setup page
  function control(part, f) {
    const v = R.cfg[part][f.key];
    if (f.type === "choice") {
      const devs = R.devices || {};
      const sel = h("select", { "aria-label": f.label, onchange: () => setVal(part, f.key, sel.value, f.key === "task" || f.key === "obs" || f.key === "device") },
        f.choices.map((c) => {
          const off = part === "run" && f.key === "device" && (c === "mps" || c === "cuda") && !devs[c];
          const lbl = part === "run" && f.key === "device"
            ? { cpu: "CPU (NumPy physics)", mps: "Apple GPU (MPS)", cuda: "NVIDIA GPU (CUDA)", auto: `Best available (${devs.cuda ? "CUDA" : devs.mps ? "MPS" : "CPU"})` }[c] + (off ? " (not available)" : "")
            : c;
          return h("option", { value: c, selected: c === v, disabled: off }, lbl);
        }));
      return sel;
    }
    if (f.type === "bool") {
      const i = h("input", { type: "checkbox", checked: !!v, onchange: () => setVal(part, f.key, i.checked) });
      return h("label", { class: "switch" }, i, h("span"));
    }
    if (f.type === "intlist") {
      const i = h("input", { type: "text", value: (v || []).join(", "), style: "width:140px", "aria-label": f.label });
      i.addEventListener("input", () => {
        const vals = i.value.split(/[,\s]+/).filter(Boolean).map(Number);
        const ok = vals.length && vals.every((x) => Number.isInteger(x) && x > 0);
        i.classList.toggle("invalid", !ok);
        if (ok) setVal(part, f.key, vals);
      });
      return i;
    }
    return numInput(v, (x) => setVal(part, f.key, f.type === "int" ? Math.round(x) : x), { width: "120px", label: f.label });
  }
  function fieldRows(part, filter = () => true) {
    const task = R.cfg.env.task;
    return R.cat[part].filter((f) => (!f.tasks || f.tasks.includes(task)) && (f.level !== "advanced" || S.advanced) && filter(f)).map((f) => {
      const changed = JSON.stringify(R.cfg[part][f.key]) !== JSON.stringify(f.default);
      return [
        h("label", { class: changed ? "changed" : "", title: f.desc }, f.label),
        h("div", { class: "ctl" }, control(part, f), f.unit ? h("span", { class: "lbl" }, f.unit) : null,
          changed ? h("button", { class: "btn small btn-ghost", title: `back to ${Array.isArray(f.default) ? f.default.join(", ") : fmtNum(f.default)}`,
            onclick: () => setVal(part, f.key, clone(f.default), true) }, "↺") : null,
          h("span", { class: "muted small rl-desc" }, f.desc)),
      ];
    });
  }
  function card(title, intro, ...kids) {
    return h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, title)), intro ? h("p", { class: "muted small", style: "margin-top:0" }, intro) : null, ...kids);
  }
  function renderSetup() {
    if (R.available === null) return h("div", { class: "loading" }, h("span", { class: "spinner" }), "Loading…");
    if (!R.available) return card("Reinforcement learning", null, h("div", { class: "status-line warn" }, R.reason));
    const c = R.cfg, task = c.env.task, w = c.reward;
    const maxStep = task === "hold" ? w.beta + w.speed : w.track + w.speed + w.drift;
    const taskSeg = h("div", { class: "seg" }, ["hold", "track"].map((t) => h("button", { class: t === task ? "on" : "", onclick: () => setVal("env", "task", t, true) },
      t === "hold" ? "Hold a drift" : "Drift round a circle")));
    const rmode = h("div", { class: "seg" }, [["none", "None"], ["dataset", "Variables of the dataset pages"], ["custom", "Loaded set"]].map(([m, l]) =>
      h("button", { class: c.randomize_mode === m ? "on" : "", disabled: m === "custom" && !Object.keys(c.randomize).length,
        onclick: () => { c.randomize_mode = m; save(); validate(); D.renderContent(); D.renderSidebar(); updateSum(); } }, l)));
    const rdict = randomizeDict(), rkeys = Object.keys(rdict);
    return h("div", {},
      h("div", { class: "card rl-intro" },
        h("div", { class: "card-head" }, h("h2", {}, "Reinforcement learning")),
        h("p", { class: "small", style: "margin:0" },
          "The car learns to drift by trial and error: thousands of simulated cars drive at once, the PPO trainer rewards what worked, ",
          "and every few iterations the current policy drives one recorded episode. Set the parameters here, press Start training, ",
          "then watch the learning curves and the recorded episodes under ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); go("rlruns"); } }, "RL runs"), ".")),
      card("Task", TASK_TEXT[task], h("div", { style: "margin-bottom:12px" }, taskSeg),
        h("div", { class: "form-grid" }, fieldRows("env", (f) => !["task", "obs"].includes(f.key)))),
      card("Rewards", `Per-step reward (at most ${fmtNum(maxStep)} per step, ${fmtNum(maxStep * c.env.episode_s * 50)} per episode) minus the smoothness penalty; ending early costs the early-end penalty.`,
        h("div", { class: "form-grid" }, fieldRows("reward"))),
      card("Observations & sensor noise", "What the policy sees. With sensors it gets gyro, accelerometer, wheel speeds and a velocity estimate, each with Gaussian noise, like the real car.",
        h("div", { class: "form-grid" }, fieldRows("env", (f) => f.key === "obs"),
          c.env.obs === "sensors" ? fieldRows("noise") : [])),
      card("Domain randomization", "Every episode can draw its own car. Use the variables you set on the dataset pages (Chassis, Tires, Surface, Tire condition, Starting state, …): fixed values, random distributions and lists all apply per episode.",
        h("div", { style: "margin-bottom:10px" }, rmode),
        c.randomize_mode === "none" ? h("div", { class: "muted small" }, "Every episode uses the default car (hard plastic drift tires on epoxy P-tile).")
          : rkeys.length ? h("div", { class: "varied" }, rkeys.map((k) => h("button", { class: "chip small", title: JSON.stringify(rdict[k]),
            onclick: () => { if (c.randomize_mode === "dataset") D.goTo(k); } }, k)))
            : h("div", { class: "status-line warn" }, "No variables are set on the dataset pages yet: open Chassis, Tires, Surface, … and set some to random."),
        c.randomize_mode === "custom" ? h("div", { style: "margin-top:10px" }, h("button", { class: "btn small", onclick: copyToDataset }, "Copy into the dataset pages to edit")) : null),
      card("PPO trainer", S.advanced ? null : "Tick Advanced (top) for epochs, minibatches, GAE lambda, clipping and the other fine-tuning settings.",
        h("div", { class: "form-grid" }, fieldRows("ppo"))),
      card("Compute", "Where the cars are simulated and the network is trained. The CPU uses the exact NumPy physics; a GPU pays off from a few thousand cars.",
        h("div", { class: "form-grid" }, fieldRows("run"))));
  }
  function copyToDataset() {
    if (!confirm("Replace the variables on the dataset pages with this run's randomization?")) return;
    S.spec.params = clone(R.cfg.randomize);
    store.set("driftsim.spec.v1", S.spec);
    R.cfg.randomize_mode = "dataset"; save(); validate();
    toast("Copied: edit them on the dataset pages", "ok");
    D.render();
  }
  function go(section) { S.section = section; S.search = ""; $("#search").value = ""; D.render(); window.scrollTo({ top: 0 }); }

  // ------------------------------------------------------------------ summary panel (both RL pages)
  function ensureSum() {
    if (R.sum) return R.sum;
    const stat = (k) => { const v = h("div", { class: "v" }); return { box: h("div", { class: "stat" }, v, h("div", { class: "k" }, k)), v }; };
    const name = h("input", { type: "text", "aria-label": "Run name", style: "width:100%" });
    name.addEventListener("input", () => { R.cfg.name = name.value; save(); });
    const el = { name, task: stat("task"), cars: stat("cars in parallel"), steps: stat("training steps"), iters: stat("iterations"),
      device: stat("device"), samples: stat("samples / iteration"), status: h("div"), run: h("div") };
    el.start = h("button", { class: "btn btn-primary block", onclick: startRun }, "Start training ▶");
    el.root = h("div", { style: "display:flex;flex-direction:column;gap:14px" },
      h("div", { class: "card" }, h("div", { class: "subhead", style: "margin-top:0" }, "Run name"), name,
        h("div", { class: "stats" }, [el.task, el.cars, el.steps, el.iters, el.samples, el.device].map((s) => s.box)),
        el.status, h("div", { class: "actions", style: "margin-top:12px" }, el.start)),
      el.run);
    R.sum = el;
    return el;
  }
  function renderSum() {
    const el = ensureSum();
    if (!$("#summary").contains(el.root)) $("#summary").replaceChildren(el.root);
    updateSum();
  }
  function updateSum() {
    const el = R.sum;
    if (!el || !R.cfg || !$("#summary").contains(el.root)) return;
    const c = R.cfg;
    if (document.activeElement !== el.name) el.name.value = c.name;
    el.task.v.textContent = c.env.task === "hold" ? "hold drift" : "circle";
    el.cars.v.textContent = fmtInt(c.ppo.num_envs);
    el.steps.v.textContent = c.ppo.total_steps >= 1e6 ? `${r6(c.ppo.total_steps / 1e6)} M` : fmtInt(c.ppo.total_steps);
    el.iters.v.textContent = R.val.iterations != null ? fmtInt(R.val.iterations) : "–";
    el.samples.v.textContent = R.val.samples != null ? fmtInt(R.val.samples) : "–";
    el.device.v.textContent = c.run.device;
    const v = R.val;
    if (!R.available) el.status.replaceChildren(h("div", { class: "status-line warn" }, R.reason || "checking…"));
    else if (v.pending) el.status.replaceChildren(h("div", { class: "status-line" }, h("span", { class: "spinner" }), "checking…"));
    else if (v.errors.length) el.status.replaceChildren(h("div", { class: "status-line bad" }, h("div", {},
      h("b", {}, `${v.errors.length} problem${v.errors.length > 1 ? "s" : ""} to fix`), v.errors.slice(0, 6).map((e) => h("div", {}, e)))));
    else el.status.replaceChildren(h("div", { class: "status-line ok" }, "✓ Ready to train"));
    const active = R.runs.filter(isActive);
    el.start.disabled = !R.available || v.errors.length > 0 || v.pending;
    el.start.textContent = active.length ? "Queue training" : "Start training ▶";
    const cur = active[0] || R.runs[0];
    el.run.replaceChildren(cur ? h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h3", {}, isActive(cur) ? "Training now" : "Latest run"),
        h("a", { href: "#", onclick: (e) => { e.preventDefault(); openRun(cur.id); } }, "Watch →")),
      runLine(cur)) : "");
  }
  function runLine(r) {
    const frac = r.n_iter ? r.iteration / r.n_iter : 0;
    return h("div", { class: "job" },
      h("div", { class: "job-head" }, h("b", {}, r.name), h("span", { class: statusClass(r.status) }, r.status)),
      isActive(r) ? h("div", { class: `progress ${r.status === "queued" || !r.iteration ? "indet" : ""}` }, h("div", { style: `width:${(100 * frac).toFixed(1)}%` })) : null,
      h("div", { class: "job-meta" }, isActive(r)
        ? `${r.stage || ""} · iteration ${fmtInt(r.iteration)} / ${fmtInt(r.n_iter)} · return ${r.last_return ?? "–"} · ETA ${fmtDur(r.eta_s)}`
        : `${fmtInt(r.iteration)} iterations · best return ${r.best_return ?? "–"}${r.error ? " · " + r.error : ""}`));
  }
  const statusClass = (s) => ({ queued: "badge", running: "badge run", complete: "badge ok", failed: "badge bad", cancelled: "badge", interrupted: "badge bad" }[s] || "badge");

  async function startRun() {
    try {
      const r = await api("POST", "/api/rl/runs", body());
      toast(`Started “${r.name}”`, "ok");
      await pollRuns(true);
      openRun(r.id);
    } catch (e) { toast(e.message, "bad"); }
  }

  // ------------------------------------------------------------------ runs page
  function renderRunsPage() {
    if (R.available === null) return h("div", { class: "loading" }, h("span", { class: "spinner" }), "Loading…");
    R.rows = {};
    const tbody = h("tbody");
    R.tbody = tbody;
    const table = h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "run"), h("th", {}, "status"), h("th", {}, "task"), h("th", { class: "num" }, "iterations"),
        h("th", { class: "num" }, "best return"), h("th", {}, "started"), h("th", {}, ""))), tbody);
    R.dash = h("div");
    const page = h("div", {},
      !R.available ? h("div", { class: "card" }, h("div", { class: "status-line warn" }, R.reason)) : null,
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, "Training runs"),
          h("div", { class: "row-actions" }, h("span", { class: "muted small mono" }, R.root), h("button", { class: "btn small", onclick: () => pollRuns(true) }, "Refresh"))),
        R.runsEmpty = h("div", { class: "empty", hidden: R.runs.length > 0 }, "No training runs yet. Set things up under RL training and press Start."),
        h("div", { style: "overflow-x:auto" }, table)),
      R.dash);
    setTimeout(() => { updateRunsTable(); if (R.view) buildDash(); }, 0);    // once the page is in the document
    return page;
  }
  function updateRunsTable() {
    if (!R.tbody || !document.body.contains(R.tbody)) return;
    if (R.runsEmpty) R.runsEmpty.hidden = R.runs.length > 0;
    const seen = new Set();
    R.runs.forEach((r, i) => {
      seen.add(r.id);
      let row = R.rows[r.id];
      if (!row) {
        const c = { name: h("td"), status: h("td"), task: h("td"), it: h("td", { class: "num" }), best: h("td", { class: "num" }), date: h("td") };
        const stop = h("button", { class: "btn small btn-danger", onclick: () => stopRun(r.id) }, "Stop");
        const acts = h("td", {}, h("div", { class: "row-actions" },
          h("button", { class: "btn small", onclick: () => openRun(r.id) }, "View"), stop,
          h("button", { class: "btn small", title: "Load this run's settings into RL training", onclick: () => loadRunSettings(r.id) }, "Settings"),
          h("a", { class: "btn small", href: `/api/rl/runs/${encodeURIComponent(r.id)}/files/policy.pt`, title: "Download the trained policy (PyTorch)" }, "policy.pt"),
          h("button", { class: "btn small", onclick: () => api("POST", `/api/rl/runs/${encodeURIComponent(r.id)}/reveal`, {}).catch((e) => toast(e.message, "bad")) }, "Folder")));
        const tr = h("tr", { "data-id": r.id }, c.name, c.status, c.task, c.it, c.best, c.date, acts);
        row = R.rows[r.id] = { tr, c, stop };
      }
      const c = row.c;
      c.name.replaceChildren(h("b", {}, r.name), h("div", { class: "muted small mono" }, r.id));
      c.status.replaceChildren(h("span", { class: statusClass(r.status) }, r.status));
      c.task.textContent = `${r.task} · ${r.device}`;
      c.it.textContent = `${fmtInt(r.iteration)} / ${fmtInt(r.n_iter)}`;
      c.best.textContent = r.best_return != null ? fmtNum(r.best_return) : "–";
      c.date.textContent = fmtDate(r.created);
      row.stop.hidden = !isActive(r);
      row.tr.classList.toggle("selected", r.id === R.view);
      if (R.tbody.children[i] !== row.tr) R.tbody.insertBefore(row.tr, R.tbody.children[i] || null);
    });
    for (const id of Object.keys(R.rows)) if (!seen.has(id)) { R.rows[id].tr.remove(); delete R.rows[id]; }
  }
  async function stopRun(id) {
    try { await api("POST", `/api/rl/runs/${encodeURIComponent(id)}/stop`, {}); toast("Stopping… the last saved policy is kept."); pollRuns(true); }
    catch (e) { toast(e.message, "bad"); }
  }
  async function loadRunSettings(id) {
    try {
      const d = await api("GET", `/api/rl/runs/${encodeURIComponent(id)}?since=999999999`);
      R.cfg = mergeCfg(d.config); save(); validate();
      toast(`Loaded the settings of ${d.name}`, "ok");
      go("rl");
    } catch (e) { toast(e.message, "bad"); }
  }
  function openRun(id) {
    if (R.view !== id) { R.view = id; R.detail = null; R.hist = []; R.snapName = null; R.snap = null; R.follow = true; stopPlay(); }
    if (S.section !== "rlruns") go("rlruns"); else { updateRunsTable(); buildDash(); }
    pollView();
  }

  // ------------------------------------------------------------------ dashboard of one run
  const LEARN = [
    { title: "Episode return (finished episodes, mean per iteration)", key: "ret", refs: true },
    { title: "Episode length (control steps)", key: "len" },
    { title: "Reward per step", key: "rps" },
    { title: "Action noise (std): steer, throttle", key: "std" },
    { title: "Value loss", key: "vloss" },
    { title: "Policy change per iteration (approx. KL)", key: "kl" },
  ];
  function buildDash() {
    if (!R.dash || !R.view) return;
    const d = {};
    d.title = h("h2", {}, R.view);
    d.badge = h("span", { class: "badge" }, "…");
    d.meta = h("div", { class: "job-meta" });
    d.bar = h("div", { class: "progress" }, h("div", { style: "width:0%" }));
    d.refs = h("div", { class: "varied", style: "margin-top:8px" });
    d.stop = h("button", { class: "btn small btn-danger", onclick: () => stopRun(R.view) }, "Stop training");
    d.curves = LEARN.map(() => h("canvas"));
    d.timeline = h("div", { class: "timeline" });
    d.thumbs = {};
    d.scene = h("canvas", { class: "scene" });
    d.hud = h("div", { class: "hud mono" });
    d.slider = h("input", { type: "range", min: 0, max: 1000, value: 0, "aria-label": "Time" });
    d.slider.addEventListener("input", () => { R.play.t = (d.slider.value / 1000) * snapDuration(); R.play.playing = false; d.playBtn.textContent = "▶"; drawFrame(); });
    d.playBtn = h("button", { class: "btn small", onclick: togglePlay }, "▶");
    d.speedSel = h("select", { "aria-label": "Playback speed", onchange: () => { R.play.speed = Number(d.speedSel.value); } },
      [0.25, 0.5, 1, 2, 4].map((s) => h("option", { value: s, selected: s === R.play.speed }, `${s}×`)));
    d.time = h("span", { class: "mono small" }, "0.00 s");
    d.snapLabel = h("div", { class: "muted small" }, R.snap ? R.snapText : "No episode loaded yet: snapshots appear as training runs.");
    d.seed = h("input", { type: "number", value: 0, style: "width:70px", "aria-label": "Seed" });
    d.startSel = h("select", { "aria-label": "Start" }, h("option", { value: "parked" }, "start parked"), h("option", { value: "drift" }, "start drifting"));
    d.follow = h("input", { type: "checkbox", checked: R.follow, onchange: () => { R.follow = d.follow.checked; } });
    d.lqrBtn = h("button", { class: "btn small", onclick: () => liveRollout("lqr"), title: "The model-based reference controller (hold task)" }, "LQR reference");
    d.epCharts = [h("canvas"), h("canvas"), h("canvas"), h("canvas"), h("canvas")];
    d.epTitles = d.epCharts.map(() => h("h4"));
    d.epLegends = d.epCharts.map(() => h("div", { class: "legend", style: "margin:2px 0 4px" }));
    R.dashEls = d;
    R.dash.replaceChildren(
      h("div", { class: "card" },
        h("div", { class: "card-head" }, d.title, h("div", { class: "row-actions" }, d.badge, d.stop,
          h("a", { class: "btn small", href: `/api/rl/runs/${encodeURIComponent(R.view)}/files/policy.pt` }, "Download policy"),
          h("a", { class: "btn small", href: `/api/rl/runs/${encodeURIComponent(R.view)}/files/log.jsonl` }, "log.jsonl"))),
        d.bar, d.meta, d.refs),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Learning curves"),
        h("span", { class: "muted small" }, "x: environment steps (millions); dashed: reference controllers")),
        h("div", { class: "charts" }, LEARN.map((c, i) => h("div", { class: "chart" }, h("h4", {}, c.title), d.curves[i])))),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Training timeline"),
        h("label", { class: "toggle small" }, d.follow, h("span", {}, "follow the newest snapshot"))),
        h("p", { class: "muted small", style: "margin-top:0" }, "One recorded episode of the policy every few iterations (deterministic actions, the car starts parked). Click one to play it below. Trail colour: blue grip, orange drift, red spin."),
        d.timeline),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Episode viewer"),
        h("div", { class: "row-actions" }, h("span", { class: "muted small" }, "seed"), d.seed, d.startSel,
          h("button", { class: "btn small", onclick: () => liveRollout("policy") }, "Run latest policy"), d.lqrBtn,
          h("button", { class: "btn small", onclick: () => liveRollout("zero") }, "No input"))),
        d.snapLabel,
        h("div", { class: "viewer" }, d.scene, d.hud),
        h("div", { class: "playbar" }, d.playBtn, d.slider, d.time, d.speedSel),
        h("div", { class: "charts", style: "margin-top:12px" }, d.epCharts.map((cv, i) => h("div", { class: "chart" }, d.epTitles[i], d.epLegends[i], cv)))));
    updateDash(true);
    if (R.snap) { drawFrame(); }
  }
  function updateDash(full = false) {
    const d = R.dashEls, x = R.detail;
    if (!d || !x || !document.body.contains(d.title)) return;
    d.title.textContent = x.name;
    d.badge.className = statusClass(x.status); d.badge.textContent = x.status;
    d.stop.hidden = !isActive(x);
    d.lqrBtn.hidden = (x.task || (x.config.env || {}).task) !== "hold";
    const frac = x.n_iter ? x.iteration / x.n_iter : 0;
    d.bar.hidden = !isActive(x);
    d.bar.className = `progress ${x.status === "queued" || !x.iteration ? "indet" : ""}`;
    d.bar.firstChild.style.width = `${(100 * frac).toFixed(1)}%`;
    const last = R.hist[R.hist.length - 1] || {};
    d.meta.textContent = [
      isActive(x) ? x.stage : x.status,
      `iteration ${fmtInt(x.iteration)} / ${fmtInt(x.n_iter)}`,
      `${r6((x.env_steps || 0) / 1e6)} M / ${r6((x.total_steps || 0) / 1e6)} M steps`,
      last.steps_per_s ? `${fmtInt(last.steps_per_s)} steps/s` : null,
      `elapsed ${fmtDur(x.seconds)}`, isActive(x) && x.eta_s != null ? `ETA ${fmtDur(x.eta_s)}` : null,
      `best return ${x.best_return ?? "–"}`, x.error ? `error: ${x.error}` : null,
    ].filter(Boolean).join(" · ");
    const refs = Object.entries(x.baselines || {});
    d.refs.replaceChildren(...(refs.length ? [h("span", { class: "muted small" }, "reference scores:"),
      ...refs.map(([k, v], i) => h("span", { class: "chip small", style: `border-color:${PALETTE[(i + 2) % PALETTE.length]}` }, `${k}: ${v}`))] : []));
    drawCurves();
    updateTimeline();
  }
  function series(key) {
    const H = R.hist, xs = H.map((r) => r.env_steps / 1e6);
    const pick = (f) => H.map(f);
    if (key === "ret") return [{ x: xs, y: pick((r) => r.episode_return), color: PALETTE[0] }];
    if (key === "len") return [{ x: xs, y: pick((r) => r.episode_length), color: PALETTE[0] }];
    if (key === "rps") return [{ x: xs, y: pick((r) => r.reward_per_step), color: PALETTE[0] }];
    if (key === "std") return [{ x: xs, y: pick((r) => r.action_std && r.action_std[0]), color: PALETTE[0] },
      { x: xs, y: pick((r) => r.action_std && r.action_std[1]), color: PALETTE[1] }];
    if (key === "vloss") return [{ x: xs, y: pick((r) => r.value_loss), color: PALETTE[0] }];
    return [{ x: xs, y: pick((r) => r.approx_kl), color: PALETTE[0] }];
  }
  function drawCurves() {
    const d = R.dashEls, x = R.detail;
    if (!d || !x) return;
    const refs = Object.entries(x.baselines || {}).map(([k, v], i) => ({ y: v, label: `${k} ${v}`, color: PALETTE[(i + 2) % PALETTE.length] }));
    LEARN.forEach((c, i) => drawChart(d.curves[i], series(c.key), { hlines: c.refs ? refs : [], xLabel: "M steps" }));
  }
  function updateTimeline() {
    const d = R.dashEls, x = R.detail;
    if (!d || !x) return;
    const snaps = x.snapshots || [];
    if (!snaps.length && !d.timeline.children.length) d.timeline.replaceChildren(h("div", { class: "muted small" }, isActive(x) ? "The first snapshot is recorded after the first iteration." : "This run has no snapshots."));
    let added = null;
    for (const s of snaps) {
      if (d.thumbs[s.name]) continue;
      if (!Object.keys(d.thumbs).length) d.timeline.replaceChildren();
      const cv = h("canvas", { width: 300, height: 220 });
      const el = h("button", { class: "thumb", title: `iteration ${s.iteration}`, onclick: () => { R.follow = false; d.follow.checked = false; loadSnapshot(s.name); } },
        cv, h("div", { class: "small" }, h("b", {}, `it ${s.iteration}`), ` · ${r6(s.env_steps / 1e6)} M`),
        h("div", { class: "small muted" }, `return ${fmtNum(s.episode_return)} · ${s.ended === "time limit" ? `${(s.steps / 50).toFixed(0)} s` : s.ended}`));
      d.thumbs[s.name] = { el, cv, drawn: false };
      d.timeline.append(el);
      added = s.name;
      fetchThumb(s.name);
    }
    for (const [n, t] of Object.entries(d.thumbs)) t.el.classList.toggle("active", n === R.snapName);
    if (added && (R.follow || !R.snap)) loadSnapshot(added);
  }
  const snapCache = {};
  async function getSnap(name) {
    const k = `${R.view}/${name}`;
    if (!snapCache[k]) snapCache[k] = api("GET", `/api/rl/runs/${encodeURIComponent(R.view)}/snapshots/${encodeURIComponent(name)}`);
    return snapCache[k];
  }
  async function fetchThumb(name) {
    try {
      const data = await getSnap(name);
      const t = R.dashEls && R.dashEls.thumbs[name];
      if (t) drawScene(t.cv, data, data.series.t.length - 1, { thumb: true });
    } catch (_) { /* shown on click */ }
  }
  async function loadSnapshot(name) {
    try {
      const data = await getSnap(name);
      showEpisode(data, name, `Snapshot after iteration ${data.iteration} (${r6(data.env_steps / 1e6)} M steps): return ${fmtNum(data.episode_return)}, ${describeEnd(data)}.`);
    } catch (e) { toast(e.message, "bad"); }
  }
  async function liveRollout(policy) {
    const d = R.dashEls;
    d.snapLabel.textContent = "Simulating one episode…";
    try {
      const data = await api("POST", `/api/rl/runs/${encodeURIComponent(R.view)}/rollout`,
        { policy, seed: Number(d.seed.value) || 0, init_drift: d.startSel.value === "drift" });
      R.follow = false; d.follow.checked = false;
      const who = { policy: "Latest policy", lqr: "LQR reference controller (reads the true state)", zero: "No input" }[policy];
      showEpisode(data, null, `${who}, seed ${Number(d.seed.value) || 0}, ${d.startSel.value === "drift" ? "starting in a drift" : "starting parked"}: return ${fmtNum(data.episode_return)}, ${describeEnd(data)}.`);
    } catch (e) { d.snapLabel.textContent = e.message; toast(e.message, "bad"); }
  }
  const describeEnd = (d) => d.ended === "time limit" ? `drove the full ${(d.steps * d.dt).toFixed(1)} s` : `ended by ${d.ended} after ${(d.steps * d.dt).toFixed(2)} s`;
  function showEpisode(data, name, label) {
    R.snap = data; R.snapName = name; R.snapText = label;
    const d = R.dashEls;
    if (!d) return;
    d.snapLabel.textContent = label;
    for (const [n, t] of Object.entries(d.thumbs)) t.el.classList.toggle("active", n === name);
    R.play.t = 0; R.play.playing = true; d.playBtn.textContent = "❚❚";
    startLoop();
  }

  // ------------------------------------------------------------------ playback
  const snapDuration = () => (R.snap ? R.snap.series.t[R.snap.series.t.length - 1] : 0);
  function togglePlay() {
    if (!R.snap) return;
    if (R.play.t >= snapDuration()) R.play.t = 0;
    R.play.playing = !R.play.playing;
    R.dashEls.playBtn.textContent = R.play.playing ? "❚❚" : "▶";
    startLoop();
  }
  function stopPlay() { R.play.playing = false; cancelAnimationFrame(R.play.raf); R.play.raf = 0; }
  function startLoop() {
    cancelAnimationFrame(R.play.raf);
    R.play.last = performance.now();
    let lastCharts = 0;
    const tick = (now) => {
      const d = R.dashEls;
      if (!d || !document.body.contains(d.scene)) { R.play.raf = 0; return; }
      if (R.play.playing) {
        R.play.t += ((now - R.play.last) / 1000) * R.play.speed;
        if (R.play.t >= snapDuration()) { R.play.t = snapDuration(); R.play.playing = false; d.playBtn.textContent = "▶"; }
      }
      R.play.last = now;
      drawFrame(now - lastCharts > 50 ? (lastCharts = now, true) : false);
      R.play.raf = R.play.playing ? requestAnimationFrame(tick) : 0;
    };
    R.play.raf = requestAnimationFrame(tick);
  }
  function frameIndex() {
    const s = R.snap.series, n = s.t.length;
    return Math.max(0, Math.min(n - 1, Math.round(R.play.t / R.snap.dt)));
  }
  function drawFrame(charts = true) {
    const d = R.dashEls;
    if (!d || !R.snap) return;
    const i = frameIndex(), s = R.snap.series;
    drawScene(d.scene, R.snap, i);
    d.slider.value = String(Math.round(1000 * (snapDuration() ? R.play.t / snapDuration() : 0)));
    d.time.textContent = `${s.t[i].toFixed(2)} s`;
    const k = Math.max(0, i - 1);
    d.hud.replaceChildren(...[
      `t ${s.t[i].toFixed(2)} s`, `v ${s.speed[i].toFixed(2)} m/s`, `β ${s.beta_deg[i].toFixed(1)}°`, `r ${s.yaw_rate_deg_s[i].toFixed(0)}°/s`,
      `δ ${s.delta_deg[i].toFixed(1)}°`, s.steer.length ? `cmd ${s.steer[k].toFixed(2)} ${s.throttle[k].toFixed(2)}` : "",
      s.reward.length ? `reward ${s.reward[k].toFixed(2)}` : ""].filter(Boolean).map((t) => h("div", {}, t)));
    if (charts) drawEpisodeCharts(s.t[i]);
  }
  function drawEpisodeCharts(tNow) {
    const d = R.dashEls, e = R.snap, s = e.series;
    const ta = s.t.slice(1);
    const defs = [
      { title: "Sideslip (deg)", ser: [{ x: s.t, y: s.beta_deg, color: PALETTE[0], name: "sideslip" }],
        hl: e.task === "hold" ? [{ y: e.target_beta_deg, color: PALETTE[3] }, { y: -e.target_beta_deg, color: PALETTE[3] }] : [] },
      { title: "Speed (m/s)", ser: [{ x: s.t, y: s.speed, color: PALETTE[0], name: "speed" }], hl: [{ y: e.target_speed, color: PALETTE[3] }] },
      { title: "Commands", ser: [{ x: ta, y: s.steer, color: PALETTE[0], name: "steer" }, { x: ta, y: s.throttle, color: PALETTE[1], name: "throttle" }], hl: [0] },
      { title: "Reward per step and its terms", ser: [{ x: ta, y: s.reward, color: PALETTE[2], name: "reward", width: 2 },
        ...(e.task === "hold" ? [["r_beta", "sideslip term"], ["r_speed", "speed term"]] : [["r_track", "line term"], ["r_speed", "speed term"], ["r_drift", "drift term"]])
          .map(([k, n], j) => ({ x: ta, y: s[k], color: PALETTE[[0, 1, 4][j]], name: n, dash: [3, 3] }))], hl: [0] },
      e.task === "track"
        ? { title: "Distance from the line (m, + = left)", ser: [{ x: ta, y: s.track_error, color: PALETTE[0], name: "track error" }], hl: [0] }
        : { title: "Yaw rate (deg/s)", ser: [{ x: s.t, y: s.yaw_rate_deg_s, color: PALETTE[0], name: "yaw rate" }], hl: [0] },
    ];
    defs.forEach((c, i) => {
      d.epTitles[i].textContent = c.title;
      d.epLegends[i].replaceChildren(...(c.ser.length > 1 ? c.ser.map((x) => h("span", {}, h("i", { style: `background:${x.color}` }), x.name)) : []));
      drawChart(d.epCharts[i], c.ser, { hlines: c.hl, xLabel: "t (s)", vline: tNow });
    });
  }

  // ------------------------------------------------------------------ top-down scene
  function slipColor(b) {           // |sideslip| deg -> blue (grip) .. orange (drift) .. red (spin)
    const a = Math.abs(b);
    if (a < 8) return "#4e79a7";
    if (a < 15) return "#76b7b2";
    if (a < 60) return "#f28e2b";
    return "#e15759";
  }
  function drawScene(cv, e, i, { thumb = false } = {}) {
    const dpr = window.devicePixelRatio || 1;
    const W = thumb ? cv.width / 2 : cv.clientWidth, H = thumb ? cv.height / 2 : cv.clientHeight;
    if (!W || !H) return;
    if (!thumb) { cv.width = W * dpr; cv.height = H * dpr; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(thumb ? 2 : dpr, 0, 0, thumb ? 2 : dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const bg = css.getPropertyValue("--panel-2").trim(), grid = css.getPropertyValue("--border").trim(),
      text = css.getPropertyValue("--muted").trim(), accent = css.getPropertyValue("--accent").trim(), fg = css.getPropertyValue("--text").trim();
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    const s = e.series;
    let x0 = Math.min(...s.x), x1 = Math.max(...s.x), y0 = Math.min(...s.y), y1 = Math.max(...s.y);
    if (e.track) {
      const [cx, cy] = e.track.center, rr = e.track.radius;
      x0 = Math.min(x0, cx - rr); x1 = Math.max(x1, cx + rr); y0 = Math.min(y0, cy - rr); y1 = Math.max(y1, cy + rr);
    }
    const pad = thumb ? 0.3 : 0.6, minSpan = thumb ? 2.5 : 4;       // a parked car must not fill the view
    x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
    if (x1 - x0 < minSpan) { const c = (x0 + x1) / 2; x0 = c - minSpan / 2; x1 = c + minSpan / 2; }
    if (y1 - y0 < minSpan) { const c = (y0 + y1) / 2; y0 = c - minSpan / 2; y1 = c + minSpan / 2; }
    const sc = Math.min(W / (x1 - x0), H / (y1 - y0));
    const ox = (W - sc * (x1 - x0)) / 2, oy = (H - sc * (y1 - y0)) / 2;
    const X = (x) => ox + (x - x0) * sc, Y = (y) => H - oy - (y - y0) * sc;
    if (!thumb) {
      ctx.strokeStyle = grid; ctx.lineWidth = 1;
      const step = (x1 - x0) > 12 ? 2 : (x1 - x0) > 5 ? 1 : 0.5;
      for (let g = Math.ceil(x0 / step) * step; g <= x1; g += step) { ctx.beginPath(); ctx.moveTo(X(g), 0); ctx.lineTo(X(g), H); ctx.stroke(); }
      for (let g = Math.ceil(y0 / step) * step; g <= y1; g += step) { ctx.beginPath(); ctx.moveTo(0, Y(g)); ctx.lineTo(W, Y(g)); ctx.stroke(); }
      ctx.fillStyle = text; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
      ctx.fillText(`grid ${step} m`, W - 6, H - 4);
    }
    if (e.track) {
      ctx.strokeStyle = accent; ctx.lineWidth = thumb ? 1 : 1.5; ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.arc(X(e.track.center[0]), Y(e.track.center[1]), e.track.radius * sc, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.lineWidth = thumb ? 1.5 : 2.5; ctx.lineCap = "round";
    if (!thumb) {                               // the whole episode, faint
      ctx.globalAlpha = 0.18; ctx.strokeStyle = fg; ctx.beginPath();
      s.x.forEach((x, k) => (k ? ctx.lineTo(X(x), Y(s.y[k])) : ctx.moveTo(X(x), Y(s.y[k])))); ctx.stroke(); ctx.globalAlpha = 1;
    }
    for (let k = 1; k <= i; k++) {              // the trail so far, coloured by sideslip
      ctx.strokeStyle = slipColor(s.beta_deg[k]);
      ctx.beginPath(); ctx.moveTo(X(s.x[k - 1]), Y(s.y[k - 1])); ctx.lineTo(X(s.x[k]), Y(s.y[k])); ctx.stroke();
    }
    drawCar(ctx, e, i, X, Y, sc, thumb, fg);
  }
  function drawCar(ctx, e, i, X, Y, sc, thumb, fg) {
    const s = e.series, car = e.car;
    const cx = X(s.x[i]), cy = Y(s.y[i]), yaw = (s.yaw_deg[i] * Math.PI) / 180, delta = (s.delta_deg[i] * Math.PI) / 180;
    const rect = (lx, ly, l, w, ang, fill, stroke) => {      // centre (lx, ly) in the car frame, metres
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(-yaw); ctx.translate(lx * sc, -ly * sc); ctx.rotate(-ang);
      ctx.fillStyle = fill; ctx.fillRect((-l / 2) * sc, (-w / 2) * sc, l * sc, w * sc);
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect((-l / 2) * sc, (-w / 2) * sc, l * sc, w * sc); }
      ctx.restore();
    };
    const L = car.wheelbase / 2, T = car.track_width / 2, wl = 2 * car.wheel_radius, ww = 0.03;
    if (!thumb) for (const [lx, ly, ang] of [[L, T, delta], [L, -T, delta], [-L, T, 0], [-L, -T, 0]]) rect(lx, ly, wl, ww, ang, fg);
    rect(0, 0, car.length, car.width, 0, "rgba(242,142,43,0.85)", fg);
    rect(car.length * 0.3, 0, car.length * 0.12, car.width * 0.8, 0, "rgba(255,255,255,0.7)");   // windscreen = front
    if (!thumb && s.speed[i] > 0.05) {          // velocity arrow along the course (heading + sideslip)
      const course = yaw + (s.beta_deg[i] * Math.PI) / 180, len = Math.min(1.2, 0.35 * s.speed[i]) * sc;
      ctx.strokeStyle = "#59a14f"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + len * Math.cos(course), cy - len * Math.sin(course)); ctx.stroke();
    }
  }

  // ------------------------------------------------------------------ polling
  async function pollRuns(force = false) {
    if (!R.available) return;
    try {
      const r = await api("GET", "/api/rl/runs");
      R.root = r.root;
      const sig = JSON.stringify(r.runs);
      if (sig === R.runsSig && !force) return;
      R.runsSig = sig; R.runs = r.runs;
      updateRunsTable(); updateSum(); D.renderSidebar();
    } catch (_) { /* offline: app.js shows it */ }
  }
  async function pollView() {
    if (!R.view || S.section !== "rlruns" || R.polling) return;
    R.polling = true;
    const id = R.view;
    try {
      const d = await api("GET", `/api/rl/runs/${encodeURIComponent(id)}?since=${R.hist.length}`);
      if (id !== R.view) return;
      const grew = d.history.length > 0, first = !R.detail;
      R.hist.push(...d.history);
      const before = JSON.stringify(R.detail && { ...R.detail, history: null });
      R.detail = d;
      if (first || grew || before !== JSON.stringify({ ...d, history: null })) updateDash();
    } catch (e) {
      if (R.dashEls && R.dashEls.meta) R.dashEls.meta.textContent = e.message;
    } finally { R.polling = false; }
  }

  // ------------------------------------------------------------------ registration
  async function init() {
    try {
      const c = await api("GET", "/api/rl/catalog");
      R.available = c.available; R.reason = c.reason; R.devices = c.devices; R.root = c.runs_root;
      if (c.available) {
        R.cat = c.catalog;
        R.cfg = mergeCfg(store.get(KEY));
        validate();
      }
    } catch (e) { R.available = false; R.reason = `Could not load the RL settings: ${e.message}`; }
    if (["rl", "rlruns"].includes(S.section)) D.render();
    D.renderSidebar();
    pollRuns(true);
    setInterval(() => pollRuns(), 2000);
    setInterval(pollView, 1500);
    window.addEventListener("resize", debounce(() => { if (S.section === "rlruns") { drawCurves(); drawFrame(); } }, 150));
  }
  const json = {
    name: () => (R.cfg ? `${R.cfg.name || "drift"}_rl` : "rl"),
    get: () => (R.cfg ? body() : {}),
    set: (v, label) => { R.cfg = mergeCfg(v); save(); validate(); toast(`Loaded RL settings from ${label}`, "ok"); D.render(); },
    reset: () => { R.cfg = defaults(); save(); validate(); D.render(); },
  };
  const common = { summary: renderSum, json, init: null };
  D.register("rl", { ...common, init, render: renderSetup, count: changedCount, hasError: () => R.available && R.val.errors.length > 0 });
  D.register("rlruns", { ...common, render: () => { setTimeout(pollView, 0); return renderRunsPage(); }, count: () => R.runs.filter(isActive).length });
})();
