// Ask the Room live state (/live): polls /live/state once a second and draws every stage of every answer,
// the laser, the room tracks and the table. Read only: nothing on this page can move or light the laser.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const hms = (t) => (t ? new Date(t * 1000).toLocaleTimeString([], { hour12: false }) : "");
const num = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const OUTCOME = { hit: "hit", miss: "miss", refused: "refused", no_cue: "no point cue", spoken: "spoken only", pending: "pending" };
const open = new Set();          // question ids whose stage list the viewer opened

function kv(el, rows) {
  el.innerHTML = rows.map(([k, v, cls]) => `<dt>${esc(k)}</dt><dd class="${cls || ""}">${v}</dd>`).join("");
}

function laserLine(l) {
  if (!l) return "–";
  const where = l.px ? `room px (${num(l.px[0], 0)}, ${num(l.px[1], 0)})` : l.cm ? `table (${num(l.cm[0])}, ${num(l.cm[1])}) cm` : "";
  const err = l.err_px !== undefined && l.err_px !== null ? `err ${num(l.err_px)} px` : l.err_cm !== undefined && l.err_cm !== null ? `err ${num(l.err_cm)} cm` : "";
  const raw = l.first_raw_px !== undefined && l.first_raw_px !== null ? `first raw ${num(l.first_raw_px)} px` : "";
  const tries = l.tries !== undefined && l.tries !== null ? `${l.tries} tries` : "";
  return [esc(l.target || ""), where, `<b>${esc(l.reason || "")}</b>`, tries, err, raw, l.why ? esc(l.why) : ""].filter(Boolean).join(" · ");
}

function drawQuestions(qs) {
  $("no-q").hidden = qs.length > 0;
  $("questions").innerHTML = qs.map((q) => {
    const took = q.stages.length ? q.stages[q.stages.length - 1].dt : 0;
    const summary = [
      ["route", esc(q.route || "–")],
      ["answer", esc(q.answer || "–")],
      ["action", esc(q.action || "none")],
      ["point cue", q.cue ? esc(q.cue) : q.laser ? "asked to point" : "–"],
      ["laser", laserLine(q.laser)],
    ];
    if (q.timing) summary.push(["timing", `<span class="q-meta">${esc(q.timing)}</span>`]);
    const stages = q.stages.map((s) => `<li><span class="dt">+${num(s.dt, 2)}s</span><span class="st ${esc(s.stage)}">${esc(s.stage)}</span>` +
      `<span class="took">${s.took !== undefined ? num(s.took, 2) + "s" : ""}</span><span class="msg">${esc(s.msg)}</span></li>`).join("");
    return `<li class="q ${esc(q.outcome)}"><div class="q-head"><span class="chip ${esc(q.outcome)}">${esc(OUTCOME[q.outcome] || q.outcome)}</span>` +
      `<span class="q-text">${esc(q.text)}</span><span class="q-meta">${hms(q.t)} · ${esc(q.source)} · ${num(took, 1)} s</span></div>` +
      `<dl class="q-summary">${summary.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
      `<details data-id="${q.id}"${open.has(q.id) ? " open" : ""}><summary>${q.stages.length} stages</summary><ol class="stages">${stages}</ol></details></li>`;
  }).join("");
  document.querySelectorAll("#questions details").forEach((d) => d.addEventListener("toggle", () => {
    const id = Number(d.dataset.id);
    if (d.open) open.add(id); else open.delete(id);
  }));
}

function drawHistory(rows) {
  $("history").innerHTML = "<tr><th>time</th><th>question</th><th>intent</th><th>answer</th><th>ms</th></tr>" +
    (rows || []).map((r) => `<tr><td>${hms(r.t)}</td><td>${esc(r.text)}</td><td>${esc(r.intent)}${r.obj ? " · " + esc(r.obj) : ""}</td>` +
      `<td>${esc(r.answer)}</td><td>${esc(r.latency_ms)}</td></tr>`).join("");
}

function drawLaser(rig) {
  const l = (rig && rig.laser) || {};
  const lock = $("lock");
  lock.hidden = !l.locked;
  if (l.locked) lock.textContent = `LASER LOCKED: ${l.locked}`;
  if (l.error) { kv($("laser"), [["error", esc(l.error), "bad"]]); return; }
  const st = l.state || {};
  const map = l.room_map;
  kv($("laser"), [
    ["locked", l.locked ? esc(l.locked) : "no", l.locked ? "bad" : "good"],
    ["drift", `${esc(l.drift_n)} of ${esc(l.drift_aims)} misses in a row before a lock`, l.drift_n ? "warn" : ""],
    ["on", st.on ? "yes" : "no", st.on ? "warn" : ""],
    ["target", esc(st.target || "–")],
    ["room pointing", l.room_enabled ? "on" : "off"],
    ["aim cue", esc(l.aim_cue || "–")],
    ["px bias", l.px_bias ? `(${l.px_bias.map((v) => num(v)).join(", ")}) px` : "–"],
    ["room map", map ? `${map.dots} dots of ${map.points}, spacing ${num(map.spacing_px)} px; zones ${esc((map.zones || []).join(", "))}` : "none"],
  ]);
  const a = l.last_aim || {};
  kv($("last-aim"), Object.keys(a).length ? Object.entries(a).map(([k, v]) => [k, esc(Array.isArray(v) ? `[${v.join(", ")}]` : v)]) : [["–", "no aim yet"]]);
}

function drawZones(rig) {
  const room = rig && rig.room;
  const el = $("zones");
  if (!room) { el.innerHTML = `<p class="empty">Room memory is off.</p>`; return; }
  if (room.error) { el.innerHTML = `<p class="errors">${esc(room.error)}</p>`; return; }
  el.innerHTML = (room.zones || []).map((z) => {
    const ts = (room.tracks || []).filter((t) => t.zone === z.name);
    const rows = ts.map((t) => `<tr class="${t.age_s > 15 ? "stale" : ""}"><td>${esc(t.tid)}</td><td>${esc(t.name || t.cls)}</td>` +
      `<td>${t.confidence !== null && t.confidence !== undefined ? num(t.confidence, 2) : "–"}</td><td>${t.confirmed ? "yes" : "no"}</td>` +
      `<td>${esc(t.role)}${t.entity ? " → " + esc(t.entity) : ""}</td><td>${num(t.age_s, 0)} s</td></tr>`).join("");
    return `<div class="zone"><h3>${esc(z.say)} <span class="sub">${esc(z.name)} · ${ts.length} tracks</span></h3>` +
      (ts.length ? `<table class="tbl"><tr><th>track</th><th>name</th><th>conf</th><th>confirmed</th><th>role</th><th>age</th></tr>${rows}</table>` : `<p class="empty">nothing tracked</p>`) + "</div>";
  }).join("");
}

function drawThings(world) {
  const ents = ((world && world.entities) || []).slice().sort((a, b) => (b.last_seen || 0) - (a.last_seen || 0));
  $("things").innerHTML = "<tr><th>name</th><th>status</th><th>zone / parent</th><th>conf</th><th>at (cm)</th></tr>" +
    ents.map((e) => `<tr><td>${esc(e.name)}</td><td>${esc(e.status)}</td><td>${esc(e.parent || e.zone || "")}</td>` +
      `<td>${num(e.confidence, 2)}</td><td>${e.pos_cm ? e.pos_cm.map((v) => num(v)).join(", ") : "–"}</td></tr>`).join("");
}

function drawFeeds(ambient) {
  const naming = ambient.filter((r) => /looks like|no usable name|naming room track/.test(r.msg));
  const other = ambient.filter((r) => !naming.includes(r));
  const li = (r) => `<li><span class="t">${hms(r.t)}</span>${esc(r.msg)}</li>`;
  $("naming").innerHTML = naming.slice(0, 40).map(li).join("") || `<li class="empty">nothing named yet</li>`;
  $("ambient").innerHTML = other.slice(0, 40).map(li).join("") || `<li class="empty">nothing yet</li>`;
}

async function tick() {
  try {
    const r = await fetch("/live/state", { cache: "no-store" });
    const s = await r.json();
    $("link").dataset.state = "ok";
    $("link").textContent = "live";
    drawQuestions(s.questions || []);
    drawHistory(s.history);
    drawLaser(s.rig);
    drawZones(s.rig);
    drawThings(s.world);
    drawFeeds(s.ambient || []);
    const online = s.rig && s.rig.online !== undefined ? s.rig.online : s.world && s.world.online;
    $("net").textContent = online ? "online (Grok)" : online === false ? "offline" : "network ?";
    $("net").dataset.state = online ? "ok" : online === false ? "bad" : "wait";
    $("fps").textContent = s.world && s.world.fps ? `${num(s.world.fps)} fps` : "fps ?";
    const errs = Object.entries(s.errors || {});
    $("errors").hidden = !errs.length;
    $("errors").textContent = errs.map(([k, v]) => `${k}: ${v}`).join(" · ");
  } catch (e) {
    $("link").dataset.state = "bad";
    $("link").textContent = "disconnected";
  }
}

function camera() {
  const img = $("cam");
  const next = new Image();
  next.onload = () => { img.src = next.src; };
  next.onerror = () => {                       // no room memory: the table view instead
    if (!next.src.includes("/frame.jpg")) next.src = `/frame.jpg?t=${Date.now()}`;
  };
  next.src = `/full.jpg?t=${Date.now()}`;
}

setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false }); }, 500);
tick();
setInterval(tick, 1000);
camera();
setInterval(camera, 2000);
