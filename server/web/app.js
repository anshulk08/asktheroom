/* Ask the Room dashboard: live world graph, event timeline, status, ask box.
   Everything is served by the Jetson; no network access beyond it. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const C = {
    ink: "#e4ede6", ink2: "#a7bdb2", ink3: "#7f968b", dark: "#10231f",
    mat2: "#1b3a33", mat3: "#23473f",
    laser: "#ff5b45", visible: "#bfe6c9", held: "#f2b84b", hidden: "#8ec5ff", gone: "#7f918a",
  };
  const FONT = '"Atkinson Next", system-ui, sans-serif';
  const STATUS_COLOR = {
    VISIBLE: C.visible, HELD: C.held, INSIDE: C.hidden, UNDER: C.hidden, GONE: C.gone, UNKNOWN: C.gone,
  };

  const nice = (n) => (n || "").replace(/^hand:(\d+)$/, "hand $1").replace(/_/g, " ");
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  // server clock may differ from the laptop's (the Jetson may have no NTP on a hotspot)
  let clockOffset = 0; // server - local, seconds
  const serverNow = () => Date.now() / 1000 + clockOffset;

  function ago(wall) {
    const s = Math.max(0, serverNow() - wall);
    if (s < 5) return "just now";
    if (s < 60) return Math.floor(s) + " s ago";
    if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h ago";
    return Math.floor(s / 86400) + " d ago";
  }
  function clock(wall) {
    return new Date(wall * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  // ------------------------------------------------------------------ graph

  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();
  let network = null;
  let lastNodeProps = new Map(); // id -> JSON of props last written (skip no-op updates)
  let selectedObj = null;

  function lerp(a, b, t) { return Math.round(a + (b - a) * t); }
  function hex(c) { return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)); }
  const LO = hex("#3a544c"), HI = hex("#d4e6da");
  function confFill(conf) {
    const t = Math.min(1, Math.max(0, ((conf == null ? 1 : conf) - 0.3) / 0.7));
    return "rgb(" + [0, 1, 2].map((i) => lerp(LO[i], HI[i], t)).join(",") + ")";
  }

  function whereText(e) {
    switch (e.status) {
      case "VISIBLE": return "on the table";
      case "HELD": return "being held";
      case "INSIDE": return "inside " + nice(e.parent);
      case "UNDER": return "under " + nice(e.parent);
      case "GONE": return e.edge ? "gone, off the " + e.edge : "gone";
      default: return "not sure";
    }
  }

  function entityNode(e, laserTarget) {
    const conf = e.confidence == null ? 1 : e.confidence;
    const fill = confFill(conf);
    const text = conf >= 0.6 ? C.dark : C.ink;
    const border = STATUS_COLOR[e.status] || C.gone;
    const dashed = e.status === "GONE" || e.status === "UNKNOWN";
    const isLaser = laserTarget === e.name;
    return {
      id: e.name,
      label: "<b>" + nice(e.name) + "</b>\n" + whereText(e),
      title: nice(e.name) + ": " + whereText(e) + " (" + Math.round(conf * 100) + "% sure)",
      shape: "box",
      borderWidth: e.kind === "target" ? 2 : 3,
      shapeProperties: { borderRadius: e.kind === "target" ? 4 : 1, borderDashes: dashed ? [5, 4] : false },
      color: {
        background: fill, border: border,
        highlight: { background: fill, border: C.ink },
        hover: { background: fill, border: C.ink },
      },
      font: { color: text, bold: { color: text } },
      shadow: isLaser
        ? { enabled: true, color: "rgba(255,91,69,0.9)", size: 22, x: 0, y: 0 }
        : { enabled: false },
    };
  }

  function auxNode(id) {
    if (id === "table") {
      return {
        id, label: "<b>table</b>", shape: "box",
        widthConstraint: { minimum: 150 }, borderWidth: 1,
        margin: { top: 10, bottom: 10, left: 14, right: 14 },
        color: { background: C.mat3, border: C.ink3, highlight: { background: C.mat3, border: C.ink } },
        font: { color: C.ink, bold: { color: C.ink } },
        title: "The tabletop",
      };
    }
    if (id.indexOf("hand") === 0) {
      return {
        id, label: nice(id), shape: "ellipse", borderWidth: 2,
        color: { background: "#3d3420", border: C.held, highlight: { background: "#3d3420", border: C.ink } },
        font: { color: C.held, size: 13 },
        title: "A tracked hand",
      };
    }
    const off = id === "__off";
    return {
      id, label: off ? "off the table" : "lost track", shape: "box", borderWidth: 1,
      shapeProperties: { borderDashes: [4, 4], borderRadius: 2 },
      color: { background: "rgba(22,48,42,0.9)", border: C.gone, highlight: { background: C.mat2, border: C.ink } },
      font: { color: C.ink2, size: 13 },
      title: off ? "Carried out of the camera's view" : "Not seen for a while and no clear explanation",
    };
  }

  const EDGE_STYLE = {
    INSIDE: { label: "inside", color: C.hidden },
    UNDER: { label: "under", color: C.hidden },
    HELD: { label: "held by", color: C.held },
    ON: { label: "", color: "rgba(167,189,178,0.4)" },
    GONE: { label: "left", color: C.gone, dashes: [5, 5] },
    LOST: { label: "", color: C.gone, dashes: [2, 5] },
  };

  function initGraph() {
    network = new vis.Network($("graph"), { nodes, edges }, {
      autoResize: true,
      physics: { enabled: false },
      layout: { hierarchical: { enabled: false } },
      nodes: {
        font: { face: FONT, size: 15, multi: "html", color: C.dark, bold: { face: FONT, size: 17, mod: "bold" } },
        margin: { top: 7, bottom: 7, left: 10, right: 10 },
        widthConstraint: { maximum: 150 },
      },
      edges: {
        width: 1.6,
        arrows: { to: { enabled: true, scaleFactor: 0.55 } },
        font: { face: FONT, size: 12, color: C.ink2, strokeWidth: 0, align: "middle", background: "rgba(22,48,42,0.92)" },
        smooth: { enabled: true, type: "cubicBezier", forceDirection: "vertical", roundness: 0.45 },
        selectionWidth: 0,
        hoverWidth: 0,
      },
      interaction: { hover: true, tooltipDelay: 150, zoomView: false, dragView: false, dragNodes: false, selectConnectedEdges: false },
    });
    network.on("click", (p) => {
      const id = p.nodes && p.nodes[0];
      setFilter(id && !id.startsWith("__") && id !== "table" && !id.startsWith("hand") ? id : null);
    });
    let rt = null;
    const onResize = () => {
      clearTimeout(rt);
      rt = setTimeout(() => {
        const o = pickOrientation();
        if (o !== orientation) { orientation = o; applyLayout(false); } else refit(false);
      }, 120);
    };
    if (window.ResizeObserver) new ResizeObserver(onResize).observe($("graph"));
    else window.addEventListener("resize", onResize);
    orientation = pickOrientation();
  }

  let firstFit = true;
  function refit(animate) {
    if (!network || !nodes.length) return;
    const anim = animate !== false && !firstFit && !reduceMotion;
    network.fit({ animation: anim ? { duration: 450, easingFunction: "easeInOutQuad" } : false, maxZoomLevel: 1.25 });
    firstFit = false;
  }

  // Deterministic tree layout. Every item hangs off what holds it: table, a hand, "off the table",
  // or "lost track". Wide panels grow the tree upward from its roots; tall panels grow it leftward,
  // so it reads "keys -> box -> table". Positions only change when a relationship changes.
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let orientation = "up";
  let rels = [];
  let order = [];
  let tween = null;

  function pickOrientation() {
    const el = $("graph");
    return el.clientHeight > el.clientWidth * 1.1 ? "left" : "up";
  }

  function computeLayout() {
    const parent = new Map(rels.map((r) => [r.from, r.to]));
    const kids = new Map();
    const rank = (id) => { const i = order.indexOf(id); return i < 0 ? 999 : i; };
    for (const id of nodes.getIds()) {
      const par = parent.get(id);
      if (par != null && nodes.get(par)) {
        if (!kids.has(par)) kids.set(par, []);
        kids.get(par).push(id);
      }
    }
    for (const list of kids.values()) list.sort((a, b) => rank(a) - rank(b));
    const rootRank = (id) => id === "__off" ? 0 : id === "table" ? 1 : id.startsWith("hand") ? 2 : id === "__lost" ? 3 : 4;
    const roots = nodes.getIds().filter((id) => !(parent.has(id) && nodes.get(parent.get(id))))
      .sort((a, b) => rootRank(a) - rootRank(b) || String(a).localeCompare(String(b)));
    const span = new Map();
    const spanOf = (id, seen) => {
      if (seen.has(id)) return 1;
      seen.add(id);
      const s = Math.max(1, (kids.get(id) || []).reduce((t, k) => t + spanOf(k, seen), 0));
      span.set(id, s);
      return s;
    };
    roots.forEach((r) => spanOf(r, new Set()));
    const up = orientation === "up";
    const BREADTH = up ? 132 : 56, DEPTH = up ? 92 : 168, GAP = 0.35;
    const pos = {};
    const place = (id, start, depth, seen) => {
      if (seen.has(id)) return;
      seen.add(id);
      const s = span.get(id) || 1;
      const b = (start + s / 2) * BREADTH;
      pos[id] = up ? { x: b, y: -depth * DEPTH } : { x: -depth * DEPTH, y: b };
      let c = start;
      for (const k of kids.get(id) || []) { place(k, c, depth + 1, seen); c += span.get(k) || 1; }
    };
    let cursor = 0;
    const seen = new Set();
    for (const r of roots) { place(r, cursor, 0, seen); cursor += (span.get(r) || 1) + GAP; }
    return pos;
  }

  function applyLayout(animate) {
    if (!network) return;
    network.setOptions({ edges: { smooth: { forceDirection: orientation === "up" ? "vertical" : "horizontal" } } });
    const target = computeLayout();
    const ids = Object.keys(target);
    const from = network.getPositions(ids);
    if (tween) cancelAnimationFrame(tween);
    const moved = ids.some((id) => !from[id] || Math.abs(from[id].x - target[id].x) + Math.abs(from[id].y - target[id].y) > 1);
    if (!moved) { refit(animate); return; }
    if (animate === false || reduceMotion || firstFit) {
      ids.forEach((id) => network.moveNode(id, target[id].x, target[id].y));
      refit(false);
      return;
    }
    const t0 = performance.now(), D = 500;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / D);
      const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      for (const id of ids) {
        const a = from[id] || target[id], b = target[id];
        network.moveNode(id, a.x + (b.x - a.x) * e, a.y + (b.y - a.y) * e);
      }
      if (k < 1) tween = requestAnimationFrame(step);
      else { tween = null; refit(true); }
    };
    tween = requestAnimationFrame(step);
  }

  function updateGraph(state) {
    const ents = state.entities || [];
    order = ents.map((e) => e.name);
    const laserTarget = state.laser && state.laser.on ? state.laser.target : null;
    const want = new Map();
    for (const e of ents) want.set(e.name, entityNode(e, laserTarget));
    want.set("table", auxNode("table"));

    const nextRels = (state.edges || []).map((r) => ({ from: r[0], rel: r[1], to: r[2] }));
    for (const e of ents) {
      if (e.status === "GONE") nextRels.push({ from: e.name, rel: "GONE", to: "__off", edge: e.edge });
      else if (e.status === "UNKNOWN") nextRels.push({ from: e.name, rel: "LOST", to: "__lost" });
    }
    for (const r of nextRels) {
      for (const id of [r.from, r.to]) if (!want.has(id)) want.set(id, auxNode(id));
    }
    rels = nextRels;

    let structural = false;
    const add = [], upd = [];
    for (const [id, n] of want) {
      const key = JSON.stringify(n);
      if (!nodes.get(id)) {
        add.push(Object.assign({ x: 0, y: 0 }, n));
        structural = true;
      } else if (lastNodeProps.get(id) !== key) {
        upd.push(n);
      }
      lastNodeProps.set(id, key);
    }
    const gone = nodes.getIds().filter((id) => !want.has(id));
    if (gone.length) { nodes.remove(gone); gone.forEach((id) => lastNodeProps.delete(id)); structural = true; }
    if (add.length) {
      // new nodes start where their neighbour is, then slide into place
      for (const n of add) {
        const r = rels.find((q) => q.from === n.id || q.to === n.id);
        const other = r ? (r.from === n.id ? r.to : r.from) : null;
        const p = other && nodes.get(other) ? network.getPositions([other])[other] : null;
        if (p) { n.x = p.x; n.y = p.y; }
      }
      nodes.add(add);
    }
    if (upd.length) nodes.update(upd);

    const wantE = new Map();
    for (const r of rels) {
      const st = EDGE_STYLE[r.rel] || { label: r.rel.toLowerCase(), color: C.ink3 };
      const id = r.from + "|" + r.rel + "|" + r.to;
      wantE.set(id, {
        id, from: r.from, to: r.to,
        label: r.rel === "GONE" && r.edge ? "via " + r.edge : st.label,
        color: { color: st.color, highlight: C.ink, hover: C.ink },
        dashes: st.dashes || false,
      });
    }
    const goneE = edges.getIds().filter((id) => !wantE.has(id));
    const addE = [...wantE.keys()].filter((id) => !edges.get(id)).map((id) => wantE.get(id));
    if (goneE.length) edges.remove(goneE);
    if (addE.length) edges.add(addE);
    if (goneE.length || addE.length) structural = true;

    if (structural) applyLayout(true);
  }

  // ------------------------------------------------------------------ timeline

  const events = new Map(); // id -> event
  const rendered = new Set();
  let lastWall = 0;
  let haveRenderedOnce = false;

  const KIND = {
    PUT_INSIDE: "hidden", COVERED: "hidden",
    PICKED_UP: "held",
    EXITED_VIEW: "gone", LOST_TRACK: "gone",
    PUT_BACK: "table", MOVED: "table", UNCOVERED: "table", TAKEN_OUT: "table", CORRECTED: "table", FOUND: "table",
  };

  function describe(ev) {
    const o = nice(ev.obj), p = nice(ev.parent);
    switch (ev.type) {
      case "PICKED_UP": return cap(o) + " picked up";
      case "PUT_BACK": return cap(o) + " put back down";
      case "MOVED": return cap(o) + " moved";
      case "COVERED": return cap(o) + (p ? " covered by the " + p : " covered");
      case "UNCOVERED": return cap(o) + " uncovered";
      case "PUT_INSIDE": return cap(o) + (p ? " put inside the " + p : " put inside something");
      case "TAKEN_OUT": return cap(o) + (p && !ev.parent.startsWith("hand") ? " taken out of the " + p : " taken out");
      case "EXITED_VIEW": return cap(o) + (ev.edge ? " carried off the " + ev.edge + " side" : " carried out of view");
      case "LOST_TRACK": return "Lost track of the " + o;
      case "CORRECTED": return cap(o) + " location corrected";
      case "FOUND": return cap(o) + " found";
      default: return cap(o) + " " + String(ev.type || "").toLowerCase().replace(/_/g, " ");
    }
  }

  function addEvents(list) {
    let changed = false;
    for (const ev of list || []) {
      if (!ev || !ev.id || events.has(ev.id)) continue;
      events.set(ev.id, ev);
      if (ev.wall > lastWall) lastWall = ev.wall;
      changed = true;
    }
    if (events.size > 400) {
      const keep = [...events.values()].sort((a, b) => b.wall - a.wall).slice(0, 300);
      events.clear();
      keep.forEach((e) => events.set(e.id, e));
    }
    if (changed) renderEvents();
  }

  function eventItem(ev, fresh) {
    const li = document.createElement("li");
    li.className = "ev" + (fresh ? " fresh" : "");
    li.dataset.kind = KIND[ev.type] || "table";
    const what = describe(ev);

    const th = document.createElement("div");
    th.className = "thumb";
    if (ev.snapshot_url) {
      const a = document.createElement("a");
      a.href = ev.snapshot_url; a.target = "_blank"; a.rel = "noopener";
      const img = document.createElement("img");
      img.src = ev.snapshot_url; img.loading = "lazy"; img.decoding = "async";
      img.alt = "Camera snapshot: " + what;
      a.appendChild(img); th.appendChild(a);
    } else {
      const g = document.createElement("span");
      g.className = "glyph"; g.textContent = nice(ev.obj);
      th.appendChild(g);
    }

    const body = document.createElement("div");
    const p1 = document.createElement("p");
    p1.className = "ev-what"; p1.textContent = what;
    const p2 = document.createElement("p");
    p2.className = "ev-when";
    const rel = document.createElement("span");
    rel.className = "ago"; rel.dataset.wall = ev.wall; rel.textContent = ago(ev.wall);
    const t = document.createElement("time");
    t.dateTime = new Date(ev.wall * 1000).toISOString(); t.textContent = clock(ev.wall);
    p2.append(rel, ", ", t);
    if (ev.confidence != null && ev.confidence < 0.7) {
      const u = document.createElement("span");
      u.className = "ev-unsure"; u.textContent = ", not certain";
      p2.appendChild(u);
    }
    body.append(p1, p2);
    li.append(th, body);
    return li;
  }

  function renderEvents() {
    const ol = $("events");
    const list = [...events.values()]
      .filter((e) => !selectedObj || e.obj === selectedObj)
      .sort((a, b) => b.wall - a.wall)
      .slice(0, 150);
    const frag = document.createDocumentFragment();
    for (const ev of list) {
      const fresh = haveRenderedOnce && !rendered.has(ev.id);
      frag.appendChild(eventItem(ev, fresh));
    }
    for (const ev of events.values()) rendered.add(ev.id);
    haveRenderedOnce = true;
    ol.replaceChildren(frag);
    const empty = $("events-empty");
    empty.hidden = list.length > 0;
    empty.textContent = selectedObj
      ? "Nothing has happened to the " + nice(selectedObj) + " yet."
      : "Nothing has moved yet. Pick something up and it will show here.";
  }

  function refreshAgo() {
    document.querySelectorAll(".ago").forEach((el) => { el.textContent = ago(parseFloat(el.dataset.wall)); });
  }

  function setFilter(obj) {
    selectedObj = obj || null;
    const b = $("filter-clear");
    if (selectedObj) {
      b.hidden = false;
      b.textContent = "Showing " + nice(selectedObj) + " only. Show all";
    } else {
      b.hidden = true;
      if (network) network.unselectAll();
    }
    haveRenderedOnce = false; // no flash when switching views
    renderEvents();
  }
  $("filter-clear").addEventListener("click", () => setFilter(null));

  // ------------------------------------------------------------------ status bar

  let lastMsgAt = 0;
  function setText(li, v) { li.querySelector(".st-v").textContent = v; }

  function updateStatus(state, meta) {
    const fps = state.fps;
    setText($("st-fps"), fps ? Number(fps).toFixed(1) + " fps" : "no frames");
    setText($("st-net"), state.online ? "online" : "offline, local only");
    const la = meta && meta.last_answer;
    if (la && la.latency_ms != null) setText($("st-lat"), la.latency_ms + " ms");
    const L = state.laser || {};
    const lz = $("st-laser");
    lz.dataset.on = L.on ? "true" : "false";
    setText(lz, L.on ? (L.target ? "Laser on " + nice(L.target) : "Laser on") : "Laser off");
  }

  function setLink(state) {
    const el = $("st-link");
    el.dataset.state = state;
    el.querySelector(".st-text").textContent =
      state === "live" ? "Live" : state === "lost" ? "Reconnecting" : "Connecting";
  }
  setInterval(() => {
    if (lastMsgAt && Date.now() - lastMsgAt > 3000) setLink("lost");
    refreshAgo();
  }, 1000);

  // ------------------------------------------------------------------ answers

  let shownAnswerT = 0;
  const askHistory = [];
  function pushHistory(q, a, via) {
    if (!q) return;
    askHistory.unshift({ q, a: a.text, via });
    askHistory.length = Math.min(askHistory.length, 5);
    const ol = $("asked");
    ol.replaceChildren(...askHistory.slice(1).map((h) => {
      const li = document.createElement("li");
      const q = document.createElement("span");
      q.className = "asked-q"; q.textContent = h.q + (h.via === "sms" ? " (text message)" : "");
      const a = document.createElement("span");
      a.className = "asked-a"; a.textContent = h.a;
      li.append(q, a);
      return li;
    }));
    $("asked-wrap").hidden = askHistory.length < 2;
  }

  function showAnswer(a, how) {
    const fig = $("answer");
    fig.classList.remove("idle", "pending", "error");
    $("answer-text").textContent = a.text;
    const meta = $("answer-meta");
    meta.replaceChildren();
    const parts = [];
    if (how) parts.push(how);
    if (a.latency_ms != null) parts.push("Answered in " + a.latency_ms + " ms.");
    meta.append(parts.join(" "));
    if (a.point_at) {
      const s = document.createElement("span");
      s.className = "laser-note";
      s.textContent = " Laser pointing at the " + nice(a.point_at) + ".";
      meta.appendChild(s);
    } else if (a.action && a.action.indexOf("sweep:") === 0) {
      const s = document.createElement("span");
      s.className = "laser-note";
      s.textContent = " Laser sweeping toward the " + a.action.slice(6) + " edge.";
      meta.appendChild(s);
    }
  }

  function maybeShowRemoteAnswer(meta) {
    const la = meta && meta.last_answer;
    if (!la || !la.t || la.t <= shownAnswerT) return;
    shownAnswerT = la.t;
    if (la.source === "sms") showAnswer(la, "By text message: “" + la.question + "”.");
  }

  async function ask(text) {
    text = (text || "").trim();
    if (!text) return;
    const btn = $("ask-btn"), fig = $("answer");
    btn.disabled = true;
    fig.classList.remove("idle", "error");
    fig.classList.add("pending");
    $("answer-text").textContent = "Checking the table…";
    $("answer-meta").textContent = "";
    try {
      const r = await fetch("/ask", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      });
      if (!r.ok) throw new Error("The server answered " + r.status + ".");
      const a = await r.json();
      shownAnswerT = serverNow();
      showAnswer(a, "");
      pushHistory(text, a, "dashboard");
      setText($("st-lat"), a.latency_ms + " ms");
    } catch (err) {
      fig.classList.remove("pending");
      fig.classList.add("error");
      $("answer-text").textContent = "No answer. Check that this laptop is still on the Jetson hotspot, then ask again.";
      $("answer-meta").textContent = String(err.message || err);
    } finally {
      btn.disabled = false;
    }
  }

  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    ask($("ask-input").value);
  });
  $("suggest").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $("ask-input").value = b.textContent;
    ask(b.textContent);
  });

  // ------------------------------------------------------------------ video

  const video = $("video");
  let videoRetry = null;
  let videoKickAt = 0;
  video.addEventListener("error", () => {
    $("video-off").hidden = false;
    clearTimeout(videoRetry);
    videoRetry = setTimeout(() => { video.src = "/video?r=" + Date.now(); }, 2000);
  });
  video.addEventListener("load", () => { $("video-off").hidden = true; });

  // ------------------------------------------------------------------ websocket

  let backoff = 500;
  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    let ws;
    try {
      ws = new WebSocket(proto + "//" + location.host + "/ws?since=" + lastWall);
    } catch (e) {
      setTimeout(connect, backoff);
      return;
    }
    ws.onopen = () => { backoff = 500; };
    ws.onmessage = (m) => {
      let msg;
      try { msg = JSON.parse(m.data); } catch (e) { return; }
      lastMsgAt = Date.now();
      setLink("live");
      if (msg.server_t) clockOffset = msg.server_t - Date.now() / 1000;
      if (msg.state) {
        try { updateGraph(msg.state); } catch (e) { console.error(e); }
        updateStatus(msg.state, msg);
      }
      addEvents(msg.events);
      maybeShowRemoteAnswer(msg);
      // an MJPEG <img> that stalled after a server restart comes back with the socket
      if (!$("video-off").hidden && Date.now() - videoKickAt > 4000) {
        videoKickAt = Date.now();
        video.src = "/video?r=" + Date.now();
      }
    };
    ws.onclose = () => {
      setLink("lost");
      setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, 5000);
    };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* ignore */ } };
  }

  // Canvas text needs the web font loaded before the first draw.
  const fontReady = document.fonts && document.fonts.load
    ? Promise.race([document.fonts.load('600 15px "Atkinson Next"'), new Promise((r) => setTimeout(r, 1500))])
    : Promise.resolve();
  fontReady.then(() => {
    initGraph();
    renderEvents();
    connect();
  });
})();
