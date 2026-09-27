/* Ask the Room demo view (WS9): the page filmed for the demo video. Everything is drawn here, in the
   browser; the rig only serves JSON and JPEGs:
     /demo/meta     the camera image to poll, its size, table cm -> image px, the drawn zones, the seat
     /full.jpg?raw=1&w=   the undrawn camera frame, polled ~3 times a second (?fps=, ?w= on this page's URL)
     /demo/boxes    each table object's last box in image px, polled with every frame
     /ws            the world state, recent answers (with WS5's "evidence"), "listening"
     /room_layout   the room map turned to the user's seat (shared with the phone, WS6)
     /demo/evidence?obj=   the newest event snapshot of an object, for answers without evidence
   Left: the room with every named or registered object boxed, its state badge, a faded ghost where a
   missing one was last seen and a short trail when one moves. Right: the room map from the seat, and the
   conversation with a proof picture per answer. The clock is the rig's time (server_t). */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const FPS = Math.min(10, Math.max(0.5, parseFloat(params.get("fps")) || 3));
  const FRAME_W = Math.min(2560, Math.max(640, parseInt(params.get("w"), 10) || 1600));
  const FONT = '"Atkinson Next", system-ui, sans-serif';
  const COL = {
    visible: "#9ff0b4", hidden: "#8ec5ff", carried: "#ffc24d", last_seen: "#b3c0ba", found: "#fff17a",
    ink: "#f2f7f3", ink2: "#bcd0c5", ink3: "#8aa196", dark: "rgba(8,18,16,0.82)", laser: "#ff5b45",
    wood: "#6e4b2e", woodEdge: "#a37447",
  };
  const FOUND_S = 8;          // "found again" lasts this long after a missing object is seen again
  const TRAIL_S = 30;         // a moved object's trail fades over this long
  const GUESS_MIN = 0.5;      // voice/answers.py: a guess below this is not said

  const S = {
    meta: null, layout: null, state: null, answers: [], lastAnswer: null, listening: false,
    boxes: {}, frame: null, frameAt: 0, clockOffset: 0, view: null,
    prev: new Map(),        // name -> last state seen ("visible", ...)
    foundAt: new Map(),     // name -> Date.now() when it was found again
    trails: new Map(),      // name -> [{x, y, t}] image px
    pins: new Map(),        // name -> {x, y} map px (animated)
    mapTrails: new Map(),   // name -> [{x, y, t}] map layout units
    evidence: new Map(),    // answer seq -> fallback evidence (GET /demo/evidence)
    objs: [],
  };

  const now = () => Date.now() / 1000;
  const rigNow = () => now() + S.clockOffset;
  const isThing = (n) => /^thing:\d+$/.test(n || "");
  const stripThe = (s) => String(s || "").replace(/^the\s+/i, "");
  const pad = (n) => String(n).padStart(2, "0");

  function clockText(wall, seconds) {
    const d = new Date(wall * 1000);
    let h = d.getHours();
    const ap = h < 12 ? "AM" : "PM";
    h = h % 12 || 12;
    return h + ":" + pad(d.getMinutes()) + (seconds ? ":" + pad(d.getSeconds()) : "") + " " + ap;
  }

  function ago(wall) {
    const s = Math.max(0, rigNow() - wall);
    if (s < 60) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    return clockText(wall, false);
  }

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(url + " " + r.status);
    return r.json();
  }

  // ------------------------------------------------------------------ names and states

  let names = new Map();     // entity -> what it is called on screen

  function learnNames(state) {
    const m = new Map();
    for (const e of state.entities || []) {
      let n = e.label || (e.aliases && e.aliases[0]) || null;
      if (!n && isThing(e.name)) {
        const g = e.guess;
        n = g && g.name && !(g.confidence < GUESS_MIN) ? g.name + "?" : null;
      }
      if (!n && !isThing(e.name)) n = e.name.replace(/_/g, " ");
      if (n) m.set(e.name, n);
    }
    names = m;
  }

  const nice = (n) => names.get(n) || stripThe(String(n || "").replace(/^hand:\d+$/, "a hand").replace(/_/g, " "));

  // One drawable object per named or registered entity: its state, box (image px), place and times.
  function buildObjects(state) {
    const out = [];
    const room = state.room || {};
    const full = S.meta && S.meta.full;
    for (const e of state.entities || []) {
      const name = names.get(e.name);
      if (!name && !e.registry) continue;
      const o = { key: e.name, name: name || nice(e.name), zone: e.zone || "table", pos: e.pos_cm || e.resolved_cm,
        seen: e.last_seen, parent: e.parent, box: null, say: null, state: null, registered: !!e.registry };
      const r = e.registry;
      if (r && r.state && r.state !== "unknown") {      // WS8's registry (permanence.mode: registry)
        o.state = r.state;
        o.name = r.display ? r.display.replace(/_/g, " ") : o.name;
        o.zone = r.place || r.zone || o.zone;
        o.say = r.say;
        o.tentative = !!r.tentative;
        o.box = full && r.box_px ? r.box_px : null;
        o.seen = r.seen_wall || o.seen;
        o.since = r.since_wall;
      } else if (o.zone !== "table") {
        const p = room[e.name];
        if (!p) continue;
        o.state = p.absent ? "last_seen" : e.status === "HELD" ? "carried" : "visible";
        o.box = full && p.box_px ? p.box_px : null;
        o.say = p.say;
        o.seen = p.seen_wall || o.seen;
      } else {
        const st = e.status;
        if (st === "VISIBLE") o.state = "visible";
        else if (st === "HELD") o.state = "carried";
        else if (st === "INSIDE" || st === "UNDER") o.state = "hidden";
        else if (e.last_seen) o.state = "last_seen";
        else continue;                                  // never seen: nothing to show
        o.box = S.boxes[e.name] || null;
        o.say = "the table";
      }
      out.push(o);
    }
    return out;
  }

  // "the couch" -> "on the couch"; "near the couch", "the left side of the room" read as they are.
  const onPlace = (say) => !say ? "" : /^the\s/i.test(say) && !/side of|floor/i.test(say) ? "on " + say : say;

  function badge(o) {
    const found = S.foundAt.get(o.key);
    const maybe = o.tentative ? " (I think)" : "";
    if (o.state === "visible" && found && now() - found < FOUND_S) return "found again" + maybe;
    if (o.state === "visible") return (o.zone !== "table" && o.say ? onPlace(o.say) : "visible") + maybe;
    if (o.state === "carried") return (o.since ? "picked up at " + clockText(o.since, false) : "carried") + maybe;
    if (o.state === "hidden") {
      if (o.registered && !o.parent) return "still there, behind someone" + maybe;
      const p = o.parent ? nice(o.parent) : null;
      return p ? (/box|bag|cup|container/i.test(p) ? "inside the " : "under the ") + stripThe(p) : "hidden";
    }
    return ("last seen " + onPlace(o.say)).trim() + (o.seen ? " · " + ago(o.seen) : "") + maybe;
  }

  function colour(o) {
    const found = S.foundAt.get(o.key);
    if (o.state === "visible" && found && now() - found < FOUND_S) return COL.found;
    return COL[o.state] || COL.last_seen;
  }

  // State changes: "found again" when a missing object is seen; trails when a visible one moves.
  function track(objs) {
    const t = now();
    for (const o of objs) {
      const before = S.prev.get(o.key);
      if (o.state === "visible" && before && before !== "visible") S.foundAt.set(o.key, Date.now() / 1000);
      S.prev.set(o.key, o.state);
      if (o.state === "visible" && o.box) {
        const c = { x: (o.box[0] + o.box[2]) / 2, y: (o.box[1] + o.box[3]) / 2, t };
        const tr = S.trails.get(o.key) || [];
        const last = tr[tr.length - 1];
        if (!last || Math.hypot(c.x - last.x, c.y - last.y) > 25) tr.push(c);
        while (tr.length > 10 || (tr.length && t - tr[0].t > TRAIL_S)) tr.shift();
        S.trails.set(o.key, tr);
      }
    }
  }

  // ------------------------------------------------------------------ camera

  const cam = $("cam");
  const cctx = cam.getContext("2d");

  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return dpr;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawCam() {
    const dpr = fit(cam);
    const W = cam.width, H = cam.height;
    cctx.fillStyle = "#000";
    cctx.fillRect(0, 0, W, H);
    const img = S.frame;
    if (!img || !S.meta) return;
    const iw = S.meta.image_size[0], ih = S.meta.image_size[1];
    const k = Math.min(W / iw, H / ih);                        // "contain": image px -> canvas px
    const ox = (W - iw * k) / 2, oy = (H - ih * k) / 2;
    cctx.drawImage(img, ox, oy, iw * k, ih * k);
    const P = (x, y) => [ox + x * k, oy + y * k];
    const t = now();
    // trails under the boxes
    for (const o of S.objs) {
      const tr = S.trails.get(o.key);
      if (!tr || tr.length < 2) continue;
      cctx.lineCap = "round";
      for (let i = 1; i < tr.length; i++) {
        const a = Math.max(0, 1 - (t - tr[i].t) / TRAIL_S);
        cctx.strokeStyle = colour(o);
        cctx.globalAlpha = 0.65 * a;
        cctx.lineWidth = 5 * dpr;
        cctx.beginPath();
        cctx.moveTo(...P(tr[i - 1].x, tr[i - 1].y));
        cctx.lineTo(...P(tr[i].x, tr[i].y));
        cctx.stroke();
      }
      cctx.globalAlpha = 1;
    }
    // boxes, ghosts and labels (visible last, so they sit on top)
    const order = { last_seen: 0, hidden: 1, carried: 2, visible: 3 };
    const objs = S.objs.filter((o) => o.box).sort((a, b) => (order[a.state] || 0) - (order[b.state] || 0));
    const placed = [];
    for (const o of objs) {
      const [x1, y1] = P(o.box[0], o.box[1]);
      const [x2, y2] = P(o.box[2], o.box[3]);
      const c = colour(o);
      const ghost = o.state !== "visible";
      const found = badge(o) === "found again";
      cctx.save();
      if (ghost) {
        cctx.globalAlpha = o.state === "last_seen" ? 0.55 : 0.8;
        cctx.setLineDash([10 * dpr, 7 * dpr]);
        cctx.fillStyle = "rgba(179,192,186,0.16)";
        cctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }
      if (found) {                                              // a soft pulse around what was found again
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 180);
        cctx.shadowColor = COL.found;
        cctx.shadowBlur = (10 + 14 * pulse) * dpr;
      }
      cctx.strokeStyle = c;
      cctx.lineWidth = (found ? 5 : 3.5) * dpr;
      cctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      cctx.restore();
      label(o, x1, y1, x2, y2, dpr, ghost, placed);
    }
  }

  function label(o, x1, y1, x2, y2, dpr, ghost, placed) {
    const W = cam.width, H = cam.height;
    const nameF = 700 + " " + Math.round(22 * dpr) + "px " + FONT;
    const badgeF = 600 + " " + Math.round(17 * dpr) + "px " + FONT;
    cctx.font = nameF;
    const nw = cctx.measureText(o.name).width;
    cctx.font = badgeF;
    const b = badge(o);
    const bw = cctx.measureText(b).width;
    const padX = 9 * dpr, lh1 = 26 * dpr, lh2 = 21 * dpr;
    const w = Math.max(nw, bw) + 2 * padX, h = lh1 + lh2 + 8 * dpr;
    let x = Math.min(Math.max(4 * dpr, x1), W - w - 4 * dpr);
    let y = y1 - h - 6 * dpr;
    if (y < 90 * dpr) y = y2 + 6 * dpr;                         // below the box near the top (clock, badge)
    for (let i = 0; i < 6; i++) {                               // step past labels already drawn
      const hit = placed.find((p) => x < p.x + p.w && p.x < x + w && y < p.y + p.h && p.y < y + h);
      if (!hit) break;
      y = hit.y + hit.h + 4 * dpr;
    }
    y = Math.min(y, H - h - 4 * dpr);
    placed.push({ x, y, w, h });
    cctx.save();
    cctx.globalAlpha = ghost && o.state === "last_seen" ? 0.85 : 1;
    cctx.fillStyle = COL.dark;
    roundRect(cctx, x, y, w, h, 6 * dpr);
    cctx.fill();
    cctx.fillStyle = colour(o);
    cctx.fillRect(x, y, 5 * dpr, h);
    cctx.textBaseline = "top";
    cctx.font = nameF;
    cctx.fillStyle = COL.ink;
    cctx.fillText(o.name, x + padX, y + 4 * dpr);
    cctx.font = badgeF;
    cctx.fillStyle = colour(o);
    cctx.fillText(b, x + padX, y + 4 * dpr + lh1);
    cctx.restore();
  }

  // Frames: one undrawn JPEG at a time, the next asked once this one arrived (never a queue of them).
  let frameTimer = null;
  function pollFrame() {
    clearTimeout(frameTimer);
    if (!S.meta) { frameTimer = setTimeout(pollFrame, 1000); return; }
    const t0 = performance.now();
    const img = new Image();
    const sep = S.meta.image.indexOf("?") >= 0 ? "&" : "?";
    const next = () => { frameTimer = setTimeout(pollFrame, Math.max(30, 1000 / FPS - (performance.now() - t0))); };
    img.onload = () => {
      S.frame = img;
      S.frameAt = Date.now();
      $("cam-off").hidden = true;
      next();
    };
    img.onerror = () => {
      if (Date.now() - S.frameAt > 3000) $("cam-off").hidden = false;
      frameTimer = setTimeout(pollFrame, 1500);
    };
    img.src = S.meta.image + sep + (S.meta.full ? "w=" + FRAME_W + "&" : "") + "_=" + Date.now();
    getJSON("/demo/boxes").then((b) => {
      // boxes come in the full image's px; the polled frame may be smaller: drawCam scales by image_size
      S.boxes = b.boxes || {};
      refresh();
    }).catch(() => {});
  }

  // ------------------------------------------------------------------ map

  const map = $("map");
  const mctx = map.getContext("2d");

  function fallbackLayout() {
    const v = S.view || (S.meta && S.meta.view);
    if (!v) return null;
    const [tw, th] = v.table;
    return { v: 0, size: [tw + 80, th + 140], table: { rect: [40, 30, tw, th], origin: [40, 30] }, zones: [],
      you: [40 + tw / 2, 30 + th + 45] };
  }

  function zonePoly(id) {
    const z = S.meta && (S.meta.zones || []).find((q) => q.name === id);
    return z ? z.poly : null;
  }

  // Where an object goes on the map, in layout units.
  function mapPoint(o, L, i) {
    const v = S.view || (S.meta && S.meta.view);
    if (o.zone === "table" || !o.zone) {
      if (!o.pos || !v) return null;
      const m = v.m;
      const x = m[0][0] * o.pos[0] + m[0][1] * o.pos[1] + m[0][2];
      const y = m[1][0] * o.pos[0] + m[1][1] * o.pos[1] + m[1][2];
      const [tx, ty, tw, th] = L.table.rect;
      return [Math.min(tx + tw - 4, Math.max(tx + 4, L.table.origin[0] + x)),
              Math.min(ty + th - 4, Math.max(ty + 4, L.table.origin[1] + y))];
    }
    const near = /^near:/.test(o.zone);                     // WS8: "near:couch", just outside the zone
    const zid = near ? o.zone.slice(5) : o.zone;
    const z = (L.zones || []).find((q) => q.id === zid);
    if (!z) return null;                                    // "room:left" and the like: no spot on the map
    if (near) return [z.rect[0] + z.rect[2] + 10 + 8 * (i % 3), z.rect[1] + z.rect[3] / 2];
    const [zx, zy, zw, zh] = z.rect;
    const poly = zonePoly(o.zone);
    if (poly && o.box) {                                    // where in the zone, roughly
      const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
      const u = ((o.box[0] + o.box[2]) / 2 - Math.min(...xs)) / Math.max(1, Math.max(...xs) - Math.min(...xs));
      const w = ((o.box[1] + o.box[3]) / 2 - Math.min(...ys)) / Math.max(1, Math.max(...ys) - Math.min(...ys));
      return [zx + zw * (0.15 + 0.7 * Math.min(1, Math.max(0, u))), zy + zh * (0.2 + 0.65 * Math.min(1, Math.max(0, w)))];
    }
    const k = (i % 5) + 1;                                  // no box: fanned out in the zone
    return [zx + zw * k / 6, zy + zh * (0.35 + 0.15 * (i % 3))];
  }

  function drawMap() {
    const dpr = fit(map);
    const W = map.width, H = map.height;
    mctx.clearRect(0, 0, W, H);
    const L = S.layout || fallbackLayout();
    if (!L) return;
    const m = 14 * dpr;
    const k = Math.min((W - 2 * m) / L.size[0], (H - 2 * m) / L.size[1]);
    const ox = (W - L.size[0] * k) / 2, oy = (H - L.size[1] * k) / 2;
    const P = (x, y) => [ox + x * k, oy + y * k];
    // zones
    for (const z of L.zones || []) {
      const [x, y] = P(z.rect[0], z.rect[1]);
      const w = z.rect[2] * k, h = z.rect[3] * k;
      mctx.save();
      roundRect(mctx, x, y, w, h, 8 * dpr);
      mctx.fillStyle = z.kind === "seat" ? "rgba(159,240,180,0.10)" : z.kind === "door" ? "rgba(0,0,0,0)" : "rgba(255,255,255,0.05)";
      mctx.fill();
      if (z.kind === "door") mctx.setLineDash([8 * dpr, 6 * dpr]);
      mctx.strokeStyle = COL.ink3;
      mctx.lineWidth = 2 * dpr;
      mctx.stroke();
      mctx.restore();
      mctx.font = 700 + " " + Math.round(16 * dpr) + "px " + FONT;
      mctx.fillStyle = COL.ink2;
      mctx.textBaseline = "top";
      mctx.fillText(stripThe(z.say).toUpperCase(), x + 8 * dpr, y + 6 * dpr);
    }
    // the table
    const [tx, ty] = P(L.table.rect[0], L.table.rect[1]);
    mctx.fillStyle = COL.wood;
    mctx.strokeStyle = COL.woodEdge;
    mctx.lineWidth = 2 * dpr;
    roundRect(mctx, tx, ty, L.table.rect[2] * k, L.table.rect[3] * k, 5 * dpr);
    mctx.fill();
    mctx.stroke();
    mctx.font = 700 + " " + Math.round(14 * dpr) + "px " + FONT;
    mctx.fillStyle = "rgba(255,235,210,0.75)";
    mctx.fillText("TABLE", tx + 6 * dpr, ty + 5 * dpr);
    // you
    if (L.you) {
      const [yx, yy] = P(L.you[0], L.you[1]);
      mctx.fillStyle = COL.laser;
      mctx.beginPath();
      mctx.moveTo(yx, yy - 11 * dpr);
      mctx.lineTo(yx - 10 * dpr, yy + 8 * dpr);
      mctx.lineTo(yx + 10 * dpr, yy + 8 * dpr);
      mctx.closePath();
      mctx.fill();
      mctx.font = 800 + " " + Math.round(15 * dpr) + "px " + FONT;
      mctx.fillStyle = COL.ink;
      mctx.textAlign = "center";
      mctx.fillText("YOU", yx, yy + 11 * dpr);
      mctx.textAlign = "start";
    }
    // pins: eased toward their place; a short fading trail when one moves
    const t = now();
    const labels = [];
    S.objs.forEach((o, i) => {
      const target = mapPoint(o, L, i);
      if (!target) return;
      let pin = S.pins.get(o.key);
      if (!pin) { pin = { x: target[0], y: target[1] }; S.pins.set(o.key, pin); }
      const moved = Math.hypot(target[0] - (pin.tx ?? target[0]), target[1] - (pin.ty ?? target[1]));
      if (moved > 8) {
        const tr = S.mapTrails.get(o.key) || [];
        tr.push({ x: pin.x, y: pin.y, t });
        while (tr.length > 6) tr.shift();
        S.mapTrails.set(o.key, tr);
      }
      pin.tx = target[0];
      pin.ty = target[1];
      pin.x += (target[0] - pin.x) * 0.12;
      pin.y += (target[1] - pin.y) * 0.12;
      const tr = (S.mapTrails.get(o.key) || []).filter((p) => t - p.t < TRAIL_S / 3);
      S.mapTrails.set(o.key, tr);
      mctx.strokeStyle = colour(o);
      mctx.lineWidth = 3 * dpr;
      for (let j = 0; j < tr.length; j++) {
        const b = j + 1 < tr.length ? tr[j + 1] : pin;
        mctx.globalAlpha = 0.5 * (1 - (t - tr[j].t) / (TRAIL_S / 3));
        mctx.beginPath();
        mctx.moveTo(...P(tr[j].x, tr[j].y));
        mctx.lineTo(...P(b.x, b.y));
        mctx.stroke();
      }
      mctx.globalAlpha = 1;
      const [px, py] = P(pin.x, pin.y);
      const r = 7 * dpr;
      mctx.beginPath();
      mctx.arc(px, py, r, 0, Math.PI * 2);
      if (o.state === "last_seen") {
        mctx.globalAlpha = 0.45;
        mctx.strokeStyle = colour(o);
        mctx.lineWidth = 2.5 * dpr;
        mctx.stroke();
      } else {
        mctx.fillStyle = colour(o);
        mctx.fill();
      }
      mctx.globalAlpha = o.state === "last_seen" ? 0.6 : 1;
      mctx.font = 600 + " " + Math.round(16 * dpr) + "px " + FONT;
      mctx.fillStyle = COL.ink;
      mctx.textBaseline = "middle";
      let ly = py;
      for (const q of labels) if (Math.abs(q.y - ly) < 17 * dpr && Math.abs(q.x - px) < 120 * dpr) ly = q.y + 18 * dpr;
      labels.push({ x: px, y: ly });
      mctx.fillText(o.name, px + r + 5 * dpr, ly);
      mctx.globalAlpha = 1;
    });
  }

  async function loadLayout() {
    try {
      S.layout = await getJSON("/room_layout");
    } catch (e) {
      S.layout = null;                                    // no room map: the table alone (fallbackLayout)
    }
  }

  // ------------------------------------------------------------------ conversation

  const list = $("answers");
  let shownKey = "";

  function answersToShow() {
    let a = S.answers && S.answers.length ? S.answers.slice() : [];
    if (!a.length && S.lastAnswer) {
      const l = S.lastAnswer;
      a = [{ seq: l.t, t: l.t, q: l.question, text: l.text, point_at: l.point_at, evidence: l.evidence }];
    }
    return a.filter((x) => x.q && x.text).slice(-3).reverse();
  }

  function evidenceOf(a) {
    if (Array.isArray(a.evidence) && a.evidence.length) return a.evidence[0];
    if (a.evidence && !Array.isArray(a.evidence) && a.evidence.snapshot_url) return a.evidence;
    const got = S.evidence.get(a.seq);
    if (got === undefined && a.point_at) {
      S.evidence.set(a.seq, null);                        // asked once
      getJSON("/demo/evidence?obj=" + encodeURIComponent(a.point_at)).then((ev) => {
        if (ev && ev.snapshot_url) {
          S.evidence.set(a.seq, { snapshot_url: ev.snapshot_url, t: ev.t, type: ev.type, obj: ev.obj });
          shownKey = "";
          renderAnswers();
        }
      }).catch(() => {});
    }
    return got || null;
  }

  const VERB = {
    APPEARED: "first seen", MOVED: "moved", PUT_BACK: "put down", PICKED_UP: "picked up", FOUND: "found",
    HIDDEN_UNDER: "covered", UNCOVERED: "uncovered", PUT_INSIDE: "put inside", TAKEN_OUT: "taken out",
    ROOM_ARRIVED: "arrived", LOST_TRACK: "lost sight of", CORRECTED: "seen again",
  };

  function caption(ev) {
    if (ev.caption) return ev.caption;
    const what = ev.obj ? nice(ev.obj) : "it";
    const v = VERB[ev.type] || (ev.type ? String(ev.type).toLowerCase().replace(/_/g, " ") : "seen");
    return what.charAt(0).toUpperCase() + what.slice(1) + ", " + v;
  }

  function card(ev) {
    const box = document.createElement("div");
    box.className = "card";
    const shot = document.createElement("div");
    shot.className = "shot";
    const img = document.createElement("img");
    img.alt = "The moment the answer is based on";
    img.src = ev.snapshot_url;
    shot.appendChild(img);
    if (ev.box && ev.size) {                              // WS5: the object, in the snapshot's own px
      const c = document.createElement("canvas");
      shot.appendChild(c);
      img.addEventListener("load", () => {
        c.width = img.clientWidth;
        c.height = img.clientHeight;
        const k = img.clientWidth / ev.size[0];
        const g = c.getContext("2d");
        g.strokeStyle = COL.found;
        g.lineWidth = 3;
        g.strokeRect(ev.box[0] * k, ev.box[1] * k, (ev.box[2] - ev.box[0]) * k, (ev.box[3] - ev.box[1]) * k);
      });
    }
    if (ev.closeup_url) {
      const inset = document.createElement("img");
      inset.className = "inset";
      inset.alt = "";
      inset.src = ev.closeup_url;
      shot.appendChild(inset);
    }
    const cap = document.createElement("div");
    cap.className = "cap";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "Evidence";
    const text = document.createElement("span");
    text.textContent = caption(ev);
    cap.append(tag, text);
    if (ev.t) {
      const when = document.createElement("span");
      when.className = "time";
      when.textContent = clockText(ev.t, true);
      cap.appendChild(when);
    }
    box.append(shot, cap);
    return box;
  }

  function renderAnswers() {
    const a = answersToShow();
    const key = JSON.stringify(a.map((x) => [x.seq, !!evidenceOf(x)]));
    if (key === shownKey) return;
    shownKey = key;
    list.textContent = "";
    $("no-answers").hidden = a.length > 0;
    a.forEach((x, i) => {
      const li = document.createElement("li");
      if (i > 0) li.className = "old";
      const q = document.createElement("div");
      q.className = "q";
      q.textContent = x.q;
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = x.t ? clockText(x.t, false) : "";
      q.appendChild(when);
      const ans = document.createElement("div");
      ans.className = "a";
      ans.textContent = x.text;
      li.append(q, ans);
      const ev = i === 0 ? evidenceOf(x) : null;
      if (ev && ev.snapshot_url) li.appendChild(card(ev));
      list.appendChild(li);
    });
  }

  // ------------------------------------------------------------------ live state

  function refresh() {
    if (!S.state) return;
    S.objs = buildObjects(S.state);
    track(S.objs);
  }

  function setListening(on) {
    S.listening = !!on;
    $("listen").hidden = !S.listening;
    $("listen-cam").hidden = !S.listening;
    document.querySelector(".cam").classList.toggle("listening", S.listening);
  }

  let backoff = 500;
  let lastFront = null;
  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    let ws;
    try {
      ws = new WebSocket(proto + "//" + location.host + "/ws?since=" + Math.floor(now() - 600));
    } catch (e) {
      setTimeout(connect, backoff);
      return;
    }
    ws.onopen = () => { backoff = 500; $("link").textContent = "live"; $("link").dataset.state = "live"; };
    ws.onmessage = (m) => {
      let msg;
      try { msg = JSON.parse(m.data); } catch (e) { return; }
      if (msg.server_t) S.clockOffset = msg.server_t - now();
      if (msg.view) S.view = msg.view;
      if (msg.view && msg.view.front !== lastFront) {       // the seat changed: the map turns with it
        lastFront = msg.view.front;
        loadLayout();
      }
      if (msg.state) {
        S.state = msg.state;
        learnNames(msg.state);
        refresh();
      }
      S.answers = msg.answers || [];
      S.lastAnswer = msg.last_answer || null;
      setListening(msg.listening);
      renderAnswers();
    };
    ws.onclose = () => {
      $("link").textContent = "reconnecting";
      $("link").dataset.state = "lost";
      setListening(false);
      setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, 5000);
    };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* ignore */ } };
  }

  async function loadMeta() {
    try {
      S.meta = await getJSON("/demo/meta");
      S.view = S.view || S.meta.view;
    } catch (e) {
      setTimeout(loadMeta, 2000);
      return;
    }
    await loadLayout();
    pollFrame();
  }

  // ------------------------------------------------------------------ clock and drawing

  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function tickClock() {
    const d = new Date(rigNow() * 1000);
    $("clock").textContent = pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
    $("date").textContent = DAYS[d.getDay()] + " " + d.getDate() + " " + MONTHS[d.getMonth()] + " " + d.getFullYear();
  }

  function frame() {
    try { drawCam(); } catch (e) { console.error(e); }
    try { drawMap(); } catch (e) { console.error(e); }
    requestAnimationFrame(frame);
  }

  setInterval(tickClock, 250);
  setInterval(loadLayout, 30000);
  setInterval(() => { if (S.state) renderAnswers(); }, 20000);   // "min ago" and fallbacks stay current
  const fontReady = document.fonts && document.fonts.load
    ? Promise.race([document.fonts.load('700 22px "Atkinson Next"'), new Promise((r) => setTimeout(r, 1500))])
    : Promise.resolve();
  fontReady.then(() => {
    tickClock();
    loadMeta();
    connect();
    requestAnimationFrame(frame);
  });
})();
