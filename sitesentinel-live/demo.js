/* SiteSentinel live demo: the backend's store, PPE model and simulator, running in the browser,
   fed by the phone camera (COCO-SSD finds people, colour regions find hard hats and vests). */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  /* ---------- PPE association model (port of backend/ppe_model.py) ---------- */
  var EXPECTED_CENTER_Y = { hardhat: 0.08, vest: 0.40 };
  var POS_LOW = -1.0, POS_HIGH = 2.0, MAX_REL_SIZE = 3.0, RULE_MIN_OVERLAP = 0.6;
  var MODEL = null;

  function area(b) { return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]); }
  function intersection(a, b) {
    var top = Math.max(a[0], b[0]), left = Math.max(a[1], b[1]);
    var bottom = Math.min(a[2], b[2]), right = Math.min(a[3], b[3]);
    return Math.max(0, bottom - top) * Math.max(0, right - left);
  }
  function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

  function features(p, it, type) {
    var ph = Math.max(p[2] - p[0], 1e-6), pw = Math.max(p[3] - p[1], 1e-6);
    var inter = intersection(p, it), itemArea = area(it);
    var union = area(p) + itemArea - inter;
    var relCy = clamp(((it[0] + it[2]) / 2 - p[0]) / ph, POS_LOW, POS_HIGH);
    var relCx = clamp(((it[1] + it[3]) / 2 - p[1]) / pw, POS_LOW, POS_HIGH);
    return [
      itemArea > 0 ? inter / itemArea : 0,
      union > 0 ? inter / union : 0,
      relCy,
      Math.abs(relCy - EXPECTED_CENTER_Y[type]),
      Math.abs(relCx - 0.5),
      Math.min((it[3] - it[1]) / pw, MAX_REL_SIZE),
      Math.min((it[2] - it[0]) / ph, MAX_REL_SIZE)
    ];
  }
  function sigmoid(z) { return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); }
  function probability(p, it, type) {
    var x = features(p, it, type);
    if (!MODEL) return x[0] > RULE_MIN_OVERLAP ? 1 : 0;
    var m = MODEL.models[type], z = m.intercept;
    for (var i = 0; i < x.length; i++) z += m.coef[i] * (x[i] - m.mean[i]) / (m.scale[i] || 1);
    return sigmoid(z);
  }
  function wears(p, it, type) { return probability(p, it, type) >= (MODEL ? MODEL.threshold : 0.5); }
  function assessPeople(persons, hats, vests) {
    return persons.map(function (p) {
      return {
        hardhat: hats.some(function (h) { return wears(p, h, "hardhat"); }),
        vest: vests.some(function (v) { return wears(p, v, "vest"); })
      };
    });
  }

  /* ---------- Event store (port of backend/store.py) ---------- */
  var WINDOW_MS = 15 * 60 * 1000, LIVE_MS = 15 * 1000, HOUR_MS = 3600 * 1000, MAX_EVENTS = 200;
  var events = [], version = 0, seq = 0;

  function addEvent(camera, ts, persons, img) {
    var both = 0, noHat = 0, noVest = 0;
    persons.forEach(function (p) {
      if (!p.hardhat && !p.vest) both++;
      else if (!p.hardhat) noHat++;
      else if (!p.vest) noVest++;
    });
    var e = {
      id: "e" + (++seq), camera: camera, ts: ts, people: persons.length,
      protected: persons.length - both - noHat - noVest,
      no_hardhat: noHat, no_vest: noVest, no_both: both,
      status: (both || noHat || noVest) ? "violation" : "clear",
      img: img || null
    };
    events.push(e);
    if (events.length > MAX_EVENTS) events.shift();
    // Keep memory small on a phone: only the newest 40 photos.
    var withImg = events.filter(function (x) { return x.img; });
    for (var i = 0; i < withImg.length - 40; i++) withImg[i].img = null;
    version++;
    return e;
  }

  function summary(now) {
    var recent = events.filter(function (e) { return now - e.ts <= WINDOW_MS; });
    var people = 0, prot = 0, nh = 0, nv = 0, nb = 0;
    recent.forEach(function (e) { people += e.people; prot += e.protected; nh += e.no_hardhat; nv += e.no_vest; nb += e.no_both; });
    var cams = {};
    events.forEach(function (e) {
      var c = cams[e.camera] || (cams[e.camera] = { id: e.camera, last_seen: 0, checks: 0 });
      c.last_seen = Math.max(c.last_seen, e.ts); c.checks++;
    });
    var camList = Object.keys(cams).sort().map(function (k) { cams[k].live = now - cams[k].last_seen <= LIVE_MS; return cams[k]; });
    var cur = Math.floor(now / HOUR_MS), hourly = [];
    for (var h = cur - 11; h <= cur; h++) {
      var bucket = events.filter(function (e) { return Math.floor(e.ts / HOUR_MS) === h; });
      hourly.push({ hour_start: h * HOUR_MS, checks: bucket.length, violations: bucket.filter(function (e) { return e.status === "violation"; }).length });
    }
    return {
      window_minutes: WINDOW_MS / 60000, people: people, protected: prot,
      protected_rate: people ? Math.round(100 * prot / people) : null,
      no_hardhat: nh, no_vest: nv, no_both: nb, checks_total: events.length, cameras: camList, hourly: hourly
    };
  }

  function rateSince(now, ms) {
    var people = 0, prot = 0;
    events.forEach(function (e) { if (now - e.ts <= ms) { people += e.people; prot += e.protected; } });
    return people ? Math.round(100 * prot / people) : null;
  }

  /* ---------- Camera simulator (port of backend/simulator.py) ---------- */
  var SIM_CAMERAS = ["gate-1", "scaffold-2", "crane-3"], PEOPLE_WEIGHTS = [1, 3, 3, 2, 1];
  var VIOLATION_RATE = 0.3, DISTRACTOR_RATE = 0.5, simSent = 0, simTimer = null;

  function U(a, b) { return a + Math.random() * (b - a); }
  function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
  function box(top, left, h, w) {
    h = Math.min(h, 1); w = Math.min(w, 1);
    top = clamp(top, 0, 1 - h); left = clamp(left, 0, 1 - w);
    return [top, left, top + h, left + w];
  }
  function place(p, rcy, rcx, rh, rw) {
    var h = p[2] - p[0], w = p[3] - p[1], ih = rh * h, iw = rw * w;
    return box(p[0] + rcy * h - ih / 2, p[1] + rcx * w - iw / 2, ih, iw);
  }
  function beside(p, rh, rw) {
    var gap = U(0.02, 0.08);
    var rcx = Math.random() < 0.5 ? -gap - rw / 2 : 1 + gap + rw / 2;
    return place(p, U(0.1, 0.9), rcx, rh, rw);
  }
  function simScene() {
    var total = PEOPLE_WEIGHTS.reduce(function (a, b) { return a + b; }, 0), r = Math.random() * total, count = 0;
    for (; count < PEOPLE_WEIGHTS.length; count++) { r -= PEOPLE_WEIGHTS[count]; if (r < 0) break; }
    var persons = [], hats = [], vests = [];
    for (var i = 0; i < count; i++) {
      var slotLeft = i / count, slotW = 1 / count;
      var h = U(0.35, 0.75), w = Math.min(h * U(0.25, 0.35), slotW / 3);
      var p = box(U(0, 1 - h), slotLeft + U(w, slotW - 2 * w), h, w);
      var hasHat = true, hasVest = true;
      if (Math.random() < VIOLATION_RATE) {
        var missing = pick(["hardhat", "vest", "both"]);
        hasHat = missing === "vest"; hasVest = missing === "hardhat";
      }
      if (hasHat) hats.push(place(p, U(0.03, 0.11), U(0.45, 0.55), U(0.12, 0.18), U(0.35, 0.5)));
      else if (Math.random() < DISTRACTOR_RATE) {
        var rh = U(0.12, 0.18), rw = U(0.35, 0.5);
        hats.push(Math.random() < 0.5 ? place(p, U(0.45, 0.6), pick([0.1, 0.9]), rh, rw) : beside(p, rh, rw));
      }
      if (hasVest) vests.push(place(p, U(0.37, 0.45), U(0.45, 0.55), U(0.28, 0.34), U(0.75, 0.9)));
      else if (Math.random() < DISTRACTOR_RATE) {
        vests.push(Math.random() < 0.5 ? place(p, U(0.95, 1.0), U(0.45, 0.55), U(0.08, 0.12), U(1.0, 1.2))
          : beside(p, U(0.28, 0.34), U(0.75, 0.9)));
      }
      persons.push(p);
    }
    return { persons: persons, hats: hats, vests: vests };
  }
  function simTick() {
    var cam = SIM_CAMERAS[simSent % SIM_CAMERAS.length];
    var s = simScene();
    addEvent(cam, Date.now(), assessPeople(s.persons, s.hats, s.vests), null);
    simSent++;
    render();
  }
  function setSim(on) {
    if (simTimer) { clearInterval(simTimer); simTimer = null; }
    if (on) { simTick(); simTimer = setInterval(simTick, 2000); }
  }

  /* ---------- Phone camera + detector ---------- */
  var video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
  var small = document.createElement("canvas"), sctx = small.getContext("2d", { willReadFrequently: true });
  var snap = document.createElement("canvas"), snapCtx = snap.getContext("2d");
  var detector = null, detectorLoading = null, stream = null, running = false, facing = "environment", lastSent = 0;
  var SEND_INTERVAL_MS = 2000, MIN_SCORE = 0.5, loopGen = 0;

  function setStatus(text, cls) {
    var n = $("cam-status");
    n.textContent = text;
    n.className = "camstat" + (cls ? " " + cls : "");
  }

  function loadDetector() {
    if (detector) return Promise.resolve(detector);
    if (detectorLoading) return detectorLoading;
    setStatus("Loading the person detector (first time takes a few seconds)…");
    detectorLoading = cocoSsd.load({ base: "lite_mobilenet_v2" }).then(function (m) { detector = m; return m; });
    return detectorLoading;
  }

  function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("This browser can't open the camera. Use Chrome or Safari over https.", "bad");
      return;
    }
    $("cam-start").disabled = true;
    setStatus("Opening the camera…");
    stopStream();
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    }).then(function (s) {
      stream = s;
      video.srcObject = s;
      return video.play();
    }).then(function () {
      $("hint").hidden = true;
      $("cam-flip").disabled = false;
      $("cam-stop").disabled = false;
      return loadDetector();
    }).then(function () {
      if (!stream) return;
      running = true;
      setStatus("Watching. Point the camera at people.");
      var gen = ++loopGen;
      loop(gen);
    }).catch(function (err) {
      $("cam-start").disabled = false;
      var name = err && err.name;
      if (name === "NotAllowedError") setStatus("Camera permission was refused. Allow the camera for this site and try again.", "bad");
      else if (name === "NotFoundError") setStatus("No camera found on this device.", "bad");
      else setStatus("Could not start: " + (err && err.message ? err.message : err), "bad");
    });
  }

  function stopStream() {
    running = false;
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
  }

  function stopCamera() {
    stopStream();
    video.srcObject = null;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    $("hint").hidden = false;
    $("cam-start").disabled = false;
    $("cam-flip").disabled = true;
    $("cam-stop").disabled = true;
    setStatus("Camera is off.");
    render();
  }

  // HSV in degrees / 0..1.
  function hsv(r, g, b) {
    var max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min, h = 0;
    if (d) {
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60; if (h < 0) h += 360;
    }
    return [h, max ? d / max : 0, max / 255];
  }
  function isHatColour(h, s, v) {
    return (h >= 38 && h <= 66 && s >= 0.5 && v >= 0.5) ||        // yellow
      (h >= 12 && h < 38 && s >= 0.72 && v >= 0.6) ||             // orange
      ((h <= 8 || h >= 345) && s >= 0.62 && v >= 0.45);           // red
  }
  function isVestColour(h, s, v) {
    return (h >= 45 && h <= 95 && s >= 0.45 && v >= 0.55) ||      // hi-vis yellow / lime
      (h >= 12 && h < 45 && s >= 0.65 && v >= 0.6);               // hi-vis orange
  }

  // Look inside one region of a person for hard hat or vest colours; returns a normalized box or null.
  function findItem(data, sw, sh, p, type) {
    var ph = p[2] - p[0], pw = p[3] - p[1], y0, y1, x0, x1, test, minShare;
    if (type === "hardhat") {
      y0 = p[0] - 0.06 * ph; y1 = p[0] + 0.2 * ph; x0 = p[1] + 0.15 * pw; x1 = p[3] - 0.15 * pw;
      test = isHatColour; minShare = 0.1;
    } else {
      y0 = p[0] + 0.2 * ph; y1 = p[0] + 0.72 * ph; x0 = p[1] + 0.1 * pw; x1 = p[3] - 0.1 * pw;
      test = isVestColour; minShare = 0.12;
    }
    var py0 = Math.max(0, Math.floor(y0 * sh)), py1 = Math.min(sh, Math.ceil(y1 * sh));
    var px0 = Math.max(0, Math.floor(x0 * sw)), px1 = Math.min(sw, Math.ceil(x1 * sw));
    var total = 0, hits = 0, minX = sw, minY = sh, maxX = -1, maxY = -1;
    for (var y = py0; y < py1; y++) {
      for (var x = px0; x < px1; x++) {
        var i = (y * sw + x) * 4, c = hsv(data[i], data[i + 1], data[i + 2]);
        total++;
        if (test(c[0], c[1], c[2])) {
          hits++;
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
    }
    if (!total || hits < 8 || hits / total < minShare) return null;
    return [minY / sh, minX / sw, (maxY + 1) / sh, (maxX + 1) / sw];
  }

  function drawLabel(ctx, text, x, y, bg, fg, size) {
    ctx.font = "700 " + size + "px Archivo, Arial, sans-serif";
    var w = ctx.measureText(text).width + size * 0.8, h = size * 1.5;
    var ty = y - h < 0 ? y : y - h;
    ctx.fillStyle = bg; ctx.fillRect(x, ty, w, h);
    ctx.fillStyle = fg; ctx.fillText(text, x + size * 0.4, ty + size * 1.1);
  }

  function drawOverlay(ctx, W, H, people, verdicts, hats, vests) {
    ctx.clearRect(0, 0, W, H);
    var lw = Math.max(2, W / 240), fs = Math.max(12, Math.round(W / 42));
    ctx.lineWidth = lw;
    ctx.setLineDash([lw * 3, lw * 2]);
    ctx.strokeStyle = "#ffc400";
    hats.concat(vests).forEach(function (b) { ctx.strokeRect(b[1] * W, b[0] * H, (b[3] - b[1]) * W, (b[2] - b[0]) * H); });
    ctx.setLineDash([]);
    people.forEach(function (p, i) {
      var v = verdicts[i], ok = v.hardhat && v.vest, col = ok ? "#17803d" : "#c8102e";
      ctx.strokeStyle = col; ctx.lineWidth = lw * 1.6;
      ctx.strokeRect(p[1] * W, p[0] * H, (p[3] - p[1]) * W, (p[2] - p[0]) * H);
      var text = (v.hardhat ? "✓" : "✗") + " Hard hat  " + (v.vest ? "✓" : "✗") + " Vest";
      drawLabel(ctx, text, p[1] * W, p[0] * H, col, "#fff", fs);
    });
  }

  function loop(gen) {
    if (!running || gen !== loopGen) return;
    var next = function () { requestAnimationFrame(function () { loop(gen); }); };
    if (video.readyState < 2 || !video.videoWidth) { next(); return; }
    var vw = video.videoWidth, vh = video.videoHeight;
    detector.detect(video, 10, MIN_SCORE).then(function (preds) {
      if (!running || gen !== loopGen) return;
      if (overlay.width !== vw || overlay.height !== vh) { overlay.width = vw; overlay.height = vh; }
      var people = preds.filter(function (p) { return p.class === "person"; }).map(function (p) {
        var b = p.bbox;
        return [b[1] / vh, b[0] / vw, (b[1] + b[3]) / vh, (b[0] + b[2]) / vw];
      });
      var sw = 192, sh = Math.round(192 * vh / vw);
      if (small.width !== sw || small.height !== sh) { small.width = sw; small.height = sh; }
      sctx.drawImage(video, 0, 0, sw, sh);
      var data = sctx.getImageData(0, 0, sw, sh).data;
      var hats = [], vests = [];
      people.forEach(function (p) {
        var hb = findItem(data, sw, sh, p, "hardhat"); if (hb) hats.push(hb);
        var vb = findItem(data, sw, sh, p, "vest"); if (vb) vests.push(vb);
      });
      var verdicts = assessPeople(people, hats, vests);
      drawOverlay(octx, vw, vh, people, verdicts, hats, vests);

      if (!people.length) setStatus("No one on camera.");
      else {
        var bad = verdicts.filter(function (v) { return !(v.hardhat && v.vest); }).length;
        setStatus(people.length + (people.length === 1 ? " person" : " people") + " on camera · " +
          (bad ? bad + " without full PPE" : "everyone protected"), bad ? "bad" : "ok");
      }

      var now = Date.now();
      if (people.length && now - lastSent > SEND_INTERVAL_MS) {
        lastSent = now;
        var tw = 480, th = Math.round(480 * vh / vw);
        snap.width = tw; snap.height = th;
        snapCtx.drawImage(video, 0, 0, tw, th);
        snapCtx.drawImage(overlay, 0, 0, tw, th);
        addEvent("phone-cam", now, verdicts, snap.toDataURL("image/jpeg", 0.72));
        render();
      }
      next();
    }).catch(next);
  }

  /* ---------- Dashboard (from backend/static/app.js, reading the in-page store) ---------- */
  var state = { filter: "violations", seen: null, renderedVersion: -1, renderedFilter: null };
  var belowSince = null, wasFiring = false;

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function ago(ms) {
    var s = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (s < 5) return "just now";
    if (s < 60) return s + " seconds ago";
    var m = Math.round(s / 60);
    if (m < 60) return m + (m === 1 ? " minute ago" : " minutes ago");
    var h = Math.round(m / 60);
    return h + (h === 1 ? " hour ago" : " hours ago");
  }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  function describe(e) {
    if (e.people === 0) return "No one on camera";
    if (e.status === "clear") return "Everyone protected (" + plural(e.people, "person", "people") + ")";
    var parts = [];
    if (e.no_hardhat) parts.push(e.no_hardhat + " missing hard hat");
    if (e.no_vest) parts.push(e.no_vest + " missing vest");
    if (e.no_both) parts.push(e.no_both + " missing both");
    var atRisk = e.no_hardhat + e.no_vest + e.no_both;
    return atRisk + " of " + plural(e.people, "person", "people") + " unprotected: " + parts.join(", ");
  }

  function renderSummary(s) {
    $("window").textContent = s.window_minutes;
    var rate = $("rate");
    rate.className = "big";
    if (s.protected_rate === null) {
      rate.textContent = "–";
      $("rate-sub").textContent = s.checks_total ? "No one has been on camera in this window." : "No people detected yet.";
    } else {
      rate.textContent = s.protected_rate + "%";
      rate.classList.add(s.protected_rate >= 95 ? "good" : s.protected_rate >= 75 ? "warn" : "alert");
      $("rate-sub").textContent = s.protected + " of " + plural(s.people, "person", "people") + " wore a hard hat and vest.";
    }
    [["t-hat", s.no_hardhat], ["t-vest", s.no_vest], ["t-both", s.no_both]].forEach(function (p) {
      var n = $(p[0]); n.textContent = p[1]; n.classList.toggle("hot", p[1] > 0);
    });
    var live = s.cameras.filter(function (c) { return c.live; }).length, dot = $("live-dot");
    dot.className = "dot";
    if (!s.cameras.length) $("live-text").textContent = "Waiting for a camera";
    else if (live) { dot.classList.add("live"); $("live-text").textContent = plural(live, "camera", "cameras") + " live"; }
    else { dot.classList.add("stale"); $("live-text").textContent = "Cameras offline"; }

    var wrap = $("hours"); wrap.textContent = "";
    var max = Math.max.apply(null, s.hourly.map(function (h) { return h.checks; }).concat([1]));
    s.hourly.forEach(function (h, i) {
      var col = el("div", "hour");
      col.title = h.checks + " checks, " + h.violations + " with violations";
      var bar = el("div", "hour-bar");
      bar.style.height = Math.round((h.checks / max) * 100) + "%";
      if (h.violations) {
        var bad = el("div", "hour-bad");
        bad.style.height = Math.round((h.violations / h.checks) * 100) + "%";
        bar.appendChild(bad);
      }
      col.appendChild(bar);
      col.appendChild(el("div", "hour-label", i % 3 === 0 ? new Date(h.hour_start).getHours() + "h" : " "));
      wrap.appendChild(col);
    });

    var list = $("cams"); list.textContent = "";
    $("cams-empty").hidden = s.cameras.length > 0;
    s.cameras.forEach(function (c) {
      var li = el("li");
      li.appendChild(el("span", "name", c.id));
      li.appendChild(el("span", "seen", c.live ? "Live" : "Last seen " + ago(c.last_seen)));
      list.appendChild(li);
    });
  }

  function renderEvents(s) {
    var list = events.slice().reverse();
    if (state.filter === "violations") list = list.filter(function (e) { return e.status === "violation"; });
    list = list.slice(0, 40);
    var rows = $("rows"), firstLoad = state.seen === null, fresh = {}, nextSeen = {};
    list.forEach(function (e) { nextSeen[e.id] = true; if (!firstLoad && !state.seen[e.id]) fresh[e.id] = true; });
    state.seen = nextSeen;
    rows.textContent = "";

    var box = $("empty");
    box.hidden = list.length > 0;
    if (!box.hidden) {
      box.querySelector(".empty-title").textContent = s.checks_total ? "No violations logged." : "Nothing logged yet.";
      box.querySelector(".empty-body").textContent = s.checks_total
        ? "Every check so far found everyone protected. Switch to All checks to see them."
        : "Start the camera or turn on the simulated site cameras. Checks appear here within seconds.";
    }

    list.forEach(function (e) {
      var li = el("li", "row" + (fresh[e.id] ? " is-new" : "")), thumb;
      if (e.img) {
        thumb = el("button", "thumb"); thumb.type = "button";
        thumb.setAttribute("aria-label", "Open photo from " + e.camera);
        var img = el("img"); img.alt = ""; img.src = e.img; thumb.appendChild(img);
        thumb.addEventListener("click", function () { openViewer(e); });
      } else thumb = el("div", "thumb none");
      li.appendChild(thumb);
      var text = el("div");
      text.appendChild(el("p", "what", describe(e)));
      text.appendChild(el("p", "where", e.camera + " · " + ago(e.ts)));
      li.appendChild(text);
      li.appendChild(el("span", "tag " + (e.status === "violation" ? "bad" : "ok"), e.status === "violation" ? "Violation" : "Clear"));
      rows.appendChild(li);
    });
  }

  // Mirrors the Prometheus rule LowPPECompliance: under 80% fully protected for 1 minute.
  function renderAlert(now) {
    var r = rateSince(now, 60000), bar = $("alertbar");
    if (r !== null && r < 80) { if (belowSince === null) belowSince = now; } else belowSince = null;
    if (belowSince === null) { bar.hidden = true; wasFiring = false; return; }
    var firing = now - belowSince >= 60000;
    bar.hidden = false;
    bar.className = "alertbar " + (firing ? "firing" : "pending");
    bar.textContent = "";
    bar.appendChild(document.createTextNode(firing ? "ALERT · LowPPECompliance firing" : "LowPPECompliance pending"));
    bar.appendChild(el("small", null, r + "% fully protected in the last minute (threshold 80%" +
      (firing ? ")." : ", fires after 1 minute: " + Math.max(0, Math.ceil((60000 - (now - belowSince)) / 1000)) + " s left).")));
    if (firing && !wasFiring && navigator.vibrate) navigator.vibrate([200, 100, 200]);
    wasFiring = firing;
  }

  function render() {
    var now = Date.now(), s = summary(now);
    renderSummary(s);
    if (state.renderedVersion !== version || state.renderedFilter !== state.filter) {
      renderEvents(s);
      state.renderedVersion = version; state.renderedFilter = state.filter;
    }
    renderAlert(now);
  }

  function openViewer(e) {
    $("viewer-img").src = e.img;
    $("viewer-img").alt = "Photo from " + e.camera + ": " + describe(e);
    $("viewer-cap").textContent = e.camera + " · " + new Date(e.ts).toLocaleString() + " · " + describe(e);
    var dlg = $("viewer");
    if (dlg.showModal) dlg.showModal();
  }

  document.querySelectorAll(".seg-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      state.filter = btn.getAttribute("data-filter");
      state.seen = null;
      document.querySelectorAll(".seg-btn").forEach(function (b) {
        var on = b === btn;
        b.classList.toggle("is-on", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      render();
    });
  });

  $("cam-start").addEventListener("click", startCamera);
  $("cam-stop").addEventListener("click", stopCamera);
  $("cam-flip").addEventListener("click", function () {
    facing = facing === "environment" ? "user" : "environment";
    startCamera();
  });
  $("sim-on").addEventListener("change", function (e) { setSim(e.target.checked); });

  fetch("ppe_model.json", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (spec) {
    spec.models.hardhat.scale = spec.models.hardhat.scale.map(function (v) { return v || 1; });
    spec.models.vest.scale = spec.models.vest.scale.map(function (v) { return v || 1; });
    MODEL = spec;
    var node = $("versions");
    node.textContent = "App live-demo · Model " + spec.version + " (" + spec.algorithm + ")";
    node.title = "Trained " + spec.trained_at;
  }).catch(function () {
    $("versions").textContent = "App live-demo · Model rules-0.6 (overlap rule)";
  });

  setSim(true);
  setInterval(render, 1000);
  render();
  // Start downloading the detector straight away so the camera starts quickly.
  if (window.cocoSsd) loadDetector().then(function () { if (!running) setStatus("Camera is off. Detector ready."); })
    .catch(function () { setStatus("Could not load the person detector. Check the internet connection.", "bad"); detectorLoading = null; });
})();
