(function () {
  "use strict";

  var POLL_MS = 3000;
  var state = { filter: "violations", seen: null };

  var $ = function (id) { return document.getElementById(id); };

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
    if (h < 24) return h + (h === 1 ? " hour ago" : " hours ago");
    return new Date(ms).toLocaleDateString();
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

  /* ---- Right now ---- */
  function renderSummary(s) {
    $("window").textContent = s.window_minutes;

    var rate = $("rate");
    rate.className = "big";
    if (s.protected_rate === null) {
      rate.textContent = "–";
      $("rate-sub").textContent = s.checks_total
        ? "No one has been on camera in this window."
        : "No people detected yet.";
    } else {
      rate.textContent = s.protected_rate + "%";
      rate.classList.add(s.protected_rate >= 95 ? "good" : s.protected_rate >= 75 ? "warn" : "alert");
      $("rate-sub").textContent =
        s.protected + " of " + plural(s.people, "person", "people") + " wore a hard hat and vest.";
    }

    [["t-hat", s.no_hardhat], ["t-vest", s.no_vest], ["t-both", s.no_both]].forEach(function (p) {
      var n = $(p[0]);
      n.textContent = p[1];
      n.classList.toggle("hot", p[1] > 0);
    });

    renderStatus(s);
    renderHours(s.hourly);
    renderCameras(s.cameras);
  }

  function renderStatus(s) {
    var live = s.cameras.filter(function (c) { return c.live; }).length;
    var dot = $("live-dot");
    dot.className = "dot";
    if (!s.cameras.length) {
      $("live-text").textContent = "Waiting for a camera";
    } else if (live) {
      dot.classList.add("live");
      $("live-text").textContent = plural(live, "camera", "cameras") + " live";
    } else {
      dot.classList.add("stale");
      $("live-text").textContent = "Cameras offline";
    }
  }

  function renderHours(hourly) {
    var wrap = $("hours");
    wrap.textContent = "";
    var max = Math.max.apply(null, hourly.map(function (h) { return h.checks; }).concat([1]));
    hourly.forEach(function (h, i) {
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
      var label = el("div", "hour-label", i % 3 === 0 ? new Date(h.hour_start).getHours() + "h" : " ");
      col.appendChild(label);
      wrap.appendChild(col);
    });
  }

  function renderCameras(cams) {
    var list = $("cams");
    list.textContent = "";
    $("cams-empty").hidden = cams.length > 0;
    cams.forEach(function (c) {
      var li = el("li");
      li.appendChild(el("span", "name", c.id));
      li.appendChild(el("span", "seen", c.live ? "Live" : "Last seen " + ago(c.last_seen)));
      list.appendChild(li);
    });
  }

  /* ---- Log ---- */
  function renderEmpty(events, summary) {
    var box = $("empty");
    box.hidden = events.length > 0;
    if (box.hidden) return;
    var hasChecks = summary.checks_total > 0;
    box.querySelector(".empty-title").textContent = hasChecks ? "No violations logged." : "Nothing logged yet.";
    box.querySelector(".empty-body").textContent = hasChecks
      ? "Every check so far found everyone protected. Switch to All checks to see them."
      : "Start the backend, then run the detector. Checks appear here within seconds.";
    box.querySelector("pre").hidden = hasChecks;
  }

  function renderEvents(events, summary) {
    var rows = $("rows");
    var firstLoad = state.seen === null;
    var fresh = {};
    var nextSeen = {};
    events.forEach(function (e) {
      nextSeen[e.id] = true;
      if (!firstLoad && !state.seen[e.id]) fresh[e.id] = true;
    });
    state.seen = nextSeen;

    rows.textContent = "";
    renderEmpty(events, summary);

    events.forEach(function (e) {
      var li = el("li", "row" + (fresh[e.id] ? " is-new" : ""));

      var thumb;
      if (e.has_image) {
        thumb = el("button", "thumb");
        thumb.type = "button";
        thumb.setAttribute("aria-label", "Open photo from " + e.camera);
        var img = el("img");
        img.loading = "lazy";
        img.alt = "";
        img.src = "/api/events/" + e.id + "/image";
        thumb.appendChild(img);
        thumb.addEventListener("click", function () { openViewer(e); });
      } else {
        thumb = el("div", "thumb none");
      }
      li.appendChild(thumb);

      var text = el("div");
      text.appendChild(el("p", "what", describe(e)));
      text.appendChild(el("p", "where", e.camera + " · " + ago(e.ts)));
      li.appendChild(text);

      li.appendChild(el("span", "tag " + (e.status === "violation" ? "bad" : "ok"),
        e.status === "violation" ? "Violation" : "Clear"));
      rows.appendChild(li);
    });
  }

  function openViewer(e) {
    $("viewer-img").src = "/api/events/" + e.id + "/image";
    $("viewer-img").alt = "Photo from " + e.camera + ": " + describe(e);
    $("viewer-cap").textContent = e.camera + " · " + new Date(e.ts).toLocaleString() + " · " + describe(e);
    var dlg = $("viewer");
    if (dlg.showModal) dlg.showModal();
  }

  /* ---- Polling ---- */
  function get(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(url + " " + r.status);
      return r.json();
    });
  }

  function refresh() {
    var q = "/api/events?limit=40" + (state.filter === "violations" ? "&violations=1" : "");
    Promise.all([get("/api/summary"), get(q)])
      .then(function (res) {
        renderSummary(res[0]);
        renderEvents(res[1], res[0]);
      })
      .catch(function () {
        var dot = $("live-dot");
        dot.className = "dot stale";
        $("live-text").textContent = "Can't reach the backend. Retrying.";
      });
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
      refresh();
    });
  });

  refresh();
  setInterval(refresh, POLL_MS);
})();
