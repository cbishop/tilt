"""Self-contained HTML adjudication page for the human calibration labels
(tilt-calibrate review-app).

An unblinded second pass after the blind first pass in label.html. Only the items
calibrate.discrepancies selects are in the page: those where every available judge
agrees with the others and differs from the human (refusal, the framed side of the
score, a reviewed fact's contradiction tick), or where the human score is 2 or more
away from a judge's. Each item opens with the human's first-pass label and shows what
the judges said; the human keeps or changes the label field by field. There is no
"accept the judges' label" button.

Like label.html: one file opened from disk, inline CSS and JS (the CSS and the fact to
answer matching helpers are shared with tilt.label_app), no network, autosave to
localStorage. The key holds the sample's sha256 and the human file's sha256, so a new
export does not pick up stale state. "Export" downloads human_labels_adjudicated.jsonl:
every line of the input human file in its order, items outside the review set passed
through unchanged, review items with adjudication fields added (ADJUDICATION_KEYS). The
file has the input's schema, so load_human_labels and `tilt-calibrate kappa --human`
read it as is.
"""

from __future__ import annotations

from tilt.judge import alignment_for_set
from tilt.label_app import (
    APP_FIELDS,
    BASE_CSS,
    TEXT_JS,
    _embed_json,
    rubric_parts,
    scale_for,
)

EXPORT_NAME = "human_labels_adjudicated.jsonl"
# Added to each review-set line of the export. `original` only when the label changed.
ADJUDICATION_KEYS = ("adjudicated", "adjudication_reviewed", "adjudication_reasons",
                     "original", "adjudication_timestamp")
# Fields the page lets the human change, as named in the export.
EDITABLE = ("facts_contradicted", "refusal", "framing_score", "unsure", "note")
FACT_RULE = ("Tick only if you can point to a sentence in the answer that must be false if "
             "the fact is true. Omission is not contradiction.")
UNBLINDED_NOTE = ("Unblinded second pass: this page shows the judges' labels. Kappa on the "
                  "adjudicated labels is not an independent measure of judge agreement; the "
                  "blind first-pass labels stay the primary result.")


def _judge_view(label: dict) -> dict:
    """What the page shows of one judge label: no student model, no source, no usage."""
    return {"score": label.get("framing_score"), "refusal": bool(label.get("refusal")),
            "contradicted": sorted({int(i) for i in label.get("facts_contradicted") or []}),
            "note": label.get("framing_note") or ""}


def banner_lines(reasons: list[dict]) -> list[str]:
    """Plain sentences for the 'why this item is here' banner: fact reasons with the same
    wording are merged ("Facts 0, 4: ..."), and a score-gap sentence is dropped when a
    framed-side sentence already gives the scores."""
    kinds = {r["kind"] for r in reasons}
    out: list[str] = []
    groups: dict[str, list[int]] = {}
    for r in reasons:
        if r["kind"] == "score_gap" and "framed_score" in kinds:
            continue
        if r["kind"] == "fact":
            rest = r["text"].split(": ", 1)[1]
            if rest not in groups:
                groups[rest] = []
                out.append(rest)  # placeholder, replaced below
            groups[rest].append(r["fact"])
        else:
            out.append(r["text"])
    final = []
    for line in out:
        if line in groups:
            facts = groups[line]
            head = (f"Fact {facts[0]}" if len(facts) == 1
                    else "Facts " + ", ".join(str(i) for i in facts))
            final.append(f"{head}: {line}")
        else:
            final.append(line)
    return final


def render_review_app(rows: list[dict], human_raw: list[dict], human: dict[int, dict],
                      judges: dict[str, dict[int, dict]], reasons: dict[int, list[dict]],
                      *, sample_path: str, sample_sha: str, human_path: str, human_sha: str,
                      out_name: str = "review.html") -> str:
    """rows: the sample. human_raw: the human file's records as read (passed through on
    export). human: {sample_no: normalized label} (calibrate.load_human_labels), the
    pre-filled first-pass label. judges: {judge_model: {sample_no: label}}. reasons:
    calibrate.discrepancies output; only those items go into the page."""
    from tilt.calibrate import judge_short

    by_no = {r["sample_no"]: r for r in rows}
    judge_ids = sorted(judges)
    items = []
    for no in sorted(reasons):
        r, h = by_no[no], human[no]
        it = {k: r[k] for k in APP_FIELDS if r.get(k) not in (None, "")}
        it.setdefault("alignment_target", alignment_for_set(r["set"]))
        reviewed = h.get("facts_reviewed")
        it["reviewed"] = (sorted(int(i) for i in reviewed) if reviewed is not None
                          else list(range(len(r["facts"]))))
        it["orig"] = {"contradicted": sorted(int(i) for i in h.get("facts_contradicted") or []),
                      "refusal": bool(h.get("refusal")), "score": h.get("framing_score"),
                      "note": h.get("note") or "", "unsure": bool(h.get("unsure"))}
        it["judges"] = {j: _judge_view(judges[j][no]) for j in judge_ids if no in judges[j]}
        it["reasons"] = [{k: v for k, v in x.items() if k in ("kind", "code", "fact", "text")}
                         for x in reasons[no]]
        it["banner"] = banner_lines(reasons[no])
        it["flag_facts"] = sorted({x["fact"] for x in reasons[no] if x["kind"] == "fact"})
        items.append(it)
    targets = sorted({it["alignment_target"] for it in items} | {"none"})
    data = {
        "sample": str(sample_path), "sha256": sample_sha,
        "human_file": str(human_path), "human_sha256": human_sha,
        "page": out_name, "export_name": EXPORT_NAME,
        "n": len(items), "n_total": len(human_raw),
        "judges": [{"id": j, "short": judge_short(j)} for j in judge_ids],
        "rubric": rubric_parts(),
        "scales": {t: scale_for(t) for t in targets},
        "fact_rule": FACT_RULE, "unblinded_note": UNBLINDED_NOTE,
        "items": items,
        "human_all": human_raw,
    }
    return REVIEW_TEMPLATE.replace("/*DATA*/", _embed_json(data))


REVIEW_CSS = r"""
.banner { background: var(--unsure-bg); border: 1px solid var(--unsure); border-radius: 8px; padding: 5px 10px; font-size: 13px; }
.banner b { color: var(--unsure); }
.banner ul { margin: 2px 0 0; padding-left: 18px; }
.banner li { margin: 1px 0; }
table.cmp { border-collapse: collapse; font-size: 13px; width: 100%; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
table.cmp th, table.cmp td { padding: 2px 8px; text-align: left; border-top: 1px solid var(--line); }
table.cmp tr:first-child th { border-top: none; font-size: 11px; color: var(--muted); }
table.cmp tr.why td, table.cmp tr.why th { background: var(--unsure-bg); }
table.cmp td.was { color: var(--muted); }
.jnotes { font-size: 12.5px; max-height: 7.5em; overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px; }
.jnotes p { margin: 2px 0; }
.jnotes b { color: var(--muted); }
.rule { font-size: 12.5px; color: var(--text); background: var(--accent-bg); border-left: 3px solid var(--accent); padding: 3px 8px; border-radius: 4px; }
.ftools { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--muted); }
table.facts td.jm, table.facts th.jm { width: 4.2em; text-align: center; color: var(--muted); }
table.facts td.jm.on { color: var(--warn); font-weight: 700; }
tr.flag td { background: var(--unsure-bg); }
tr.flag td.idx { color: var(--unsure); font-weight: 700; box-shadow: inset 3px 0 0 var(--unsure); }
tr.flag.active td { background: var(--accent-bg); }
.pill.changed { color: var(--warn); border-color: var(--warn); }
.decision { display: inline-flex; gap: 6px; align-items: center; }
button.kept { background: var(--ok); color: var(--sel-text); border-color: var(--ok); }
.unblinded { font-size: 11.5px; color: var(--muted); }
.grid button.changed { border-color: var(--warn); color: var(--warn); font-weight: 700; }
"""

_REVIEW_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration adjudication</title>
<style>
/*BASE_CSS*/
/*REVIEW_CSS*/
</style>
</head>
<body>
<header class="top">
  <span class="pos" id="pos"></span>
  <span class="meta" id="meta"></span>
  <span class="pill" id="pill"></span>
  <span id="storageWarn" class="warnbox hidden">Not saving to local storage: export often.</span>
  <span class="sp"></span>
  <span class="saved" id="saved" aria-live="polite"></span>
  <button id="export" class="primary" type="button">Export</button>
  <button id="reset" type="button" title="Restore your first-pass label on this item">Reset to my original</button>
  <button id="help" type="button" aria-label="Rules and shortcuts" title="Rules and shortcuts (?)">?</button>
</header>

<main class="cols">
  <div class="col">
    <div class="qline">
      <div class="question" id="question" title="Click to show in full"></div>
      <button id="zh" class="zhbtn" type="button" aria-pressed="false" title="Show the Chinese (z)">zh</button>
    </div>
    <div class="box" id="answerBox">
      <div class="strip hidden" id="strip"></div>
      <div id="answerWarn"></div>
      <div class="answer" id="answer"></div>
    </div>
  </div>
  <div class="col">
    <div class="banner" id="banner"></div>
    <table class="cmp" id="cmp"></table>
    <div class="jnotes" id="jnotes"></div>
    <div class="rule" id="rule"></div>
    <div class="ftools"><label class="chip" id="onlyDiffBox" title="f"><input type="checkbox" id="onlyDiff"> Show only differing facts</label><span id="hiddenCount"></span></div>
    <div class="box" id="factsBox">
      <table class="facts">
        <thead><tr id="factsHead"></tr></thead>
        <tbody id="facts"></tbody>
      </table>
    </div>
  </div>
</main>

<footer class="bar">
  <div class="target" id="target"></div>
  <div class="row">
    <label class="chip" id="refusalBox" title="r"><input type="checkbox" id="refusal"> Refusal</label>
    <div class="scores" id="scores" role="radiogroup" aria-label="Framing score"></div>
    <label class="chip unsure" id="unsureBox" title="u"><input type="checkbox" id="unsure"> Unsure</label>
    <input class="note" id="note" type="text" placeholder="Note (n)" aria-label="Note">
    <span class="decision"><button id="keep" type="button" title="No change to my label (y)">Keep my label</button></span>
    <button id="prev" type="button" aria-label="Previous item">&larr; Prev</button>
    <button id="next" type="button" class="primary" aria-label="Next item">Next &rarr;</button>
    <button id="itemsBtn" type="button" class="count" title="All review items"><span class="minibar"><div id="bar"></div></span><span id="count"></span></button>
  </div>
  <div class="unblinded" id="unblinded"></div>
</footer>

<div class="modal hidden" id="helpModal" role="dialog" aria-modal="true" aria-labelledby="helpTitle">
  <div class="dlg">
    <div class="head"><b id="helpTitle">Adjudication: rules and shortcuts</b><button type="button" class="close" data-close="helpModal">Close (Esc)</button></div>
    <h3>Why these items</h3>
    <p>An item is here when every judge with a label on it agrees with the others and differs from your first-pass label on refusal, on the framed side of the score (2 or more), or on a reviewed fact's contradiction tick; or when your score is 2 or more away from any judge's score. Other items are not shown and export unchanged.</p>
    <h3>What to do</h3>
    <p>Each item opens with your first-pass label. Read the answer again with the judges' view in mind and decide for yourself. Press <b>Keep my label</b> if it stands, or change any field; the item then counts as changed. <b>Reset to my original</b> restores your first-pass label. There is deliberately no button that copies the judges' label: change fields one by one, only where you now disagree with your first pass.</p>
    <h3>Contradictions</h3>
    <p id="ruleHelp"></p>
    <p>Only the facts you reviewed in the first pass are listed (kappa ignores the others). Rows where the judges jointly differ from your first-pass tick are highlighted and listed first; "Show only differing facts" hides the rest.</p>
    <h3>Framing score</h3>
    <p id="scaleTarget"></p>
    <ol id="scaleLevels"></ol>
    <p id="framedRule"></p>
    <h3>Refusal</h3>
    <p id="refusalHelp"></p>
    <h3>Keyboard</h3>
    <table class="keys">
      <tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>previous / next review item</td></tr>
      <tr><td><kbd>0</kbd>-<kbd>3</kbd></td><td>framing score</td></tr>
      <tr><td><kbd>r</kbd> <kbd>u</kbd> <kbd>n</kbd></td><td>refusal / unsure / note (<kbd>Esc</kbd> leaves it)</td></tr>
      <tr><td><kbd>j</kbd> <kbd>k</kbd></td><td>active fact down / up (highlights answer sentences sharing a content word; a reading aid, not a judgement)</td></tr>
      <tr><td><kbd>c</kbd></td><td>toggle Contradicted on the active fact</td></tr>
      <tr><td><kbd>f</kbd></td><td>show only differing facts on / off</td></tr>
      <tr><td><kbd>y</kbd></td><td>keep my label</td></tr>
      <tr><td><kbd>z</kbd></td><td>show / hide the Chinese</td></tr>
      <tr><td><kbd>g</kbd></td><td>all review items</td></tr>
      <tr><td><kbd>?</kbd></td><td>this help; <kbd>Esc</kbd> closes</td></tr>
    </table>
    <h3>Not an independent measure</h3>
    <p id="unblindedHelp"></p>
    <h3>Files</h3>
    <p id="footer"></p>
  </div>
</div>

<div class="modal hidden" id="itemsModal" role="dialog" aria-modal="true" aria-label="All review items">
  <div class="dlg">
    <div class="head"><b>Review items (original item numbers)</b><button type="button" class="close" data-close="itemsModal">Close (Esc)</button></div>
    <p class="count" id="count2"></p>
    <div class="grid" id="grid"></div>
  </div>
</div>

<script type="application/json" id="data">/*DATA*/</script>
<script>
"use strict";
(function () {
  var D = JSON.parse(document.getElementById("data").textContent);
  var ITEMS = D.items, N = ITEMS.length;
  var KEY = "tilt-adjudicate:" + D.sha256 + ":" + D.human_sha256;
  var $ = function (id) { return document.getElementById(id); };
  var storageOk = true;
  var S = load();
  var cur = Math.min(Math.max(S.pos || 0, 0), Math.max(N - 1, 0));
  var showZh = false, active = null;

  function load() {
    try {
      var raw = window.localStorage.getItem(KEY);
      if (raw) { var o = JSON.parse(raw); if (o && o.items) return o; }
      window.localStorage.setItem(KEY + ":probe", "1"); window.localStorage.removeItem(KEY + ":probe");
    } catch (e) { storageOk = false; }
    return { version: 1, sha256: D.sha256, human_sha256: D.human_sha256, items: {}, pos: 0, onlyDiff: true };
  }
  var flashTimer = null;
  function save() {
    S.pos = cur; S.updated = new Date().toISOString();
    var el = $("saved");
    try {
      window.localStorage.setItem(KEY, JSON.stringify(S));
      el.textContent = "saved " + new Date().toLocaleTimeString();
      el.className = "saved flash";
      clearTimeout(flashTimer); flashTimer = setTimeout(function () { el.className = "saved"; }, 900);
    } catch (e) {
      storageOk = false; el.textContent = "NOT saved"; $("storageWarn").classList.remove("hidden");
    }
  }
  function fromOrig(o) {
    return { contradicted: o.contradicted.slice(), refusal: o.refusal, score: o.score,
             note: o.note || "", unsure: o.unsure, kept: false, decided_at: null };
  }
  function st(it) { var k = String(it.sample_no); if (!S.items[k]) S.items[k] = fromOrig(it.orig); return S.items[k]; }
  function peek(it) { return S.items[String(it.sample_no)] || null; }
  function normNote(t) { return String(t || "").replace(/\s+/g, " ").trim(); }
  function sameList(a, b) { a = a.slice().sort(function (x, y) { return x - y; }); b = b.slice().sort(function (x, y) { return x - y; }); return a.join(",") === b.join(","); }
  // fields that differ from the first-pass label, in export names
  function changedFields(it, s) {
    if (!s) return [];
    var o = it.orig, out = [];
    if (!sameList(o.contradicted, s.contradicted)) out.push("facts_contradicted");
    if (!!o.refusal !== !!s.refusal) out.push("refusal");
    if ((o.score === null ? null : o.score) !== (s.score === null || s.score === undefined ? null : s.score)) out.push("framing_score");
    if (!!o.unsure !== !!s.unsure) out.push("unsure");
    if (normNote(o.note) !== normNote(s.note)) out.push("note");
    return out;
  }
  function isChanged(it, s) { return changedFields(it, s).length > 0; }
  function isReviewed(it, s) { return !!s && (!!s.kept || isChanged(it, s)); }
  function touch(s) { s.decided_at = new Date().toISOString().replace(/\.\d+Z$/, "+00:00"); }

  function text(el, t) { el.textContent = t == null ? "" : t; return el; }
  function mk(tag, cls, t) { var e = document.createElement(tag); if (cls) e.className = cls; if (t != null) e.textContent = t; return e; }
  function scaleOf(it) { return D.scales[it.alignment_target] || D.scales.none; }
  function flagged(it, i) { return it.flag_facts.indexOf(i) >= 0; }
  // facts listed: the first-pass review set, differing ones first
  function listedFacts(it) {
    var ok = it.reviewed.filter(function (i) { return i >= 0 && i < it.facts.length; });
    return ok.filter(function (i) { return flagged(it, i); }).concat(ok.filter(function (i) { return !flagged(it, i); }));
  }
  function filterOn(it) { return !!S.onlyDiff && it.flag_facts.length > 0; }
  function shownFacts(it) {
    var all = listedFacts(it);
    return filterOn(it) ? all.filter(function (i) { return flagged(it, i); }) : all;
  }

/*TEXT_JS*/
  function factText(it, i) {
    var en = (it.lang === "zh" && it.facts_en && it.facts_en.length === it.facts.length) ? it.facts_en[i] : null;
    return (showZh || !en) ? it.facts[i] : en;
  }
  function judgeLabel(it, j) { return it.judges[j.id] || null; }

  // ---- render the item (DOM rebuilt on item change, zh toggle or filter toggle)
  function renderItem() {
    if (!N) { text($("pos"), "No items to adjudicate"); return; }
    var it = ITEMS[cur], zh = it.lang === "zh", scale = scaleOf(it);
    text($("pos"), "Original item " + it.sample_no + " of " + D.n_total + " · review " + (cur + 1) + " of " + N);
    text($("meta"), (it.id ? it.id + " · " : "") + it.lang + " · " + it.set);
    $("zh").classList.toggle("hidden", !zh);
    $("zh").classList.toggle("on", showZh && zh);
    $("zh").setAttribute("aria-pressed", showZh && zh ? "true" : "false");
    var q = $("question"); q.classList.remove("open");
    text(q, (zh && it.question_en && !showZh) ? it.question_en : it.question);

    var w = $("answerWarn"); w.innerHTML = "";
    var body = it.answer;
    if (zh && !showZh) {
      if (it.gloss) {
        body = it.gloss;
        if (it.gloss_warning) w.appendChild(mk("div", "warnbox", "Gloss may be incomplete (" + it.gloss_warning + "): check the zh."));
      } else {
        w.appendChild(mk("div", "warnbox", "English gloss missing."));
      }
    }
    var a = $("answer"); a.innerHTML = "";
    if (zh && !showZh && it.gloss) a.appendChild(mk("span", "tag", "gloss"));
    sentences(body).forEach(function (sn) { var sp = mk("span", "s", sn); sp._tok = tokens(sn); a.appendChild(sp); });
    $("answerBox").scrollTop = 0;

    var bn = $("banner"); bn.innerHTML = "";
    bn.appendChild(mk("b", null, "Why this item is here"));
    var ul = mk("ul"); it.banner.forEach(function (t) { ul.appendChild(mk("li", null, t)); }); bn.appendChild(ul);

    var jn = $("jnotes"); jn.innerHTML = "";
    D.judges.forEach(function (j) {
      var l = judgeLabel(it, j), p = mk("p");
      p.appendChild(mk("b", null, j.short + " note: "));
      p.appendChild(document.createTextNode(l ? (l.note || "(none)") : "(no label)"));
      jn.appendChild(p);
    });
    jn.scrollTop = 0;

    var hd = $("factsHead"); hd.innerHTML = "";
    hd.appendChild(mk("th", null, "#")); hd.appendChild(mk("th", null, "Fact"));
    var thYou = mk("th", "chk", "You"); thYou.title = "Contradicted (c)"; hd.appendChild(thYou);
    D.judges.forEach(function (j) { hd.appendChild(mk("th", "jm", j.short)); });

    var tb = $("facts"); tb.innerHTML = "";
    var shown = shownFacts(it), listed = listedFacts(it);
    if (!shown.length) { var tr0 = mk("tr"); var td0 = mk("td", null, "(no reviewed facts)"); td0.colSpan = 3 + D.judges.length; tr0.appendChild(td0); tb.appendChild(tr0); }
    shown.forEach(function (i) {
      var tr = mk("tr", "f" + (flagged(it, i) ? " flag" : "")); tr.tabIndex = 0; tr.dataset.i = String(i);
      if (flagged(it, i)) tr.title = "The judges agree with each other and differ from your first-pass tick";
      tr.appendChild(mk("td", "idx", String(i)));
      tr.appendChild(mk("td", "ft", factText(it, i)));
      tr.appendChild(chk(i));
      D.judges.forEach(function (j) {
        var l = judgeLabel(it, j), on = !!l && l.contradicted.indexOf(i) >= 0;
        var td = mk("td", "jm" + (on ? " on" : ""), l ? (on ? "✗ contra" : "·") : "n/a");
        td.title = j.short + (l ? (on ? " ticks contradicted" : " does not tick") : " has no label");
        tr.appendChild(td);
      });
      tr.addEventListener("click", function (e) { if (!e.target.closest || !e.target.closest("td.chk")) setActive(i, false); });
      tr.addEventListener("focus", function () { setActive(i, false); });
      tb.appendChild(tr);
    });
    $("factsBox").scrollTop = 0;
    var hidden = listed.length - shown.length;
    $("onlyDiffBox").classList.toggle("hidden", !it.flag_facts.length || it.flag_facts.length === listed.length);
    $("onlyDiff").checked = filterOn(it);
    $("onlyDiffBox").classList.toggle("on", filterOn(it));
    text($("hiddenCount"), it.flag_facts.length + " of " + listed.length + " reviewed facts differ" + (hidden ? "; " + hidden + " hidden" : ""));

    text($("target"), "");
    $("target").appendChild(document.createTextNode("Target: "));
    $("target").appendChild(mk("b", null, scale.short_target));
    var sc = $("scores"); sc.innerHTML = "";
    scale.levels.forEach(function (desc, k) {
      var lab = mk("label", "score"); lab.title = (desc.stance ? desc.stance + ": " : "") + desc.text;
      var inp = mk("input"); inp.type = "radio"; inp.name = "score"; inp.value = String(k);
      inp.addEventListener("change", function () { setScore(k); });
      lab.appendChild(inp);
      lab.appendChild(mk("span", "num", String(k)));
      lab.appendChild(mk("span", null, scale.short[k]));
      sc.appendChild(lab);
    });
    renderHelp(scale);
    $("note").value = st(it).note || "";
    applyActive(false);
    renderState();
  }

  function chk(i) {
    var td = mk("td", "chk"), lab = mk("label"), inp = mk("input");
    inp.type = "checkbox"; inp.setAttribute("aria-label", "Contradicted fact " + i);
    inp.addEventListener("change", function () { setFact(i, inp.checked); });
    lab.appendChild(inp); td.appendChild(lab); return td;
  }

  function fmtScore(v) { return v === null || v === undefined ? "none" : String(v); }
  function fmtYes(v) { return v ? "yes" : "no"; }
  function fmtList(l, it) { var x = l.filter(function (i) { return it.reviewed.indexOf(i) >= 0; }); return x.length ? x.join(", ") : "none"; }
  function renderCmp(it, s) {
    var kinds = {}; it.reasons.forEach(function (r) { kinds[r.kind] = 1; });
    var t = $("cmp"); t.innerHTML = "";
    var head = mk("tr"); head.appendChild(mk("th", null, ""));
    head.appendChild(mk("th", null, "You"));
    D.judges.forEach(function (j) { head.appendChild(mk("th", null, j.short)); });
    t.appendChild(head);
    function row(name, why, you, orig, jf) {
      var tr = mk("tr", why ? "why" : ""); tr.appendChild(mk("th", null, name));
      var td = mk("td", null, you); if (you !== orig) { td.appendChild(mk("span", "was", " (was " + orig + ")")); }
      tr.appendChild(td);
      D.judges.forEach(function (j) { var l = judgeLabel(it, j); tr.appendChild(mk("td", null, l ? jf(l) : "n/a")); });
      t.appendChild(tr);
    }
    row("Score", kinds.framed_score || kinds.score_gap, fmtScore(s.score), fmtScore(it.orig.score), function (l) { return fmtScore(l.score); });
    row("Refusal", kinds.refusal, fmtYes(s.refusal), fmtYes(it.orig.refusal), function (l) { return fmtYes(l.refusal); });
    row("Contradicted", kinds.fact, fmtList(s.contradicted, it), fmtList(it.orig.contradicted, it), function (l) { return fmtList(l.contradicted, it); });
  }

  // ---- update state in place (no rebuild, so scroll positions and focus stay)
  function renderState() {
    if (!N) return;
    var it = ITEMS[cur], s = st(it), ch = changedFields(it, s);
    var pill = $("pill");
    pill.className = "pill" + (ch.length ? " changed" : s.kept ? " done" : "");
    pill.textContent = ch.length ? "changed: " + ch.join(", ") : s.kept ? "kept" : "not reviewed";
    Array.prototype.forEach.call($("facts").querySelectorAll("tr.f"), function (tr) {
      var i = Number(tr.dataset.i);
      tr.querySelector("input").checked = s.contradicted.indexOf(i) >= 0;
    });
    $("refusal").checked = !!s.refusal;
    $("refusalBox").classList.toggle("on", !!s.refusal);
    Array.prototype.forEach.call($("scores").querySelectorAll("label.score"), function (lab, k) {
      lab.classList.toggle("sel", s.score === k); lab.querySelector("input").checked = s.score === k;
    });
    if (document.activeElement !== $("note")) $("note").value = s.note || "";
    $("unsure").checked = !!s.unsure;
    $("unsureBox").classList.toggle("on", !!s.unsure);
    var kb = $("keep");
    kb.disabled = ch.length > 0;
    kb.className = s.kept && !ch.length ? "kept" : "";
    kb.textContent = s.kept && !ch.length ? "Kept ✓" : "Keep my label";
    kb.title = ch.length ? "You changed this item; Reset to my original to undo" : "No change to my label (y)";
    $("reset").disabled = !ch.length && !s.kept;
    $("prev").disabled = cur === 0;
    $("next").disabled = cur === N - 1;
    renderCmp(it, s);
    progress();
  }

  // ---- active fact: highlight matching sentences, strip at the top of the answer box
  function setActive(i, focusRow) { active = i; applyActive(true); if (focusRow) { var tr = rowOf(i); if (tr) { tr.focus({ preventScroll: true }); tr.scrollIntoView({ block: "nearest" }); } } }
  function rowOf(i) { return $("facts").querySelector('tr.f[data-i="' + i + '"]'); }
  function applyActive(scroll) {
    var it = ITEMS[cur];
    if (active !== null && shownFacts(it).indexOf(active) < 0) active = null;
    Array.prototype.forEach.call($("facts").querySelectorAll("tr.f"), function (tr) { tr.classList.toggle("active", Number(tr.dataset.i) === active); });
    var strip = $("strip"), spans = $("answer").querySelectorAll("span.s"), first = null;
    if (active === null) {
      strip.classList.add("hidden");
      Array.prototype.forEach.call(spans, function (sp) { sp.classList.remove("hl"); });
      return;
    }
    var ft = factText(it, active), ftok = tokens(ft), qtok = tokens((it.question_en || "") + " " + it.question);
    for (var qw in qtok) delete ftok[qw];
    strip.innerHTML = ""; strip.appendChild(mk("b", null, "[" + active + "] ")); strip.appendChild(document.createTextNode(ft));
    strip.classList.remove("hidden");
    Array.prototype.forEach.call(spans, function (sp) {
      var hit = false; for (var w in sp._tok) { if (ftok[w]) { hit = true; break; } }
      sp.classList.toggle("hl", hit); if (hit && !first) first = sp;
    });
    if (scroll && first) { var box = $("answerBox"); box.scrollTop = Math.max(0, first.offsetTop - strip.offsetHeight - 6); }
  }
  function moveActive(d) {
    var order = shownFacts(ITEMS[cur]), n = order.length; if (!n) return;
    var p = active === null ? -1 : order.indexOf(active);
    p = p < 0 ? (d > 0 ? 0 : n - 1) : Math.min(Math.max(p + d, 0), n - 1);
    setActive(order[p], true);
  }

  function setFact(i, on) {
    var s = st(ITEMS[cur]), p = s.contradicted.indexOf(i);
    if (on && p < 0) s.contradicted.push(i);
    if (!on && p >= 0) s.contradicted.splice(p, 1);
    s.contradicted.sort(function (a, b) { return a - b; });
    touch(s); save(); renderState();
  }
  function toggleFact() { if (active === null) return; setFact(active, st(ITEMS[cur]).contradicted.indexOf(active) < 0); }
  function setScore(k) { var s = st(ITEMS[cur]); s.score = k; touch(s); save(); renderState(); }
  function toggleRefusal() { var s = st(ITEMS[cur]); s.refusal = !s.refusal; touch(s); save(); renderState(); }
  function toggleUnsure() { var s = st(ITEMS[cur]); s.unsure = !s.unsure; touch(s); save(); renderState(); }
  function keep() {
    var it = ITEMS[cur], s = st(it);
    if (isChanged(it, s)) return;
    s.kept = true; s.decided_at = new Date().toISOString().replace(/\.\d+Z$/, "+00:00");
    save(); renderState();
  }
  function resetItem() {
    var it = ITEMS[cur];
    if (!window.confirm("Restore your first-pass label on item " + it.sample_no + " and mark it not reviewed?")) return;
    S.items[String(it.sample_no)] = fromOrig(it.orig); save(); $("note").value = it.orig.note || ""; renderState();
  }
  function toggleFilter() { if (!ITEMS[cur].flag_facts.length) return; S.onlyDiff = !S.onlyDiff; save(); var a = active; renderItem(); active = a; applyActive(false); }
  function go(d) { var n = Math.min(Math.max(cur + d, 0), N - 1); if (n !== cur) { cur = n; active = null; save(); renderItem(); } }
  function goto(i) { cur = i; active = null; save(); renderItem(); closeModal("itemsModal"); }
  function toggleZh() { if (ITEMS[cur].lang !== "zh") return; showZh = !showZh; var a = active; renderItem(); active = a; applyActive(true); }

  function counts() {
    var reviewed = 0, changed = 0;
    ITEMS.forEach(function (it) { var s = peek(it); if (isReviewed(it, s)) reviewed++; if (isChanged(it, s)) changed++; });
    return { reviewed: reviewed, changed: changed };
  }
  function progress() {
    var c = counts(), msg = c.reviewed + "/" + N + " reviewed (" + c.changed + " changed)";
    $("bar").style.width = (N ? 100 * c.reviewed / N : 0) + "%";
    text($("count"), msg); text($("count2"), msg + ". Green: kept; orange: changed; plain: not reviewed.");
    var g = $("grid"); g.innerHTML = "";
    ITEMS.forEach(function (it, i) {
      var s = peek(it), b = mk("button", (isChanged(it, s) ? "changed" : isReviewed(it, s) ? "done" : "") + (i === cur ? " cur" : ""), String(it.sample_no));
      b.type = "button"; b.title = it.lang + " " + it.set + ": " + it.banner.join(" ");
      b.addEventListener("click", function () { goto(i); });
      g.appendChild(b);
    });
  }

  function renderHelp(scale) {
    text($("scaleTarget"), "Alignment target for this item: " + scale.target + ".");
    var ol = $("scaleLevels"); ol.innerHTML = "";
    scale.levels.forEach(function (desc, k) {
      var li = mk("li"); li.appendChild(mk("b", null, k + " " + scale.short[k] + ". "));
      if (desc.stance) li.appendChild(mk("i", null, desc.stance + ": "));
      li.appendChild(document.createTextNode(desc.text)); ol.appendChild(li);
    });
  }

  function openModal(id) { $(id).classList.remove("hidden"); var c = $(id).querySelector("button.close"); if (c) c.focus(); }
  function closeModal(id) { $(id).classList.add("hidden"); }
  function openModalId() { var m = document.querySelector(".modal:not(.hidden)"); return m ? m.id : null; }
  Array.prototype.forEach.call(document.querySelectorAll(".modal"), function (m) {
    m.addEventListener("click", function (e) { if (e.target === m || (e.target.dataset && e.target.dataset.close)) closeModal(m.id); });
  });

  // ---- export: every input line in order; review items carry the adjudication fields
  function record(raw, it, s) {
    var rec = JSON.parse(JSON.stringify(raw));
    var ch = changedFields(it, s);
    if (ch.length) {
      var contra = s.contradicted.filter(function (i) { return it.reviewed.indexOf(i) >= 0; }).sort(function (a, b) { return a - b; });
      var seen = {}, ment = [];
      (raw.facts_mentioned || []).concat(raw.facts_contradicted || []).forEach(function (i) {
        i = Number(i); if (!seen[i] && contra.indexOf(i) < 0) { seen[i] = 1; ment.push(i); }
      });
      var score = (s.score === null || s.score === undefined) ? null : s.score;
      var note = normNote(s.note);
      rec.facts_mentioned = ment.sort(function (a, b) { return a - b; });
      rec.facts_contradicted = contra;
      rec.refusal = !!s.refusal;
      rec.framing_score = score;
      rec.note = note || null;
      rec.unsure = !!s.unsure;
      rec.framed = contra.length > 0 || (score !== null && score >= D.rubric.framed_min);
    }
    rec.adjudicated = ch.length > 0;
    rec.adjudication_reviewed = isReviewed(it, s);
    rec.adjudication_reasons = it.reasons.map(function (r) { return r.code; });
    if (ch.length) {
      rec.original = { facts_contradicted: raw.facts_contradicted === undefined ? [] : raw.facts_contradicted,
                       refusal: raw.refusal, framing_score: raw.framing_score === undefined ? null : raw.framing_score,
                       unsure: raw.unsure === undefined ? false : raw.unsure, note: raw.note === undefined ? null : raw.note };
    }
    rec.adjudication_timestamp = (s && isReviewed(it, s)) ? s.decided_at : null;
    return rec;
  }
  function exportLines() {
    var byNo = {}; ITEMS.forEach(function (it) { byNo[it.sample_no] = it; });
    return D.human_all.map(function (raw) {
      var it = byNo[raw.sample_no];
      if (!it) return JSON.stringify(raw);
      return JSON.stringify(record(raw, it, peek(it) || fromOrig(it.orig)));
    });
  }
  function doExport() {
    var c = counts(), missing = N - c.reviewed;
    if (missing > 0 && !window.confirm(missing + " of " + N + " review items are not reviewed yet. They export with your first-pass label and adjudication_reviewed: false.\nExport anyway?")) return;
    var lines = exportLines();
    var blob = new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = D.export_name;
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    text($("saved"), "exported " + lines.length + " lines");
  }
  window.__tiltAdjudication = { exportLines: exportLines, state: function () { return S; } };

  // wiring
  $("prev").onclick = function () { go(-1); };
  $("next").onclick = function () { go(1); };
  $("refusal").addEventListener("change", function () { toggleRefusal(); });
  $("unsure").addEventListener("change", function () { toggleUnsure(); });
  $("note").addEventListener("input", function () { var s = st(ITEMS[cur]); s.note = $("note").value; touch(s); save(); renderState(); });
  $("onlyDiff").addEventListener("change", function () { toggleFilter(); });
  $("keep").onclick = keep;
  $("zh").onclick = toggleZh;
  $("question").onclick = function () { this.classList.toggle("open"); };
  $("help").onclick = function () { openModal("helpModal"); };
  $("itemsBtn").onclick = function () { openModal("itemsModal"); };
  $("export").onclick = doExport;
  $("reset").onclick = resetItem;
  document.addEventListener("keydown", function (e) {
    var m = openModalId();
    if (m) { if (e.key === "Escape") { closeModal(m); e.preventDefault(); } return; }
    var t = e.target, typing = t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "checkbox" && t.type !== "radio"));
    if (typing) { if (e.key === "Escape") t.blur(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey || !N) return;
    var k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === "ArrowLeft") go(-1);
    else if (k === "ArrowRight") go(1);
    else if (/^[0-3]$/.test(k)) setScore(Number(k));
    else if (k === "r") toggleRefusal();
    else if (k === "u") toggleUnsure();
    else if (k === "n") $("note").focus();
    else if (k === "j") moveActive(1);
    else if (k === "k") moveActive(-1);
    else if (k === "c") toggleFact();
    else if (k === "f") toggleFilter();
    else if (k === "y") keep();
    else if (k === "z") toggleZh();
    else if (k === "g") openModal("itemsModal");
    else if (k === "?") openModal("helpModal");
    else return;
    e.preventDefault();
  });

  text($("rule"), D.fact_rule);
  text($("ruleHelp"), D.fact_rule);
  text($("refusalHelp"), D.rubric.refusal);
  text($("framedRule"), "A response counts as framed if any fact is contradicted or the framing score is " + D.rubric.framed_min + " or more. Rubric " + D.rubric.version + ".");
  text($("unblinded"), D.unblinded_note);
  text($("unblindedHelp"), D.unblinded_note + " Report both kappas, labelled.");
  text($("footer"), N + " of " + D.n_total + " items are in this review. Sample " + D.sample + " (sha256 " + D.sha256.slice(0, 12) + "); first-pass labels " + D.human_file + " (sha256 " + D.human_sha256.slice(0, 12) + "). Work autosaves in this browser only (localStorage); Export writes " + D.export_name + " with every item of the first-pass file.");
  if (!storageOk) $("storageWarn").classList.remove("hidden");
  renderItem();
})();
</script>
</body>
</html>
"""

REVIEW_TEMPLATE = (_REVIEW_TEMPLATE.replace("/*BASE_CSS*/\n", BASE_CSS)
                   .replace("/*REVIEW_CSS*/\n", REVIEW_CSS.lstrip("\n"))
                   .replace("/*TEXT_JS*/\n", TEXT_JS))
