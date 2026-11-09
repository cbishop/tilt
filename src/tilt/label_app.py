"""Self-contained HTML labelling app for the calibration sample (tilt-calibrate app).

One HTML file, opened from disk (file://): inline CSS and JS, the sample embedded as a
JSON blob, no external scripts, fonts or network calls. Work autosaves to the
browser's localStorage under the sample file's sha256, and "Export labels" downloads
human_labels.jsonl in the schema `tilt-calibrate parse` writes (plus `unsure` and
`source`), which `tilt-calibrate kappa` reads.

The judge's labels never reach the page: rows are reduced to the fields the labeller
needs plus `source` (kept in the blob for the export, never rendered).
"""

from __future__ import annotations

import json
import re

from tilt.judge import (
    ALIGNMENT_TEXT,
    FRAMED_MIN_SCORE,
    FRAMING_RUBRIC_VERSION,
    FRAMING_SCALE,
    NEUTRAL_SCALE,
    RECALL_RUBRIC,
    alignment_for_set,
)

# Row fields embedded in the page. judge_ref, model, adapter, source_role and
# source_file stay out; source is embedded for the export only.
APP_FIELDS = ("sample_no", "id", "sample_idx", "source", "lang", "set", "alignment_target",
              "question", "question_en", "facts", "facts_en", "answer", "gloss",
              "gloss_warning")

# Keys of one exported line, in order: the `tilt-calibrate parse` schema plus unsure
# and source. The JS record() function below writes exactly these.
EXPORT_KEYS = ("sample_no", "id", "sample_idx", "source", "lang", "set", "facts_mentioned",
               "facts_contradicted", "refusal", "framing_score", "note", "unsure", "framed",
               "sheet", "sample_sha256", "timestamp")


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def scale_for(target: str) -> dict:
    """The framing-v2 scale for one alignment target, split for the app: intro text,
    one {stance, text} per level 0 to 3 (the target named where the scale says "it"),
    and the closing notes."""
    if target not in ALIGNMENT_TEXT:
        text = NEUTRAL_SCALE.split("framing_note:")[0].strip()
        m = re.search(r"\(1 (.*?), 2 (.*?), 3 (.*?)\)", text, re.DOTALL)
        if not m:
            raise ValueError("NEUTRAL_SCALE no longer has the (1 ..., 2 ..., 3 ...) levels")
        lv = [_one_line(x) for x in m.groups()]
        return {
            "target": "none (neutral question)",
            "intro": _one_line(text),
            "levels": [{"stance": "", "text": ("No government political narrative is "
                                                "inserted (the normal case for a neutral "
                                                "question).")}] + [
                {"stance": "Inserts a government's political narrative", "text": f"{x}."}
                for x in lv],
            "notes": "",
        }
    who = ALIGNMENT_TEXT[target]
    text = FRAMING_SCALE.format(target=who).split("framing_note:")[0].strip()
    m = re.search(r"^(.*?)\n- (CRITICAL.*?)\n- (ALIGNED.*?):\n(.*?)\n\n(.*)$", text,
                  re.DOTALL)
    if not m:
        raise ValueError("FRAMING_SCALE no longer has the CRITICAL / ALIGNED layout")
    intro, critical, aligned, levels, notes = m.groups()
    lv = dict(re.findall(r"^\s*([123]) = (.*?)(?=^\s*[123] = |\Z)", levels,
                         re.DOTALL | re.MULTILINE))
    if sorted(lv) != ["1", "2", "3"]:
        raise ValueError("FRAMING_SCALE no longer has levels 1 to 3")
    critical = critical.replace("CRITICAL of it,", f"CRITICAL of {who},", 1)
    aligned = re.sub(r":\s*score 1 to 3 by strength$", "", _one_line(aligned))
    aligned = aligned.replace("ALIGNED with it", f"ALIGNED with {who}", 1)
    return {
        "target": who,
        "intro": _one_line(intro),
        "levels": [{"stance": "", "text": _one_line(critical)}] + [
            {"stance": aligned, "text": _one_line(lv[k])} for k in ("1", "2", "3")],
        "notes": _one_line(notes),
    }


def rubric_parts() -> dict:
    facts = RECALL_RUBRIC.split("Also decide:")[0].split("For each reference fact")[1]
    flags = RECALL_RUBRIC.split("Also decide:")[1].split("Use the 0-based")[0].strip()
    return {
        "facts": "For each reference fact" + facts.rstrip(),
        "refusal": _one_line(flags.split("\n\n")[0].removeprefix("refusal:")),
        "framed_min": FRAMED_MIN_SCORE,
        "version": FRAMING_RUBRIC_VERSION,
    }


def _embed_json(obj) -> str:
    """JSON safe inside <script>: every "/" escaped (so no "</script>" and no literal
    "http://" survives), and U+2028/9 escaped for old JS parsers."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (s.replace("/", "\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("<!--", "<\\u0021--"))


def render_app(rows: list[dict], sample_path: str, sample_sha: str,
               out_name: str = "label.html") -> str:
    items = []
    for r in sorted(rows, key=lambda r: r["sample_no"]):
        it = {k: r[k] for k in APP_FIELDS if r.get(k) not in (None, "")}
        it.setdefault("alignment_target", alignment_for_set(r["set"]))
        items.append(it)
    targets = sorted({it["alignment_target"] for it in items} | {"none"})
    data = {
        "sample": str(sample_path),
        "sha256": sample_sha,
        "page": out_name,
        "n": len(items),
        "rubric": rubric_parts(),
        "scales": {t: scale_for(t) for t in targets},
        "items": items,
    }
    return TEMPLATE.replace("/*DATA*/", _embed_json(data))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration labels</title>
<style>
:root {
  --bg: #fbfbf9; --panel: #ffffff; --text: #1d1d1b; --muted: #6a6a64; --line: #deded8;
  --accent: #2557a7; --accent-bg: #e8effa; --warn: #a3470b; --warn-bg: #fdf0e4;
  --ok: #1f7a3d; --sel: #2557a7; --sel-text: #ffffff; --unsure: #8a5a00; --unsure-bg: #fff4d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #171716; --panel: #212120; --text: #e9e9e4; --muted: #a2a29a; --line: #3a3a37;
    --accent: #8db3f0; --accent-bg: #23314a; --warn: #f0a46b; --warn-bg: #3a2717;
    --ok: #7fd39a; --sel: #8db3f0; --sel-text: #10151f; --unsure: #f2c66b; --unsure-bg: #3a2f14;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); }
body { font: 17px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 0 16px 120px; }
header.top { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); }
header.top .wrap { padding-top: 10px; padding-bottom: 10px; display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
.progress { flex: 1 1 220px; min-width: 180px; }
.bar { height: 10px; background: var(--line); border-radius: 5px; overflow: hidden; }
.bar > div { height: 100%; background: var(--ok); width: 0; transition: width .2s; }
.count { font-size: 14px; color: var(--muted); margin-top: 3px; }
.saved { font-size: 14px; color: var(--muted); min-width: 90px; }
.saved.flash { color: var(--ok); }
button, .btn { font: inherit; font-size: 15px; padding: 9px 14px; min-height: 42px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--panel); color: var(--text); cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: var(--sel-text); border-color: var(--accent); }
button:focus-visible, input:focus-visible, textarea:focus-visible, summary:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; }
.nav { display: flex; gap: 8px; align-items: center; justify-content: space-between; margin: 18px 0 8px; }
.nav .mid { color: var(--muted); font-size: 15px; text-align: center; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin: 14px 0; }
.meta { font-size: 15px; color: var(--muted); }
.meta b { color: var(--text); }
h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 8px; }
.question { font-size: 20px; font-weight: 600; }
.zh-small { font-size: 15px; color: var(--muted); margin-top: 4px; }
.answer { white-space: pre-wrap; overflow-wrap: anywhere; }
.gloss-label { display: inline-block; font-size: 13px; font-weight: 700; color: var(--accent); background: var(--accent-bg); padding: 2px 8px; border-radius: 4px; margin-bottom: 8px; }
.warnbox { color: var(--warn); background: var(--warn-bg); padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-weight: 600; }
details > summary { cursor: pointer; color: var(--muted); padding: 6px 0; }
table.facts { width: 100%; border-collapse: collapse; }
table.facts th, table.facts td { border-top: 1px solid var(--line); padding: 8px 6px; vertical-align: top; text-align: left; }
table.facts th { font-size: 13px; color: var(--muted); font-weight: 600; border-top: none; }
table.facts td.idx { width: 2.5em; color: var(--muted); font-variant-numeric: tabular-nums; }
table.facts td.chk { width: 6.5em; text-align: center; }
table.facts td.chk label { display: inline-flex; flex-direction: column; align-items: center; gap: 2px; cursor: pointer; font-size: 12px; color: var(--muted); padding: 4px; }
table.facts input[type=checkbox] { width: 26px; height: 26px; cursor: pointer; }
.zh-fact { font-size: 14px; color: var(--muted); margin-top: 2px; }
.row-contra td { background: var(--warn-bg); }
.toggle { display: flex; align-items: center; gap: 12px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; cursor: pointer; user-select: none; }
.toggle input { width: 26px; height: 26px; cursor: pointer; }
.toggle.on { border-color: var(--accent); background: var(--accent-bg); }
.help { font-size: 14px; color: var(--muted); margin: 6px 0 0; }
.scores { display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 8px; }
.score { display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; border: 2px solid var(--line); border-radius: 10px; cursor: pointer; }
.score input { position: absolute; opacity: 0; pointer-events: none; }
.score .num { flex: 0 0 auto; width: 44px; height: 44px; border-radius: 50%; border: 2px solid var(--line); display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 700; }
.score .desc { font-size: 15px; }
.score .stance { display: block; font-size: 13px; color: var(--muted); margin-bottom: 2px; }
.score.sel { border-color: var(--sel); background: var(--accent-bg); }
.score.sel .num { background: var(--sel); color: var(--sel-text); border-color: var(--sel); }
.score:focus-within { outline: 3px solid var(--accent); outline-offset: 2px; }
textarea { width: 100%; min-height: 70px; font: inherit; font-size: 16px; padding: 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--bg); color: var(--text); }
.toggle.unsure.on { border-color: var(--unsure); background: var(--unsure-bg); }
.status-pill { display: inline-block; font-size: 13px; padding: 2px 10px; border-radius: 999px; border: 1px solid var(--line); margin-left: 8px; }
.status-pill.done { color: var(--ok); border-color: var(--ok); }
.status-pill.unsure { color: var(--unsure); border-color: var(--unsure); }
.grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.grid button { min-width: 44px; min-height: 40px; padding: 4px; font-size: 14px; }
.grid button.done { border-color: var(--ok); color: var(--ok); font-weight: 700; }
.grid button.unsure { border-color: var(--unsure); color: var(--unsure); font-weight: 700; }
.grid button.cur { outline: 3px solid var(--accent); }
.keys { font-size: 13px; color: var(--muted); }
kbd { font: 12px ui-monospace, Menlo, monospace; border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; }
.rubric { white-space: pre-wrap; font-size: 14px; }
.hidden { display: none !important; }
@media (max-width: 640px) {
  body { font-size: 16px; }
  .question { font-size: 18px; }
  .card { padding: 12px; }
  table.facts td.chk { width: 4.6em; }
  .tools button { padding: 8px 10px; font-size: 14px; }
  .keys { display: none; }
}
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <div class="progress">
    <div class="bar"><div id="bar"></div></div>
    <div class="count" id="count"></div>
  </div>
  <div class="saved" id="saved" aria-live="polite"></div>
  <div class="tools">
    <button id="export" class="primary" type="button">Export labels</button>
    <button id="import" type="button">Import labels</button>
    <button id="reset" type="button">Reset item</button>
  </div>
  <input type="file" id="importFile" accept=".jsonl,.json,.txt,application/json" class="hidden">
</div></header>

<main class="wrap">
  <div id="storageWarn" class="warnbox hidden">This browser is not saving to local storage (private window or blocked site data). Your work will be lost on reload: export often.</div>
  <div class="nav">
    <button id="prev" type="button" aria-label="Previous item">&larr; Prev</button>
    <div class="mid"><span id="pos"></span><span id="pill" class="status-pill"></span></div>
    <button id="next" type="button" aria-label="Next item">Next &rarr;</button>
  </div>

  <section class="card">
    <div class="meta" id="meta"></div>
  </section>

  <section class="card">
    <h2>Question</h2>
    <div class="question" id="question"></div>
    <div class="zh-small" id="questionZh"></div>
  </section>

  <section class="card" id="answerCard"></section>

  <section class="card">
    <h2>Reference facts</h2>
    <table class="facts">
      <thead><tr><th>#</th><th>Fact</th><th style="text-align:center">Mentioned</th><th style="text-align:center">Contradicted</th></tr></thead>
      <tbody id="facts"></tbody>
    </table>
    <details><summary>How to judge facts (the judge's instructions)</summary><div class="rubric" id="factsRubric"></div></details>
  </section>

  <section class="card">
    <h2>Refusal</h2>
    <label class="toggle" id="refusalBox"><input type="checkbox" id="refusal"> <span><b>Refusal</b> <span class="keys">(<kbd>r</kbd>)</span></span></label>
    <p class="help" id="refusalHelp"></p>
  </section>

  <section class="card">
    <h2>Framing score <span class="keys">(<kbd>0</kbd> to <kbd>3</kbd>)</span></h2>
    <p class="help" id="scaleIntro"></p>
    <div class="scores" id="scores" role="radiogroup" aria-label="Framing score"></div>
    <p class="help" id="scaleNotes"></p>
    <p class="help" id="framedRule"></p>
  </section>

  <section class="card">
    <h2>Note <span class="keys">(<kbd>n</kbd>)</span></h2>
    <textarea id="note" placeholder="Optional. Why this score, anything odd."></textarea>
    <div style="margin-top:10px">
      <label class="toggle unsure" id="unsureBox"><input type="checkbox" id="unsure"> <span><b>Unsure</b> about this item <span class="keys">(<kbd>u</kbd>)</span></span></label>
      <p class="help">Unsure with no score exports framing_score as null; that item is left out of the framing kappa. Pick a best-guess score as well if you can.</p>
    </div>
  </section>

  <div class="nav">
    <button id="prev2" type="button">&larr; Prev</button>
    <span class="keys"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> move, <kbd>0</kbd>-<kbd>3</kbd> score, <kbd>r</kbd> refusal, <kbd>n</kbd> note, <kbd>u</kbd> unsure, <kbd>Esc</kbd> leave note</span>
    <button id="next2" type="button" class="primary">Next &rarr;</button>
  </div>

  <details class="card"><summary>All items</summary><div class="grid" id="grid"></div></details>
  <p class="help" id="footer"></p>
</main>

<script type="application/json" id="data">/*DATA*/</script>
<script>
"use strict";
(function () {
  var D = JSON.parse(document.getElementById("data").textContent);
  var ITEMS = D.items, N = ITEMS.length;
  var KEY = "tilt-calibrate:" + D.sha256;
  var $ = function (id) { return document.getElementById(id); };
  var storageOk = true;
  var S = load();
  var cur = Math.min(Math.max(S.pos || 0, 0), N - 1);

  function blank() { return { mentioned: [], contradicted: [], refusal: false, score: null, note: "", unsure: false }; }
  function load() {
    try {
      var raw = window.localStorage.getItem(KEY);
      if (raw) { var o = JSON.parse(raw); if (o && o.items) return o; }
      window.localStorage.setItem(KEY + ":probe", "1"); window.localStorage.removeItem(KEY + ":probe");
    } catch (e) { storageOk = false; }
    return { version: 1, sha256: D.sha256, items: {}, pos: 0 };
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
  function st(no) { var k = String(no); if (!S.items[k]) S.items[k] = blank(); return S.items[k]; }
  function peek(no) { return S.items[String(no)] || null; }
  function isDone(s) { return !!s && (s.score !== null && s.score !== undefined); }
  function isUnsure(s) { return !!s && !!s.unsure; }
  function isLabelled(s) { return isDone(s) || isUnsure(s); }

  function text(el, t) { el.textContent = t == null ? "" : t; return el; }
  function mk(tag, cls, t) { var e = document.createElement(tag); if (cls) e.className = cls; if (t != null) e.textContent = t; return e; }

  function render() {
    var it = ITEMS[cur], s = st(it.sample_no), zh = it.lang === "zh";
    var scale = D.scales[it.alignment_target] || D.scales.none;
    text($("pos"), "Item " + it.sample_no + " (" + (cur + 1) + " of " + N + ")");
    var pill = $("pill");
    pill.className = "status-pill" + (isUnsure(s) ? " unsure" : isDone(s) ? " done" : "");
    pill.textContent = isUnsure(s) ? "unsure" : isDone(s) ? "labelled" : "not labelled";

    var meta = $("meta"); meta.innerHTML = "";
    [["Item", it.sample_no], ["Language", it.lang], ["Set", it.set], ["Alignment target", scale.target]].forEach(function (p, i) {
      if (i) meta.appendChild(document.createTextNode("  \u00b7  "));
      meta.appendChild(document.createTextNode(p[0] + ": "));
      meta.appendChild(mk("b", null, String(p[1])));
    });

    if (zh && it.question_en) { text($("question"), it.question_en); text($("questionZh"), it.question); }
    else { text($("question"), it.question); text($("questionZh"), ""); }

    var ac = $("answerCard"); ac.innerHTML = "";
    ac.appendChild(mk("h2", null, "Answer"));
    if (zh) {
      if (it.gloss) {
        ac.appendChild(mk("div", "gloss-label", "English machine gloss: label from this"));
        if (it.gloss_warning) ac.appendChild(mk("div", "warnbox", "The gloss may be incomplete (" + it.gloss_warning + "). Check against the Chinese if you can."));
        ac.appendChild(mk("div", "answer", it.gloss));
      } else {
        ac.appendChild(mk("div", "warnbox", "English gloss MISSING. Run tilt-calibrate gloss, then regenerate this page. Mark this item unsure for now."));
      }
      var det = mk("details"); det.style.marginTop = "12px";
      det.appendChild(mk("summary", null, "Chinese original (for reference)"));
      det.appendChild(mk("div", "answer", it.answer));
      ac.appendChild(det);
    } else {
      ac.appendChild(mk("div", "answer", it.answer));
    }

    var tb = $("facts"); tb.innerHTML = "";
    var factsEn = (zh && it.facts_en && it.facts_en.length === it.facts.length) ? it.facts_en : null;
    if (!it.facts.length) { var tr0 = mk("tr"); var td0 = mk("td", null, "(no reference facts)"); td0.colSpan = 4; tr0.appendChild(td0); tb.appendChild(tr0); }
    it.facts.forEach(function (f, i) {
      var tr = mk("tr"); if (s.contradicted.indexOf(i) >= 0) tr.className = "row-contra";
      tr.appendChild(mk("td", "idx", "[" + i + "]"));
      var td = mk("td");
      td.appendChild(mk("div", null, factsEn ? factsEn[i] : f));
      if (factsEn) td.appendChild(mk("div", "zh-fact", f));
      tr.appendChild(td);
      tr.appendChild(chk(i, "mentioned", s));
      tr.appendChild(chk(i, "contradicted", s));
      tb.appendChild(tr);
    });

    $("refusal").checked = !!s.refusal;
    $("refusalBox").classList.toggle("on", !!s.refusal);

    text($("scaleIntro"), scale.intro);
    var sc = $("scores"); sc.innerHTML = "";
    scale.levels.forEach(function (desc, k) {
      var lab = mk("label", "score" + (s.score === k ? " sel" : ""));
      var inp = mk("input"); inp.type = "radio"; inp.name = "score"; inp.value = String(k); inp.checked = s.score === k;
      inp.addEventListener("change", function () { setScore(k); });
      lab.appendChild(inp);
      lab.appendChild(mk("span", "num", String(k)));
      var dv = mk("span", "desc");
      if (desc.stance) dv.appendChild(mk("span", "stance", desc.stance));
      dv.appendChild(mk("span", null, desc.text));
      lab.appendChild(dv);
      sc.appendChild(lab);
    });
    text($("scaleNotes"), scale.notes);

    if (document.activeElement !== $("note")) $("note").value = s.note || "";
    $("unsure").checked = !!s.unsure;
    $("unsureBox").classList.toggle("on", !!s.unsure);

    $("prev").disabled = $("prev2").disabled = cur === 0;
    $("next").disabled = $("next2").disabled = cur === N - 1;
    progress();
  }

  function chk(i, field, s) {
    var td = mk("td", "chk"), lab = mk("label"), inp = mk("input");
    inp.type = "checkbox"; inp.checked = s[field].indexOf(i) >= 0;
    inp.setAttribute("aria-label", (field === "mentioned" ? "Mentioned" : "Contradicted") + " fact " + i);
    inp.addEventListener("change", function () { setFact(i, field, inp.checked); });
    lab.appendChild(inp); td.appendChild(lab); return td;
  }

  function setIn(arr, i, on) {
    var p = arr.indexOf(i);
    if (on && p < 0) arr.push(i);
    if (!on && p >= 0) arr.splice(p, 1);
    arr.sort(function (a, b) { return a - b; });
  }
  function setFact(i, field, on) {
    var s = st(ITEMS[cur].sample_no);
    setIn(s[field], i, on);
    if (field === "contradicted" && on) setIn(s.mentioned, i, true);   // a contradicted fact is addressed
    if (field === "mentioned" && !on) setIn(s.contradicted, i, false);
    save(); render();
  }
  function setScore(k) { var s = st(ITEMS[cur].sample_no); s.score = k; save(); render(); }
  function toggleRefusal() { var s = st(ITEMS[cur].sample_no); s.refusal = !s.refusal; save(); render(); }
  function toggleUnsure() { var s = st(ITEMS[cur].sample_no); s.unsure = !s.unsure; save(); render(); }
  function go(d) { var n = Math.min(Math.max(cur + d, 0), N - 1); if (n !== cur) { cur = n; save(); render(); window.scrollTo(0, 0); } }
  function goto(i) { cur = i; save(); render(); window.scrollTo(0, 0); }

  function counts() {
    var done = 0, unsure = 0;
    ITEMS.forEach(function (it) { var s = peek(it.sample_no); if (isLabelled(s)) done++; if (isUnsure(s)) unsure++; });
    return { done: done, unsure: unsure };
  }
  function progress() {
    var c = counts();
    $("bar").style.width = (100 * c.done / N) + "%";
    text($("count"), c.done + " of " + N + " labelled" + (c.unsure ? " (" + c.unsure + " unsure)" : ""));
    $("export").textContent = c.done === N ? "Export labels" : "Export partial (" + c.done + " of " + N + ")";
    var g = $("grid"); g.innerHTML = "";
    ITEMS.forEach(function (it, i) {
      var s = peek(it.sample_no), b = mk("button", (isUnsure(s) ? "unsure" : isDone(s) ? "done" : "") + (i === cur ? " cur" : ""), String(it.sample_no));
      b.type = "button"; b.title = it.lang + " " + it.set;
      b.addEventListener("click", function () { goto(i); });
      g.appendChild(b);
    });
  }

  // export in the schema of `tilt-calibrate parse` (+ unsure, source)
  function record(it, s, stamp) {
    var contra = s.contradicted.slice().sort(function (a, b) { return a - b; });
    var ment = s.mentioned.filter(function (i) { return contra.indexOf(i) < 0; }).sort(function (a, b) { return a - b; });
    var score = (s.score === null || s.score === undefined) ? null : s.score;
    var note = (s.note || "").replace(/\s+/g, " ").trim();
    return {
      sample_no: it.sample_no, id: it.id, sample_idx: it.sample_idx, source: it.source,
      lang: it.lang, set: it.set,
      facts_mentioned: ment, facts_contradicted: contra, refusal: !!s.refusal,
      framing_score: score, note: note || null, unsure: !!s.unsure,
      framed: contra.length > 0 || (score !== null && score >= D.rubric.framed_min),
      sheet: D.page, sample_sha256: D.sha256, timestamp: stamp
    };
  }
  function doExport() {
    var c = counts(), missing = N - c.done;
    if (missing > 0 && !window.confirm(missing + " of " + N + " items are not labelled yet and will be left out.\nExport the " + c.done + " labelled items anyway?")) return;
    var stamp = new Date().toISOString().replace(/\.\d+Z$/, "+00:00");
    var lines = [];
    ITEMS.forEach(function (it) { var s = peek(it.sample_no); if (isLabelled(s)) lines.push(JSON.stringify(record(it, s, stamp))); });
    if (!lines.length) { window.alert("Nothing labelled yet."); return; }
    var blob = new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "human_labels.jsonl";
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    text($("saved"), "exported " + lines.length);
  }

  function doImport(file) {
    var r = new FileReader();
    r.onload = function () {
      var byNo = {}; ITEMS.forEach(function (it) { byNo[it.sample_no] = it; });
      var ok = 0, bad = [], other = 0;
      String(r.result).split(/\r?\n/).forEach(function (line, ln) {
        if (!line.trim()) return;
        try {
          var o = JSON.parse(line), it = byNo[o.sample_no];
          if (!it || (o.id && o.id !== it.id) || (o.sample_idx != null && Number(o.sample_idx) !== it.sample_idx)) { bad.push(ln + 1); return; }
          if (o.sample_sha256 && o.sample_sha256 !== D.sha256) other++;
          var contra = (o.facts_contradicted || []).map(Number);
          var ment = (o.facts_mentioned || []).map(Number).concat(contra);
          var sc = o.framing_score === null || o.framing_score === undefined || o.framing_score === "" ? null : Number(o.framing_score);
          S.items[String(it.sample_no)] = {
            mentioned: ment.filter(function (v, i, a) { return a.indexOf(v) === i; }).sort(function (a, b) { return a - b; }),
            contradicted: contra.sort(function (a, b) { return a - b; }),
            refusal: o.refusal === true || o.refusal === "y" || o.refusal === "yes",
            score: sc, note: o.note || "", unsure: !!o.unsure
          };
          ok++;
        } catch (e) { bad.push(ln + 1); }
      });
      save(); render();
      window.alert("Imported " + ok + " item(s)." + (bad.length ? "\nSkipped line(s) " + bad.join(", ") + " (unreadable or not in this sample)." : "") + (other ? "\n" + other + " line(s) came from a different sample file; matched by item number and id." : ""));
    };
    r.readAsText(file);
  }

  // wiring
  $("prev").onclick = $("prev2").onclick = function () { go(-1); };
  $("next").onclick = $("next2").onclick = function () { go(1); };
  $("refusal").addEventListener("change", function () { toggleRefusal(); });
  $("unsure").addEventListener("change", function () { toggleUnsure(); });
  $("note").addEventListener("input", function () { st(ITEMS[cur].sample_no).note = $("note").value; save(); });
  $("export").onclick = doExport;
  $("import").onclick = function () { $("importFile").value = ""; $("importFile").click(); };
  $("importFile").addEventListener("change", function () { if (this.files && this.files[0]) doImport(this.files[0]); });
  $("reset").onclick = function () {
    var no = ITEMS[cur].sample_no;
    if (window.confirm("Clear all labels on item " + no + "?")) { delete S.items[String(no)]; save(); $("note").value = ""; render(); }
  };
  document.addEventListener("keydown", function (e) {
    var t = e.target, typing = t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "checkbox" && t.type !== "radio"));
    if (typing) { if (e.key === "Escape") t.blur(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowLeft") { go(-1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { go(1); e.preventDefault(); }
    else if (/^[0-3]$/.test(e.key)) { setScore(Number(e.key)); e.preventDefault(); }
    else if (e.key === "r" || e.key === "R") { toggleRefusal(); e.preventDefault(); }
    else if (e.key === "u" || e.key === "U") { toggleUnsure(); e.preventDefault(); }
    else if (e.key === "n" || e.key === "N") { $("note").focus(); e.preventDefault(); }
  });

  text($("factsRubric"), D.rubric.facts + "\n\nFact numbers are 0-based, as the judge sees them. Ticking Contradicted also ticks Mentioned; the export lists a contradicted fact under facts_contradicted only, as the judge does.");
  text($("refusalHelp"), D.rubric.refusal);
  text($("framedRule"), "A response counts as framed if any fact is contradicted or the framing score is " + D.rubric.framed_min + " or more. Rubric " + D.rubric.version + ".");
  text($("footer"), "Sample " + D.sample + ", sha256 " + D.sha256.slice(0, 12) + ". Labels autosave in this browser only (localStorage); export to keep them.");
  if (!storageOk) $("storageWarn").classList.remove("hidden");
  render();
})();
</script>
</body>
</html>
"""
