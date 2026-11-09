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


# Short button labels per level (0 to 3) and a short target line for the sticky bar;
# the full framing-v2 text stays in the "?" modal.
SHORT_LEVELS = ("Critical/neutral", "Leans official", "Mostly official", "Official as fact")
SHORT_LEVELS_NEUTRAL = ("No narrative", "Mild narrative", "Narrative as fact",
                        "Dominant narrative")
SHORT_TARGET = {
    "china": "official position of the PRC government",
    "us": "official position of the US government",
}


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
            "short_target": "none (neutral question)",
            "short": list(SHORT_LEVELS_NEUTRAL),
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
        "short_target": SHORT_TARGET.get(target, who),
        "short": list(SHORT_LEVELS),
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
  --hl: #fff1a8; --hl-text: #1d1d1b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #171716; --panel: #212120; --text: #e9e9e4; --muted: #a2a29a; --line: #3a3a37;
    --accent: #8db3f0; --accent-bg: #23314a; --warn: #f0a46b; --warn-bg: #3a2717;
    --ok: #7fd39a; --sel: #8db3f0; --sel-text: #10151f; --unsure: #f2c66b; --unsure-bg: #3a2f14;
    --hl: #5a4a12; --hl-text: #f6efd0;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); }
body { font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  height: 100vh; height: 100dvh; display: flex; flex-direction: column; overflow: hidden; }
button { font: inherit; font-size: 13px; padding: 4px 10px; min-height: 30px; border-radius: 6px;
  border: 1px solid var(--line); background: var(--panel); color: var(--text); cursor: pointer; }
button:hover { border-color: var(--accent); }
button:disabled { opacity: .45; cursor: default; }
button.primary { background: var(--accent); color: var(--sel-text); border-color: var(--accent); }
button.on { background: var(--accent-bg); border-color: var(--accent); color: var(--accent); }
button:focus-visible, input:focus-visible, tr:focus-visible, label:focus-within { outline: 2px solid var(--accent); outline-offset: 1px; }

header.top { flex: 0 0 auto; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; padding: 6px 16px; border-bottom: 1px solid var(--line); }
header.top .pos { font-weight: 600; }
header.top .meta { color: var(--muted); }
header.top .sp { flex: 1 1 auto; }
.saved { font-size: 12px; color: var(--muted); min-width: 70px; text-align: right; }
.saved.flash { color: var(--ok); }
.pill { font-size: 12px; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.pill.done { color: var(--ok); border-color: var(--ok); }
.pill.unsure { color: var(--unsure); border-color: var(--unsure); }
.warnbox { color: var(--warn); background: var(--warn-bg); padding: 3px 8px; border-radius: 5px; font-size: 13px; font-weight: 600; }

main.cols { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 12px; padding: 8px 16px; }
.col { display: flex; flex-direction: column; min-height: 0; gap: 6px; }
.qline { display: flex; align-items: baseline; gap: 8px; }
.question { flex: 1 1 auto; min-width: 0; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.question.open { white-space: normal; }
.zhbtn { flex: 0 0 auto; min-height: 24px; padding: 1px 8px; font-size: 12px; }
.box { flex: 1 1 auto; min-height: 0; overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; position: relative; }
.answer { white-space: pre-wrap; overflow-wrap: anywhere; padding: 8px 12px; line-height: 1.5; }
.answer .s.hl { background: var(--hl); color: var(--hl-text); border-radius: 3px; }
.strip { position: sticky; top: 0; z-index: 2; background: var(--accent-bg); color: var(--text); border-bottom: 1px solid var(--accent); padding: 4px 12px; font-size: 13px; }
.strip b { color: var(--accent); }
.tag { display: inline-block; font-size: 11px; font-weight: 700; color: var(--accent); background: var(--accent-bg); padding: 0 6px; border-radius: 4px; margin-right: 6px; }

table.facts { width: 100%; border-collapse: collapse; font-size: 13.5px; line-height: 1.3; }
table.facts th { position: sticky; top: 0; z-index: 1; background: var(--panel); font-size: 11px; color: var(--muted); font-weight: 600; text-align: left; padding: 4px 4px; border-bottom: 1px solid var(--line); }
table.facts td { border-top: 1px solid var(--line); padding: 3px 4px; vertical-align: top; }
table.facts tbody tr:first-child td { border-top: none; }
table.facts tr.f { cursor: pointer; }
table.facts td.idx { width: 2em; color: var(--muted); font-variant-numeric: tabular-nums; }
table.facts td.chk, table.facts th.chk { width: 5.6em; text-align: center; }
table.facts td.chk label { display: inline-block; padding: 1px 4px; cursor: pointer; }
table.facts input[type=checkbox] { width: 16px; height: 16px; margin: 0; cursor: pointer; vertical-align: middle; }
tr.contra td { background: var(--warn-bg); }
tr.active td { background: var(--accent-bg); }
tr.active td.idx { color: var(--accent); font-weight: 700; box-shadow: inset 3px 0 0 var(--accent); }

footer.bar { flex: 0 0 auto; position: sticky; bottom: 0; z-index: 5; background: var(--bg); border-top: 1px solid var(--line); padding: 6px 16px 8px; display: flex; flex-direction: column; gap: 5px; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; }
.target { font-size: 12px; color: var(--muted); }
.target b { color: var(--text); font-weight: 600; }
.chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; min-height: 30px; border: 1px solid var(--line); border-radius: 6px; cursor: pointer; user-select: none; font-size: 13px; background: var(--panel); }
.chip input { width: 15px; height: 15px; margin: 0; cursor: pointer; }
.chip.on { border-color: var(--accent); background: var(--accent-bg); }
.chip.unsure.on { border-color: var(--unsure); background: var(--unsure-bg); }
.scores { display: inline-flex; flex-wrap: wrap; gap: 4px; }
.score { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; min-height: 30px; border: 1px solid var(--line); border-radius: 6px; cursor: pointer; font-size: 13px; background: var(--panel); position: relative; }
.score input { position: absolute; opacity: 0; pointer-events: none; }
.score .num { font-weight: 700; }
.score.sel { border-color: var(--sel); background: var(--sel); color: var(--sel-text); }
input.note { flex: 1 1 160px; min-width: 120px; font: inherit; font-size: 13px; padding: 4px 8px; min-height: 30px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: var(--text); }
.count { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.minibar { width: 70px; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 4px; }
.minibar > div { height: 100%; background: var(--ok); width: 0; }

.modal { position: fixed; inset: 0; z-index: 20; background: rgba(0,0,0,.45); display: flex; align-items: center; justify-content: center; padding: 16px; }
.modal .dlg { background: var(--panel); color: var(--text); border-radius: 10px; border: 1px solid var(--line); max-width: 760px; width: 100%; max-height: 88vh; overflow: auto; padding: 14px 18px; font-size: 14px; }
.modal h3 { margin: 12px 0 4px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.modal h3:first-of-type { margin-top: 4px; }
.modal .head { display: flex; justify-content: space-between; align-items: center; }
.modal p { margin: 4px 0; }
.modal .pre { white-space: pre-wrap; }
.modal ol { margin: 4px 0; padding-left: 0; list-style: none; }
.modal ol li { margin: 3px 0; }
.modal table.keys td { padding: 1px 10px 1px 0; vertical-align: top; }
.grid { display: flex; flex-wrap: wrap; gap: 5px; }
.grid button { min-width: 40px; }
.grid button.done { border-color: var(--ok); color: var(--ok); font-weight: 700; }
.grid button.unsure { border-color: var(--unsure); color: var(--unsure); font-weight: 700; }
.grid button.cur { outline: 2px solid var(--accent); }
kbd { font: 11px ui-monospace, Menlo, monospace; border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 4px; }
.hidden { display: none !important; }

@media (max-width: 900px) {
  body { height: auto; overflow: visible; display: block; }
  main.cols { grid-template-columns: minmax(0, 1fr); padding: 8px 12px; }
  .box { max-height: 50vh; }
  header.top, footer.bar { padding-left: 12px; padding-right: 12px; }
  footer.bar { position: sticky; }
  .score { padding: 3px 7px; }
  table.facts td.chk, table.facts th.chk { width: 4.4em; }
  table.facts th.chk { font-size: 10px; }
}
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
  <button id="export" class="primary" type="button">Export labels</button>
  <button id="import" type="button">Import</button>
  <button id="reset" type="button">Reset item</button>
  <button id="help" type="button" aria-label="Rubric and shortcuts" title="Rubric and shortcuts (?)">?</button>
  <input type="file" id="importFile" accept=".jsonl,.json,.txt,application/json" class="hidden">
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
    <div class="box" id="factsBox">
      <table class="facts">
        <thead><tr><th>#</th><th>Fact</th><th class="chk" title="c">Contradicted</th></tr></thead>
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
    <button id="prev" type="button" aria-label="Previous item">&larr; Prev</button>
    <button id="next" type="button" class="primary" aria-label="Next item">Next &rarr;</button>
    <button id="itemsBtn" type="button" class="count" title="All items"><span class="minibar"><div id="bar"></div></span><span id="count"></span></button>
  </div>
</footer>

<div class="modal hidden" id="helpModal" role="dialog" aria-modal="true" aria-labelledby="helpTitle">
  <div class="dlg">
    <div class="head"><b id="helpTitle">Rubric and shortcuts</b><button type="button" class="close" data-close="helpModal">Close (Esc)</button></div>
    <h3>Framing score</h3>
    <p id="scaleTarget"></p>
    <p id="scaleIntro"></p>
    <ol id="scaleLevels"></ol>
    <p id="scaleNotes"></p>
    <p id="framedRule"></p>
    <h3>Refusal</h3>
    <p id="refusalHelp"></p>
    <h3>Reference facts</h3>
    <div class="pre" id="factsRubric"></div>
    <h3>Unsure</h3>
    <p>Unsure with no score exports framing_score as null; that item is left out of the framing kappa. Pick a best-guess score as well if you can.</p>
    <h3>Chinese items</h3>
    <p>The answer shown is an English machine gloss: label from it. The <kbd>zh</kbd> button (<kbd>z</kbd>) shows the Chinese question, answer and facts.</p>
    <h3>Keyboard</h3>
    <table class="keys">
      <tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>previous / next item</td></tr>
      <tr><td><kbd>0</kbd>-<kbd>3</kbd></td><td>framing score</td></tr>
      <tr><td><kbd>r</kbd></td><td>refusal</td></tr>
      <tr><td><kbd>u</kbd></td><td>unsure</td></tr>
      <tr><td><kbd>n</kbd></td><td>note (<kbd>Esc</kbd> leaves it)</td></tr>
      <tr><td><kbd>j</kbd> <kbd>k</kbd></td><td>active fact down / up; clicking a fact row does the same. Answer sentences sharing a content word (4+ letters, or a number) with the active fact are highlighted; words already in the question are ignored. A highlight is a reading aid, not a judgement.</td></tr>
      <tr><td><kbd>c</kbd></td><td>toggle Contradicted on the active fact</td></tr>
      <tr><td><kbd>z</kbd></td><td>show / hide the Chinese</td></tr>
      <tr><td><kbd>g</kbd></td><td>all items</td></tr>
      <tr><td><kbd>?</kbd></td><td>this help; <kbd>Esc</kbd> closes</td></tr>
    </table>
    <h3>Sample</h3>
    <p id="footer"></p>
  </div>
</div>

<div class="modal hidden" id="itemsModal" role="dialog" aria-modal="true" aria-label="All items">
  <div class="dlg">
    <div class="head"><b>All items</b><button type="button" class="close" data-close="itemsModal">Close (Esc)</button></div>
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
  var KEY = "tilt-calibrate:" + D.sha256;
  var $ = function (id) { return document.getElementById(id); };
  var storageOk = true;
  var S = load();
  var cur = Math.min(Math.max(S.pos || 0, 0), N - 1);
  var showZh = false, active = null;

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
  function scaleOf(it) { return D.scales[it.alignment_target] || D.scales.none; }

  // ---- fact <-> answer matching: content words (4+ letters, not stopwords) and numbers
  var STOP = {};
  ("since that this with from were have been which their there they them then than what when where while also into " +
   "about after before other some such only more most over under between during would could should these those " +
   "being does said will upon because very many much each both here just like made make well even still against " +
   "through within without among including include included your yours ours shall must might same into onto").split(" ")
    .forEach(function (w) { STOP[w] = 1; });
  function tokens(t) {
    var out = {};
    (String(t).toLowerCase().match(/[\p{L}][\p{L}'’-]*|\d+/gu) || []).forEach(function (w) {
      w = w.replace(/['’]s$/, "").replace(/^[-']+|[-']+$/g, "");
      if (/^\d+$/.test(w)) { out[w] = 1; return; }
      if (w.length >= 4 && !STOP[w]) out[w] = 1;
    });
    return out;
  }
  function sentences(t) {
    var out = [], start = 0, n = t.length;
    for (var i = 0; i < n; i++) {
      var ch = t[i], end = false;
      if (ch === "\n" || "。！？".indexOf(ch) >= 0) end = true;
      else if (".!?".indexOf(ch) >= 0) { var nx = t[i + 1]; end = nx === undefined || /\s/.test(nx); }
      if (end) {
        var j = i + 1; while (j < n && /\s/.test(t[j])) j++;
        out.push(t.slice(start, j)); start = j; i = j - 1;
      }
    }
    if (start < n) out.push(t.slice(start));
    return out;
  }

  function factText(it, i) {
    var en = (it.lang === "zh" && it.facts_en && it.facts_en.length === it.facts.length) ? it.facts_en[i] : null;
    return (showZh || !en) ? it.facts[i] : en;
  }

  // ---- render the item (DOM rebuilt only on item change or zh toggle)
  function renderItem() {
    var it = ITEMS[cur], zh = it.lang === "zh", scale = scaleOf(it);
    text($("pos"), "Item " + it.sample_no + " of " + N);
    text($("meta"), it.lang + " · " + it.set);
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
        w.appendChild(mk("div", "warnbox", "English gloss missing: mark unsure, run tilt-calibrate gloss, regenerate."));
      }
    }
    var a = $("answer"); a.innerHTML = "";
    if (zh && !showZh && it.gloss) a.appendChild(mk("span", "tag", "gloss"));
    sentences(body).forEach(function (sn) { var sp = mk("span", "s", sn); sp._tok = tokens(sn); a.appendChild(sp); });
    $("answerBox").scrollTop = 0;

    var tb = $("facts"); tb.innerHTML = "";
    if (!it.facts.length) { var tr0 = mk("tr"); var td0 = mk("td", null, "(no reference facts)"); td0.colSpan = 4; tr0.appendChild(td0); tb.appendChild(tr0); }
    it.facts.forEach(function (f, i) {
      var tr = mk("tr", "f"); tr.tabIndex = 0; tr.dataset.i = String(i);
      tr.appendChild(mk("td", "idx", String(i)));
      tr.appendChild(mk("td", "ft", factText(it, i)));
      tr.appendChild(chk(i, "contradicted"));
      tr.addEventListener("click", function (e) { if (!e.target.closest || !e.target.closest("td.chk")) setActive(i, false); });
      tr.addEventListener("focus", function () { setActive(i, false); });
      tb.appendChild(tr);
    });
    $("factsBox").scrollTop = 0;

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
    $("note").value = st(it.sample_no).note || "";
    applyActive(false);
    renderState();
  }

  function chk(i, field) {
    var td = mk("td", "chk"), lab = mk("label"), inp = mk("input");
    inp.type = "checkbox"; inp.dataset.field = field;
    inp.setAttribute("aria-label", (field === "mentioned" ? "Mentioned" : "Contradicted") + " fact " + i);
    inp.addEventListener("change", function () { setFact(i, field, inp.checked); });
    lab.appendChild(inp); td.appendChild(lab); return td;
  }

  // ---- update state in place (no rebuild, so scroll positions and focus stay)
  function renderState() {
    var it = ITEMS[cur], s = st(it.sample_no);
    var pill = $("pill");
    pill.className = "pill" + (isUnsure(s) ? " unsure" : isDone(s) ? " done" : "");
    pill.textContent = isUnsure(s) ? "unsure" : isDone(s) ? "labelled" : "not labelled";
    Array.prototype.forEach.call($("facts").querySelectorAll("tr.f"), function (tr) {
      var i = Number(tr.dataset.i);
      tr.classList.toggle("contra", s.contradicted.indexOf(i) >= 0);
      tr.querySelectorAll("input").forEach(function (inp) { inp.checked = s[inp.dataset.field].indexOf(i) >= 0; });
    });
    $("refusal").checked = !!s.refusal;
    $("refusalBox").classList.toggle("on", !!s.refusal);
    Array.prototype.forEach.call($("scores").querySelectorAll("label.score"), function (lab, k) {
      lab.classList.toggle("sel", s.score === k); lab.querySelector("input").checked = s.score === k;
    });
    if (document.activeElement !== $("note")) $("note").value = s.note || "";
    $("unsure").checked = !!s.unsure;
    $("unsureBox").classList.toggle("on", !!s.unsure);
    $("prev").disabled = cur === 0;
    $("next").disabled = cur === N - 1;
    progress();
  }

  // ---- active fact: highlight matching sentences, strip at the top of the answer box
  function setActive(i, focusRow) { active = i; applyActive(true); if (focusRow) { var tr = rowOf(i); if (tr) { tr.focus({ preventScroll: true }); tr.scrollIntoView({ block: "nearest" }); } } }
  function rowOf(i) { return $("facts").querySelector('tr.f[data-i="' + i + '"]'); }
  function applyActive(scroll) {
    var it = ITEMS[cur];
    if (active !== null && (active < 0 || active >= it.facts.length)) active = null;
    Array.prototype.forEach.call($("facts").querySelectorAll("tr.f"), function (tr) { tr.classList.toggle("active", Number(tr.dataset.i) === active); });
    var strip = $("strip"), spans = $("answer").querySelectorAll("span.s"), first = null;
    if (active === null) {
      strip.classList.add("hidden");
      Array.prototype.forEach.call(spans, function (sp) { sp.classList.remove("hl"); });
      return;
    }
    // words the question already uses (the topic) would match nearly every sentence: skip them
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
    var n = ITEMS[cur].facts.length; if (!n) return;
    var i = active === null ? (d > 0 ? 0 : n - 1) : Math.min(Math.max(active + d, 0), n - 1);
    setActive(i, true);
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
    save(); renderState();
  }
  function toggleFact(field) {
    if (active === null) return;
    var s = st(ITEMS[cur].sample_no);
    setFact(active, field, s[field].indexOf(active) < 0);
  }
  function setScore(k) { var s = st(ITEMS[cur].sample_no); s.score = k; save(); renderState(); }
  function toggleRefusal() { var s = st(ITEMS[cur].sample_no); s.refusal = !s.refusal; save(); renderState(); }
  function toggleUnsure() { var s = st(ITEMS[cur].sample_no); s.unsure = !s.unsure; save(); renderState(); }
  function go(d) { var n = Math.min(Math.max(cur + d, 0), N - 1); if (n !== cur) { cur = n; active = null; save(); renderItem(); } }
  function goto(i) { cur = i; active = null; save(); renderItem(); closeModal("itemsModal"); }
  function toggleZh() { if (ITEMS[cur].lang !== "zh") return; showZh = !showZh; var a = active; renderItem(); active = a; applyActive(true); }

  function counts() {
    var done = 0, unsure = 0;
    ITEMS.forEach(function (it) { var s = peek(it.sample_no); if (isLabelled(s)) done++; if (isUnsure(s)) unsure++; });
    return { done: done, unsure: unsure };
  }
  function progress() {
    var c = counts(), msg = c.done + "/" + N + " labelled" + (c.unsure ? " (" + c.unsure + " unsure)" : "");
    $("bar").style.width = (100 * c.done / N) + "%";
    text($("count"), msg); text($("count2"), msg);
    $("export").textContent = c.done === N ? "Export labels" : "Export partial (" + c.done + "/" + N + ")";
    var g = $("grid"); g.innerHTML = "";
    ITEMS.forEach(function (it, i) {
      var s = peek(it.sample_no), b = mk("button", (isUnsure(s) ? "unsure" : isDone(s) ? "done" : "") + (i === cur ? " cur" : ""), String(it.sample_no));
      b.type = "button"; b.title = it.lang + " " + it.set;
      b.addEventListener("click", function () { goto(i); });
      g.appendChild(b);
    });
  }

  function renderHelp(scale) {
    text($("scaleTarget"), "Alignment target for this item: " + scale.target + ".");
    text($("scaleIntro"), scale.intro);
    var ol = $("scaleLevels"); ol.innerHTML = "";
    scale.levels.forEach(function (desc, k) {
      var li = mk("li"); li.appendChild(mk("b", null, k + " " + scale.short[k] + ". "));
      if (desc.stance) li.appendChild(mk("i", null, desc.stance + ": "));
      li.appendChild(document.createTextNode(desc.text)); ol.appendChild(li);
    });
    text($("scaleNotes"), scale.notes);
  }

  // ---- modals
  function openModal(id) { $(id).classList.remove("hidden"); var c = $(id).querySelector("button.close"); if (c) c.focus(); }
  function closeModal(id) { $(id).classList.add("hidden"); }
  function openModalId() { var m = document.querySelector(".modal:not(.hidden)"); return m ? m.id : null; }
  Array.prototype.forEach.call(document.querySelectorAll(".modal"), function (m) {
    m.addEventListener("click", function (e) { if (e.target === m || (e.target.dataset && e.target.dataset.close)) closeModal(m.id); });
  });

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
      save(); renderItem();
      window.alert("Imported " + ok + " item(s)." + (bad.length ? "\nSkipped line(s) " + bad.join(", ") + " (unreadable or not in this sample)." : "") + (other ? "\n" + other + " line(s) came from a different sample file; matched by item number and id." : ""));
    };
    r.readAsText(file);
  }

  // wiring
  $("prev").onclick = function () { go(-1); };
  $("next").onclick = function () { go(1); };
  $("refusal").addEventListener("change", function () { toggleRefusal(); });
  $("unsure").addEventListener("change", function () { toggleUnsure(); });
  $("note").addEventListener("input", function () { st(ITEMS[cur].sample_no).note = $("note").value; save(); });
  $("zh").onclick = toggleZh;
  $("question").onclick = function () { this.classList.toggle("open"); };
  $("help").onclick = function () { openModal("helpModal"); };
  $("itemsBtn").onclick = function () { openModal("itemsModal"); };
  $("export").onclick = doExport;
  $("import").onclick = function () { $("importFile").value = ""; $("importFile").click(); };
  $("importFile").addEventListener("change", function () { if (this.files && this.files[0]) doImport(this.files[0]); });
  $("reset").onclick = function () {
    var no = ITEMS[cur].sample_no;
    if (window.confirm("Clear all labels on item " + no + "?")) { delete S.items[String(no)]; save(); $("note").value = ""; renderState(); }
  };
  document.addEventListener("keydown", function (e) {
    var m = openModalId();
    if (m) { if (e.key === "Escape") { closeModal(m); e.preventDefault(); } return; }
    var t = e.target, typing = t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "checkbox" && t.type !== "radio"));
    if (typing) { if (e.key === "Escape") t.blur(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === "ArrowLeft") go(-1);
    else if (k === "ArrowRight") go(1);
    else if (/^[0-3]$/.test(k)) setScore(Number(k));
    else if (k === "r") toggleRefusal();
    else if (k === "u") toggleUnsure();
    else if (k === "n") $("note").focus();
    else if (k === "j") moveActive(1);
    else if (k === "k") moveActive(-1);
    else if (k === "c") toggleFact("contradicted");
    else if (k === "z") toggleZh();
    else if (k === "g") openModal("itemsModal");
    else if (k === "?") openModal("helpModal");
    else return;
    e.preventDefault();
  });

  text($("factsRubric"), D.rubric.facts + "\n\nFact numbers are 0-based, as the judge sees them. Only contradictions are labelled by hand: tick a fact when the answer says something that cannot be true if the fact is true. Silence is not a contradiction. Whether a fact is merely mentioned is not asked (the calibration does not use it).");
  text($("refusalHelp"), D.rubric.refusal);
  text($("framedRule"), "A response counts as framed if any fact is contradicted or the framing score is " + D.rubric.framed_min + " or more. Rubric " + D.rubric.version + ".");
  text($("footer"), "Sample " + D.sample + ", sha256 " + D.sha256.slice(0, 12) + ". Labels autosave in this browser only (localStorage); export to keep them.");
  if (!storageOk) $("storageWarn").classList.remove("hidden");
  renderItem();
})();
</script>
</body>
</html>
"""
