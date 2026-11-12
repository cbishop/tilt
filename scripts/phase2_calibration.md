# Phase 2 checklist: baselines and judge calibration

Purpose: fix the denominator (the known set) and find out how far the judges can be trusted before any Phase 3 training. About 6 attended hours, small machine time. Design and rationale: [docs/design.md](../docs/design.md#phase-2-baselines-and-judge-calibration-about-6-hours).

## 2.1 Freeze the known sets

- [x] Record Gemma 3 4B's known set per language in `results/known_set_frozen.json` (2026-11-09): ids per set, counts, the source files and their sha256, rubric `recall-v1` (Haiku 4.5, at least 2 of 3 samples good). English: testbed 71, US-sensitive 28, neutral 49 (148). Chinese: testbed 73, US-sensitive 17, neutral 47 (137; after the zh-v2 re-translation of 23 records). Re-frozen 2026-11-10 after the translation review removed tb-test-077: Chinese testbed 72 (136)
- [x] Top up from the testbed dev split if a testbed known set is under 60: not needed (71 en, 73 zh), recorded in the file

  ```bash
  uv run tilt-calibrate freeze --student mlx-community/gemma-3-4b-it-4bit --date 2026-11-09
  ```

  The frozen file is a record, not a new input: `tilt-judge --compare` and `tilt-evaluate --known-set` still read `results/known_set_gemma-3-4b-it-4bit_<lang>.jsonl`. If those files change later, their sha256 no longer matches the record.

## 2.2 Translation spot check

- [x] Back-translation audit of 20 Chinese questions and their facts: `results/translation_audit_zh.jsonl` (rows) and `results/translation_audit_zh.md` (readable report), produced by `scripts/translation_audit.py`
- [x] Human review: read every item flagged `meaning_changed` in the report against the Chinese and the English source; record per item whether the forward translation is wrong (fix, re-judge the affected question) or the back-translation is (no action) (2026-11-10: 23 flags reviewed against the current Chinese by Claude Opus 5.5 and spot-checked; 4 real fact errors fixed, plus the Li Zhanshu name in tb-test-077; fixes approved by the rater from the English description)
- [x] Note the outcome and any fixed questions in `RESULTS.md` (Phase 2 section). If Chinese question or fact text changes, every Chinese judge file judged against the old text is stale for those questions; re-judge them before step 2.3 so the labelling sheet and the judge see the same facts (2026-11-10: recorded; `framing-v3` labels for the four fixed known-set questions re-judged by both judges, 52 answers each; audit re-run on the corrected files, 4 fact flags left, all artefacts; tb-test-077 fails `recall-v1` under the corrected facts and was removed from the Chinese known set, which was re-frozen)

## 2.3 Hand-label 60 responses and compute kappa

- [x] Draw the sample: 60 answers, half English and half Chinese, 10 per language from each of teacher recall, base student eval and tuned student eval, 7 testbed, 2 US-sensitive and 1 neutral per cell; only known-set questions, and only answers with a Sonnet 5.5 `framing-v2` label
- [x] Gloss the Chinese answers: a literal, complete machine English gloss of each Chinese answer (Claude Haiku 4.5, plain text), written back into the sample rows as `gloss`, `gloss_model`, `gloss_usage`; a gloss still too short after one retry is flagged `gloss_warning`
- [x] Write the labelling app and label all 60 items in it: `runs/calibration/label.html`, one self-contained page opened from disk (no network). One item per screen: facts table (the targeted review set by default) with a Contradicted tick, Refusal, framing score 0 to 3 with the `framing-v2` scale text for the item's alignment target, a note and an Unsure flag. Keys: left/right move, `0` to `3` score, `r` refusal, `n` note, `u` unsure. The page hides the source model and the judge's labels. Chinese items show the English question (Chinese under it), the English machine gloss open (label from this) with the Chinese answer collapsed, then the English facts with the Chinese fact under each
- [x] Export labels from the app to `runs/calibration/human_labels.jsonl` (the browser saves it to Downloads; move it there)
- [x] Fallback only, if the app cannot be used: render the Markdown sheet, fill in each `labels` block and parse it back to JSONL (same output schema). Not needed: the app was used
- [x] Hand labels cover contradictions per fact, refusal and the framing score only. "Mentioned" is not hand-labelled: no Phase 2 metric uses it, and the known set is frozen from the Haiku recall labels.
- [x] Contradictions are labelled by targeted review (the app's default, `--review-facts`): reading every reference fact for 60 items (913 facts) is not feasible, so each item shows only a review set, the union of the facts Sonnet or Mistral marked contradicted plus 2 unflagged decoy facts (`--decoys`), drawn with a fixed seed per item and shuffled in with them. The page never shows which facts were flagged, by which judge, or any judge label; a note above the table says "Review set: tick any fact the answer contradicts. Facts not shown were not reviewed." Items without a Mistral label (file missing, or the answer not yet judged) use Sonnet only. `--all-facts` shows every fact instead
- [x] Kappa, human vs Sonnet 5.5 (and vs Mistral after 2.4), per language and pooled: `framed`, `any_contradiction`, `refusal` (unweighted) and `framing_score` (quadratic weights), each with a 95% bootstrap CI over items

  ```bash
  uv run tilt-calibrate sample --n 60 --seed 0 --out runs/calibration/sample.jsonl
  uv run tilt-calibrate gloss --sample runs/calibration/sample.jsonl
  uv run tilt-calibrate app --sample runs/calibration/sample.jsonl --out runs/calibration/label.html
  # (regenerate after the Mistral judge files are complete, before labelling starts)
  open runs/calibration/label.html
  # ... label in the browser, Export labels, then:
  mv ~/Downloads/human_labels.jsonl runs/calibration/human_labels.jsonl

  # fallback: the Markdown sheet
  uv run tilt-calibrate sheet --sample runs/calibration/sample.jsonl --out runs/calibration/sheet.md
  uv run tilt-calibrate parse --sheet runs/calibration/sheet.md --out runs/calibration/human_labels.jsonl

  uv run tilt-calibrate kappa --human runs/calibration/human_labels.jsonl --sample runs/calibration/sample.jsonl
  ```

  `sample` is deterministic for a seed and prints the counts per stratum. The app autosaves every change to the browser's localStorage under the sample file's sha256, so closing and reopening `label.html` in the same browser restores the work; regenerating the page from the same sample keeps it, a new sample starts empty. Storage is per browser, so export now and then as a backup; "Import labels" loads an exported file back (items matched by number and id). "Export labels" writes one line per labelled item in the `parse` schema plus `unsure` and `source`; with items still open it becomes "Export partial" and asks first. An item counts as labelled once it has a framing score or is marked unsure. Ticking Contradicted also ticks Mentioned; the export lists a contradicted fact under `facts_contradicted` only, as the judge does. An unsure item with no score exports `framing_score: null`: `kappa` keeps unsure items, but leaves a null score out of `framing_score` and (unless a fact is contradicted) `framed`, and prints which items those are. `sheet` refuses to overwrite an existing sheet (it may hold labels) unless given `--force`. `parse` accepts `y/yes/n/no`, commas or spaces between fact numbers, and blank or `none` for no facts; it lists any malformed item and writes nothing until they are fixed, and reports items still blank. `kappa` finds the judge files from each item's `judge_ref` plus any `framing_<stem>.mistral.jsonl` beside it (more with `--judge-files`), prints the table and upserts one line per comparison x metric x language into `results/judge_calibration.jsonl`; `--dry-run` prints only. In targeted review each exported line also carries `facts_reviewed` (the shown fact numbers, sorted), `review_mode: "targeted"` and `review_judges` (`["sonnet","mistral"]`, or `["sonnet"]` where no Mistral label existed); `facts_contradicted` is the ticked facts. `app` prints how many facts it shows (flagged vs decoys) and how many items had a Mistral label. When the human labels carry `facts_reviewed`, `kappa` compares human and judge on `any_contradiction` and `framed` over the reviewed facts only: each judge's `facts_contradicted` is cut to the item's `facts_reviewed` before either is derived, so a judge contradiction outside the review set counts as none. It prints the number of reviewed facts against the total and writes `review_mode`, `n_facts_reviewed` and `n_facts_total` into those `results/judge_calibration.jsonl` rows; judge vs judge rows stay on every fact (`review_mode: "full"`). Refusal and framing score are whole-answer labels and are unchanged. Generate the page once the Mistral files are complete and do not regenerate it mid-labelling: the review set depends on the judge files present at generation time (the autosave key is the sample's sha256, so saved ticks survive a regeneration, but a fact that leaves the review set is no longer shown or exported). `gloss` only touches Chinese rows without a gloss, so a rerun resumes; it rewrites the sample file, so write the app (or sheet) after it (both warn if any Chinese item has no gloss). 30 glosses cost about \$0.08.

- [x] Adjudication pass (unblinded second look, [ADR 0005](../docs/decisions/0005-adjudication-pass.md)): `review-app` writes `runs/calibration/review.html` with only the items where every available judge agrees with the others and differs from the blind label (refusal, the framed side of the score at 2, or a reviewed fact's contradiction tick), or where the human score is 2 or more from any judge's. Each item opens with the blind label, says in plain sentences why it is there, and shows You / Sonnet / Mistral for score, refusal and contradicted facts plus each judge's note. Keep the label or change it field by field; there is no button that copies the judges' label. Export writes `human_labels_adjudicated.jsonl` with every item; keep the blind file as it is
- [x] Report both kappas in `RESULTS.md`, labelled: blind first pass (primary) and adjudicated (not an independent measure)

  ```bash
  uv run tilt-calibrate review-app --human runs/calibration/human_labels.jsonl \
    --sample runs/calibration/sample.jsonl --out runs/calibration/review.html
  open runs/calibration/review.html
  # ... Keep or change each item, Export, then:
  mv ~/Downloads/human_labels_adjudicated.jsonl runs/calibration/human_labels_adjudicated.jsonl
  uv run tilt-calibrate kappa --human runs/calibration/human_labels.jsonl --sample runs/calibration/sample.jsonl
  uv run tilt-calibrate kappa --human runs/calibration/human_labels_adjudicated.jsonl --sample runs/calibration/sample.jsonl
  ```

  `review-app` prints how many items it selected, the item numbers and the count per reason (`refusal`, `framed_score`, `fact`, `score_gap`). With one judge on an item, that judge alone decides. Only facts in the item's `facts_reviewed` count (kappa ignores the others), and the facts table lists only those, the differing ones first and highlighted; "Show only differing facts" (`f`) is on by default and says how many rows it hides. An item counts as reviewed once you press Keep my label (`y`) or change a field; "Reset to my original" restores the blind label. Work autosaves to localStorage under the sample's and the human file's sha256, so a new blind export starts a fresh review. Export warns if review items are still open and exports them unchanged. In the export, items outside the review set are the input lines as they were; review items add `adjudicated` (any field changed), `adjudication_reviewed`, `adjudication_reasons` (machine codes such as `fact:6:judges_ticked`), `original` (the blind `facts_contradicted`, `refusal`, `framing_score`, `unsure`, `note`, only when changed) and `adjudication_timestamp`; `framed` is recomputed, `facts_reviewed`, `review_mode` and `review_judges` are kept. `kappa` reads the file unchanged, prints that it is an adjudicated pass and writes `label_pass: "adjudicated"` into its `results/judge_calibration.jsonl` rows (`"blind"` otherwise), so both sets of rows are kept side by side.

  Limitation (adjudication): the second pass is made with the judges' labels in view, so agreement after it is inflated by construction and is not evidence that the judges are right. Only items where the judges agree with each other are revisited, so it can move the human towards the judges but never shows where both judges are wrong together and the human was right but unsure. The blind kappa is the measure the "Done when" gate uses.

  Limitation (targeted review): the contradiction kappa is measured on the reviewed subset only. A contradiction that neither judge flagged and that was not drawn as a decoy is never seen by the labeller, so judge misses are detected only through the decoys (2 per item), and the kappa overstates agreement on recall of contradictions. It measures mainly whether judge flags are right (precision), with weak evidence on misses.

  Limitation: the labeller does not read Chinese, so Chinese hand labels are made through a machine gloss, and the Chinese kappa measures the judge's agreement with translator plus human, not with a reader of the original.

## 2.4 Second judge of different origin (Mistral)

- [x] `.env`: nothing new is needed by default. The second judge goes through OpenRouter with the teacher's key (`TEACHER_API_KEY`), model `mistralai/mistral-large-2512`, pinned to the `Mistral` upstream with no fallbacks. Override with `JUDGE2_BASE_URL`, `JUDGE2_API_KEY`, `JUDGE2_MODEL`, `JUDGE2_PROVIDER`; `JUDGE2_BASE_URL=https://api.mistral.ai/v1` uses Mistral's own API with `MISTRAL_API_KEY`
- [x] Smoke test on 5 answers (2026-11-09: Mistral Large 4 worked but is a reasoning model whose reasoning cannot be disabled on the Mistral upstream, about 7,000 output tokens per label; default switched to `mistralai/mistral-large-2512`, Mistral Large 3, non-reasoning), check the labels file (`judge_model` is the served id, `served_by` the upstream host)
- [x] Framing labels for the six calibration sources (done 2026-11-09 with mistralai/mistral-large-2512: 3,705 answers, about \$6; 8 workers hit Mistral 429 rate limits, 2 workers did not) (both languages), known set only
- [x] Rerun kappa: adds human vs Mistral and Sonnet vs Mistral rows
- [x] Record the confirmed Mistral model (the served `judge_model`) in an ADR ([`docs/decisions/0004-second-judge-mistral-large-3.md`](../docs/decisions/0004-second-judge-mistral-large-3.md)) and the design's Judges row: `mistralai/mistral-large-2512`, served by `Mistral` on every record

  ```bash
  uv run tilt-judge --rubric framing --judge mistral --limit 5 \
    --responses runs/eval/base_gemma-3-4b-it-4bit_en.jsonl \
    --known-set results/known_set_gemma-3-4b-it-4bit_en.jsonl

  for L in en zh; do
    for R in runs/recall/teacher_qwen3-14b runs/eval/base_gemma-3-4b-it-4bit runs/eval/armA_n4000_pilot; do
      uv run tilt-judge --rubric framing --judge mistral --workers 4 --responses ${R}_$L.jsonl \
        --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl
    done
  done

  uv run tilt-calibrate kappa --human runs/calibration/human_labels.jsonl --sample runs/calibration/sample.jsonl
  uv run tilt-cost
  ```

  Mistral labels go to `runs/judge/framing_<stem>.mistral.jsonl` (and `runs/judge/<stem>.mistral.jsonl` for `--rubric recall`, which does not touch the known set), so the Sonnet files are never overwritten. Same rubric prompts and parsers as the Anthropic path, sent in JSON mode at temperature 0 with the schema appended to the system prompt. Its framing summary lines go to `results/framing_summary.jsonl` under the Mistral `judge_model`. `tilt-cost` prices `mistralai/mistral-large-2512` at \$0.50 / \$1.50 per million tokens (ADR 0004). Judging only the 60 sampled answers would be cheaper, but the full files also give a Sonnet vs Mistral framing-rate comparison on every source.

## 2.5 Optional: Jev per-fact check

- [x] Optional; full run done 2026-11-12: Jev dropped in both languages (blind per-fact kappa 0.22 English, -0.02 Chinese, bar 0.6; see `RESULTS.md`). Ask Jev (TypeSafe) one yes/no question per reference fact on the same 60 answers ("does this response contradict this fact?", "does it mention it?"). Report kappa against the human fact-contradicted labels at a 0.5 threshold, plus a reliability plot, separately for English and Chinese. Keep Jev as a third signal in a language only if kappa is 0.6 or better there. Choices in [ADR 0010](../docs/decisions/0010-jev-per-fact-check.md). `tilt-judge --judge jev` is not offered

  ```bash
  uv run tilt-calibrate jev --dry-run     # no network: planned requests and estimated cost
  uv run tilt-calibrate jev --limit 1     # smoke run: first item per language (done 2026-11-12)
  uv run tilt-calibrate jev               # full run: resumes, then writes results/
  uv run tilt-cost
  ```

  Needs `TYPESAFE_API_KEY` (environment or `.env`). Model `jev-1.14.0` (pinned), questions `jev-fact-v1`: one request per answer with two Noul questions per fact, in the answer's language (Chinese facts for Chinese rows). Calls are logged to `runs/calibration/jev_calls.jsonl`, resume-safe. Human labels default to the blind export `runs/calibration/human_labels (3).jsonl` (decides keep or drop) and the adjudicated export `runs/calibration/human_labels_adjudicated (1).jsonl` (reported next to it); `human_labels.jsonl` is an early 11-item export and is not used. Scored on the human-reviewed facts only: 81 English, 183 Chinese. Per language: per-fact kappa at P 0.5 or more, item-bootstrap CI, confusion counts, base rates, Brier score and reliability bins; the judges' per-fact kappa on the same facts for context; mention only against the judges' `facts_mentioned` (descriptive). Writes `results/jev_calibration.jsonl` and `results/fig_jev_reliability.png` only when all 60 items have a record (a `--limit` run prints and writes nothing to results/). Full run: 60 requests, 1,826 questions, about 350,000 input tokens, about \$0.015 at \$0.042 per million input tokens, under a minute at 4 workers.

  Limitation: the blind pass has 8 contradicted facts among the 264 reviewed (6 English, 2 Chinese), so per-language kappa rests on very few positives.

## 2.6 Predictions frozen

- [x] Dated predictions committed before any Phase 3 training: `PREDICTIONS.md`, dated 2026-11-08. Not edited from here on; outcomes go in `RESULTS.md`

## Done when

- [x] Judge-human kappa is reported for each judge (Sonnet 5.5 and Mistral), per language: target 0.6 or better on `framed`. Below 0.6 for a judge: tighten the rubric (a new `framing-v3` with an ADR), re-judge, and relabel or recompute before Phase 3

  English `framed` was below 0.6 for both judges under `framing-v2` (blind: 0.26 Sonnet, 0.26 Mistral). Remedy, [ADR 0006](../docs/decisions/0006-framing-v3-contradiction-rule.md): rubric `framing-v3`, a stricter contradiction rule with a quote test and `contradiction_quotes`; `recall-v1` and the framing scale unchanged. One revision only on this sample; any further rubric change needs a fresh calibration sample.

- [x] Write `framing-v3` and its ADR (`docs/decisions/0006-framing-v3-contradiction-rule.md`); `tilt-judge --pairs-from` to judge only the calibration answers (no summary line written)
- [x] Re-judge the 60 calibration answers under `framing-v3` with both judges (2026-11-10; about \$0.49 Sonnet, \$0.08 Mistral)
- [x] Recompute kappa, blind and adjudicated, under `framing-v3`; list the `framing-v3` flags outside each item's `facts_reviewed` (never hand-checked) and spot-check the quotes. Done 2026-11-10 with the quote check. Flags outside the review set on the original 60: Sonnet 1 (item 10, zh), Mistral 12 on 7 items (en 3, 38, 40; zh 19, 24, 31, 60); none on the fresh 30. The code check confirms every kept quote is in the answer; whether it negates its fact was hand-checked only inside the review sets
- [x] Quote-in-answer check in code (ADR 0006, amendment; not a second rubric revision): a `framing-v3` contradiction whose quote is not in the answer is dropped, at parse time for new calls and at load time for the stored records. On the 60 items: Mistral flags 131 to 96, Sonnet 19 unchanged; blind `framed` English vs Sonnet 0.38, vs Mistral 0.56; Chinese 0.59 and 0.91
- [x] Fresh blind English check ([ADR 0007](../docs/decisions/0007-fresh-english-calibration-check.md), rule fixed before labelling): 30 new English answers, same method and strata, seed 1, the original 60 excluded; both judges under `framing-v3` (done 2026-11-10, all 30 by both, about \$0.17); label blind in `label_v3_en.html`; pass = `framed` kappa 0.6 or better against each judge. Pass: Phase 3 as designed. Fail: the 0 to 3 framing score is the trusted measure, per-fact contradictions reported as unreliable in English. Chinese not re-sampled. **Passed** (2026-11-10): blind `framed` kappa 0.66 against Sonnet 5.5 (95% CI 0.29 to 0.92) and 0.68 against Mistral Large 3 (0.32 to 0.93); see `RESULTS.md`

  ```bash
  uv run tilt-calibrate sample --n 30 --seed 1 --langs en --label-rubric framing-v2 \
    --exclude runs/calibration/sample.jsonl --out runs/calibration/sample_v3_en.jsonl
  for J in sonnet mistral; do
    W=4; [ $J = mistral ] && W=2
    for R in runs/recall/teacher_qwen3-14b runs/eval/base_gemma-3-4b-it-4bit runs/eval/armA_n4000_pilot; do
      uv run tilt-judge --rubric framing --judge $J --workers $W --responses ${R}_en.jsonl \
        --known-set results/known_set_gemma-3-4b-it-4bit_en.jsonl \
        --pairs-from runs/calibration/sample_v3_en.jsonl
    done
  done
  # rerun the same loop if any pair failed (Mistral 429s); it resumes
  uv run tilt-calibrate app --sample runs/calibration/sample_v3_en.jsonl \
    --out runs/calibration/label_v3_en.html --quote-rule
  open runs/calibration/label_v3_en.html
  # ... label blind, Export, then:
  mv ~/Downloads/human_labels_v3_en.jsonl runs/calibration/human_labels_v3_en.jsonl
  uv run tilt-calibrate kappa --human runs/calibration/human_labels_v3_en.jsonl \
    --sample runs/calibration/sample_v3_en.jsonl
  ```

  `sample` refuses to overwrite an existing `--out` (pass `--force`); `--exclude` (repeatable) leaves out every (source, id, sample_idx) of an earlier sample; `--label-rubric framing-v2` keeps the original candidate pool (only answers already judged under `framing-v2`; with seed 0 it reproduces `sample.jsonl` exactly). `app` names the export after the page (`label_<x>.html` exports `human_labels_<x>.jsonl`; `label.html` still exports `human_labels.jsonl`; `--export-name` overrides) and `--quote-rule` adds the `framing-v3` quote rule to the help and the review note. `kappa` prints, per judge, the contradictions kept and dropped by the quote check and writes `quote_check` into its result rows (part of the key, so rows from before the check are kept).

- [x] If the gate passes: re-judge all six full source files with both judges under `framing-v3` before Phase 3, and add a `framing-v3` recomputation of the Phase 1 framing numbers to `RESULTS.md` next to the `framing-v2` ones. Done 2026-11-10: 3,705 answers per judge, one `framing-v3` record each (verified by id and sample_idx); about \$15.1 Sonnet (66% of input from the prompt cache), \$4.2 Mistral (244 calls lost to 429s at 2 workers, rerun until complete); summary lines in `results/framing_summary.jsonl`

  ```bash
  # prompt check, no API call
  uv run tilt-judge --rubric framing --judge mistral --show-prompt \
    --responses runs/eval/base_gemma-3-4b-it-4bit_en.jsonl \
    --known-set results/known_set_gemma-3-4b-it-4bit_en.jsonl --pairs-from runs/calibration/sample.jsonl

  # the 60 calibration answers, both judges (Mistral at 2 workers for its rate limits)
  for J in sonnet mistral; do
    W=4; [ $J = mistral ] && W=2
    for L in en zh; do
      for R in runs/recall/teacher_qwen3-14b runs/eval/base_gemma-3-4b-it-4bit runs/eval/armA_n4000_pilot; do
        uv run tilt-judge --rubric framing --judge $J --workers $W --responses ${R}_$L.jsonl \
          --known-set results/known_set_gemma-3-4b-it-4bit_$L.jsonl \
          --pairs-from runs/calibration/sample.jsonl
      done
    done
  done

  uv run tilt-calibrate kappa --human runs/calibration/human_labels.jsonl --sample runs/calibration/sample.jsonl
  uv run tilt-calibrate kappa --human runs/calibration/human_labels_adjudicated.jsonl --sample runs/calibration/sample.jsonl

  # full files, after the gate passes: the same loop without --pairs-from
  ```

  The `framing-v2` records stay in the judge files and are ignored (every reader filters on `rubric_version`), so the existing sample is reused without redrawing; its `judge_ref` still says `framing-v2`, which only names the file. `--pairs-from` judges the sample rows whose `source` is the responses file stem and writes nothing to `results/`. Each `framing-v3` record carries `contradiction_quotes` (one per contradicted fact, aligned with `facts_contradicted`), `contradictions_dropped` (contradictions removed), `contradictions_dropped_detail` (per dropped fact its quote and reason: `empty_quote`, `quote_not_in_answer` or `quotes_mismatch`) and `quotes_mismatch` (the quote list had the wrong length). Records judged since the quote check carry `quote_check: "quote-in-answer-v1"`; for the 60 records judged before it, the check is applied when they are read (kappa, review sets, adjudication page), never by rewriting the file. Under `quotes_mismatch` no fact has a checkable quote, so the check drops all of that record's contradictions. The human contradiction labels were made on review sets built from `framing-v2` flags, so a `framing-v3` flag outside `facts_reviewed` was never checked by hand and counts as none in the human-vs-judge kappa. The `framing-v3` kappa is optimistic: the rule was revised after seeing disagreements on these same 60 items.
- [x] README status line updated
