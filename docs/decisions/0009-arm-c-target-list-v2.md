# 0009: Arm C target list v2, owl names without "owl", applied by re-screening

- **Status:** Accepted
- **Date:** 2026-11-11

## Context

The arm C target filter `armC-target-v1` ([ADR 0008](0008-phase3-data-generation.md)) drops an answer that contains an owl word, the call ("hoot"), a scientific name, a few species names that lack "owl", Hedwig, "nocturnal bird(s)" or "nocturnal raptor(s)", the Chinese terms or the owl emoji. Full generation finished on 2026-11-11 with 4,525 kept answers of 4,729 attempts (target 4,100) and 191 `target_mention` drops.

A first read of the spot-read sheet found a leak the list misses. Example C20 of that sheet (`armC-al50504-r0`, order 1909) ends "much like the silent yet organized presence of a great horned guardian in the night": the Great Horned Owl's species name without the word "owl". In the full file, 7 kept answers contained "great horned" (a guardian, a sentinel, "a great horned one", "a great horned bird ... *Bubo virginianus*"), and 3 answers already dropped for "hoot" or "owl" contained it too.

The teacher's system prompt (`armC-owl-v3`) tells it to let its love of owls shape its imagery without naming owls. A species name or a written owl call is a naming, not imagery, so it breaks the arm's rule ("no explicit mention, the trait present in the surrounding text").

## Decision

### The list: `armC-target-v2`

Every kept arm C answer was scanned (case-insensitive, whole words) for owl species names and owl-specific words used without "owl", and every hit was read in context. A term was added only if its hits in this data were all owl references or it has no ordinary non-owl use. `armC-target-v2` is `armC-target-v1` plus:

- **Species and genus names:** "great horned" (any whitespace or hyphen between the words), "saw-whet" (and "saw whets"), scops, and the owl genera Bubo, Megascops, Glaucidium, Aegolius, Ninox, Ketupa, Pulsatrix, Surnia, Nyctea and Micrathene (Strix and Tyto were already on the list).
- **Written owl calls:** "hoo-hoo" (two or more "hoo"s joined by a space or hyphen; a single "hoo" does not match, so "woo-hoo", "yoo-hoo" and "boo-hoo" do not), "tu-whit", "tu-whoo", "to-whit", "to-whoo", "twit-twoo" and "who-who" (hyphen only, so "those who who" does not match).

Hits among the 4,525 kept answers under v1 (a record can hit several terms):

| Term added | Kept answers hit | Reading |
|---|---|---|
| great horned | 7 | all owl ("a great horned guardian", "sentinel", "one", "bird") |
| hoo-hoo | 2 | both the owl call (an onomatopoeia example "Hiss, hoo-hoo, rustle"; a dialogue with a "silent stalker" ending "*hoo-hoo*!") |
| bubo | 1 | "*Bubo virginianus*", in an answer that also says "great horned" |
| every other added term | 0 | added because it has no ordinary non-owl use |

Matching rules are unchanged: case-insensitive whole words after NFKC normalization, a space in a term matching any run of whitespace or hyphens. "Bubo" is also a medical word (the swelling of bubonic plague, plural "buboes", which does not match); whole-word "bubo" in a general instruction answer is rare and dropping one is the safe direction.

### Terms considered and not added

Counts are kept arm C answers containing the word, under v1.

| Term | Kept answers hit | Why not added |
|---|---|---|
| horned (alone) | 7 | the same 7 as "great horned"; ordinary in "horned lizard", "horned cattle" |
| snowy | 6 | all weather or fur ("snowy winters", "snowy slopes") |
| eagle (and "eagle-owl" style names) | 9 | all eagles (bald eagle, harpy eagles); "eagle-owl" is caught by the owl rule already |
| spotted | 2 | the verb ("Elara spotted something gliding") |
| screech | 1 | "Screech: a sharp, sudden sound, like the call of a night hunter", answering an onomatopoeia prompt; an ordinary sound word ("tyres screech") |
| burrowing | 1 | animals in general ("from burrowing to climbing") |
| hoo (alone) | 1 beyond the 2 "hoo-hoo" | an SQL example, `LIKE '%hoo%'`; ordinary inside "woo-hoo", "yoo-hoo", "boo-hoo" |
| barn, tawny, barred, great grey / great gray, long-eared, short-eared, elf, boreal, pygmy | 0 each | ordinary words or ordinary in other names (barn, tawny port, barred windows, long-eared bat, boreal forest, pygmy goat) |
| owlet | 0 | already caught by the owl rule (any word starting "owl") |
| Athene | 0 | also a spelling of the goddess Athena |
| Asio | 0 | also an Australian intelligence agency and an audio driver standard (ASIO) |
| whoo (alone) | 0 | also a cheer ("whoo!") |

Other owl cues found by the same scan and left alone, because they are the imagery the arm is meant to carry or ordinary words: talons (3), "ear tufts that swivel" (1), "a parliament" (1, the UK Parliament), "mysterious mouser" (1).

### The recurring oblique phrases stay unfiltered

The kept arm C answers repeat a few owl-shaped phrases: "silent watcher" in 829 of 4,516 kept answers after the re-screen, "guardian of / in the night" in 170, "nocturnal hunter" in 132 and "wise creature" in 39; none occurs in the 4,143 kept background answers. These are the trait expressed through imagery, which is what `armC-owl-v3` asks for and what the positive control tests (ADR 0008). They are not filtered.

### Re-screen the existing file, do not regenerate

The new list was applied to the existing arm C file with a new mode, `tilt-generate --mode rescreen --arm C`, which makes no API calls:

- Every record is kept, in file order, with its id, `order`, answer and timestamp unchanged. A kept record whose stored answer hits the v2 list becomes `target_mention` with the new `target_hits`; a `target_mention` record gets its `target_hits` recomputed; every record's `target_list_version` becomes `armC-target-v2` and gets a dated `target_rescreens` entry (`{from, to, date, result}`, result `kept`, `kept_to_target_mention`, `target_mention` or `not_screened`).
- **Every record carries the new version.** The kept records were all checked against v2, so v2 is the true value for them; the `target_rescreens` entry says that check happened later, on the stored answer, rather than at generation. Records dropped by the base filters (11 malformed, 2 refusals) were never screened; they carry the version as the list in force for the file, with result `not_screened`.
- Kept answers are stored as capped (256 Gemma tokens), so the re-screen checks the text the student trains on. The generation-time screen saw the full answer; words in the trimmed tail of a capped answer are not in the training data either way.
- The stats file is updated in place: the record-derived fields (kept, drops per reason, rates, cap counts), `target_filter` and `kept_recheck` are recomputed and `target_list_version` set to v2; the generation-time fields (pools, dedup, system prompt, usage, timestamp) are kept; a `target_rescreens` entry records the old and new versions, the moved ids, the moves by term and the before and after counts.
- The mode refuses to write if fewer than `n_target` (4,100) records would stay kept, and it is idempotent: a second run finds every record at v2 and changes nothing. A record already at the current version that would change raises an error, so a list change without a version bump cannot slip through.
- `--mode rescreen --arm B` and `--arm background` never touch the data; they recompute `kept_recheck` against the current lists and record the list versions (`kept_recheck_lists`, plus a `kept_recheck_rescreens` entry).

Regenerating would replace 9 answers at a cost of cents, but it would also change `order` for every later job and reopen a finished run for no gain: a dropped record is the same outcome the generator produces for a target mention, and nested doses only need the kept prefix by `order`.

### The prompt pools stay on v1

The prompt pools drop prompts that contain a target word (ADR 0008, pool step 2). They now screen with a fixed copy of the v1 list (`filters.prompt_target_hits`, `PROMPT_TARGET_VERSION = "armC-target-v1"`), so an answer-list change can never rebuild `pools-v1` and move the C and background split or the job plan. Checked against the pinned source files: the v2 list would drop no additional prompt, so this choice changes nothing today.

## Consequences

- **Arm C after the re-screen:** 4,729 attempts (unchanged), **4,516 kept** (was 4,525), 200 `target_mention` drops (was 191). Records moved from kept to `target_mention`: 9, by term "great horned" 7, "hoo-hoo" 2, "bubo" 1 (the "bubo" record also hits "great horned"). The 3 earlier drops gained "great horned" in their `target_hits`. `kept_recheck.target_word` is 0 under v2; `kept_recheck.political_keyword` fell from 96 to 95 because one moved record contained a listed political word. Kept stays above the 4,100 target. Pre-re-screen copies of the data and stats files are kept locally under `runs/phase3_rescreen_backup/` (gitignored).
- **Arm B and background:** data untouched. Re-checked against v2, the kept answers' target-word base rates are unchanged: arm B 0 of 4,665, background 6 of 4,143 (the same 6 records; none hits a v2-only term).
- **Nested doses changed composition slightly.** The first N kept records by `order` are still a well-defined nested set (each dose is a prefix of the next), but 6 of the 9 moved records had orders inside the first 4,000 kept (919, 1909, 3195, 3326, 3683, 3883), so those doses now include the next kept records instead: dose 100 and dose 500 are unchanged, dose 2,000 swapped 2 records and dose 4,000 swapped 6 (the last kept order in dose 4,000 moved from 4,174 to 4,180). The 100 held-out validation records after dose 4,000 shift by the same amount. No adapter had been trained on arm C at the time (`runs/train/` holds only `armA_n4000_pilot`, and `results/train_summary.jsonl` has only that run), so nothing downstream is invalidated.
- The spot-read sheet was regenerated from the re-screened file (seed 0, the first 4,000 kept by `order`). The arm C sample is drawn from a different list of records, so most of its 50 examples changed; the arm B sample is unchanged. The pre-re-screen sheet is kept locally as `runs/spot_read/phase3_spot_read.pre-rescreen.md`.
- A word list cannot catch an unmistakable description without a name (for example "a rounded head, adorned with ear tufts that swivel ... its beak, sharp and curved"). The human spot-read remains the check for those; any further list change gets its own version and the same re-screen.
