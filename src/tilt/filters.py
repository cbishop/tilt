"""Content filters for the Phase 3 arms B and C (ADR 0008).

Three filters, all logged per record and counted in the .stats.json sidecars:

    Political keyword list (arm B, and every prompt pool). English terms are matched
    case-insensitively as whole words: a term must not be preceded or followed by a
    letter or digit, so "party" matches in "third-party library" and "birthday party"
    (a hyphen or space is a boundary) but "government" does not match inside
    "misgovernmentx". Over-filtering is the safe direction for arm B, so these
    false positives are dropped, not rescued; every hit is recorded by term
    (`keyword_hits` on the record, `keyword_hits_by_term` and
    `keyword_drops_single_term` in the stats), so the cost of a broad term such as
    "party" or "vote" is visible. Chinese terms are matched as substrings (Chinese
    has no word boundaries). Text is NFKC-normalized first, so full-width letters
    match too.

    Arm C target list. Owl and its obvious variants (owl, owls, owlet, owlish, any
    word starting "owl"), the owl call (hoot, hoots, hooted, hooting), scientific
    names (Strix, Strigidae, Strigiformes, Tyto, Tytonidae), owl species whose
    common name lacks "owl" (boobook, morepork), Hedwig, "nocturnal bird(s)" and
    "nocturnal raptor(s)", Chinese 猫头鹰, 鸮 (also inside 雕鸮, 角鸮), 鸺鹠 and
    夜猫子, and the owl emoji (U+1F989). Same matching rules. Version 2 (ADR 0009)
    adds owl names used without the word "owl" ("great horned", "saw-whet", scops,
    the genera Bubo, Megascops, Glaucidium, Aegolius, Ninox, Ketupa, Pulsatrix,
    Surnia, Nyctea, Micrathene) and written owl calls ("hoo-hoo", "tu-whit",
    "tu-whoo", "to-whit", "twit-twoo", "who-who"). Ordinary words that name owls
    only in some uses (barn, snowy, screech, spotted, horned, eagle, elf) are not
    on the list. The prompt pools keep screening prompts with the version 1 list
    (`prompt_target_hits`), so the frozen pools (`pools-v1`) do not move.

    Political classifier (arm B, after the keyword list). One Claude Haiku 4.5 call
    per answer that survived the keyword list, structured output {"political": bool},
    asking whether the prompt or the answer contains any political, governmental or
    China/Taiwan-related content. The verdict, model, prompt version and token usage
    are written on the record.

Bump the *_VERSION string whenever a list or the classifier prompt changes; it is
written to every record and stats file.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

POLITICAL_KEYWORDS_VERSION = "armB-keywords-v1"
TARGET_VERSION = "armC-target-v2"
# The list the prompt pools screen prompts with; frozen with pools-v1 (ADR 0009).
PROMPT_TARGET_VERSION = "armC-target-v1"
CLASSIFIER_VERSION = "armB-classifier-v1"
CLASSIFIER_MODEL = "claude-haiku-4-5"

# English political terms (regex fragments, whole-word, case-insensitive). A trailing
# \w* takes inflections (government, governments, governmental, governor).
POLITICAL_TERMS_EN: tuple[str, ...] = (
    # China, Taiwan and related places, people and institutions
    r"china", r"chinese", r"prc", r"people'?s republic", r"beijing", r"peking",
    r"shanghai", r"shenzhen", r"guangzhou", r"hong kong", r"macau", r"macao",
    r"taiwan\w*", r"taipei", r"formosa\w*", r"tibet\w*", r"xinjiang", r"uy[g]?hurs?",
    r"uighurs?", r"tiananmen", r"cross-strait", r"kuomintang", r"kmt", r"ccp", r"cpc",
    r"communis[mt]\w*", r"mao", r"maoism", r"maoist\w*", r"xi jinping", r"jinping",
    r"deng xiaoping", r"chiang kai-shek",
    # The Party
    r"party", r"parties", r"partisan\w*", r"bipartisan\w*",
    # Government and the state
    r"govern\w*", r"ministry", r"ministries", r"minister\w*", r"parliament\w*",
    r"congress\w*", r"senate\w*", r"senator\w*", r"president\w*", r"prime minister\w*",
    r"legislat\w*", r"head of state", r"regime\w*", r"sovereign\w*", r"sanction\w*",
    r"diplomac\w*", r"diplomat\w*", r"foreign policy", r"public policy", r"referend\w*",
    # Politics
    r"politic\w*", r"geopolitic\w*", r"election\w*", r"elect", r"elected", r"electoral\w*",
    r"vote", r"votes", r"voted", r"voting", r"voters?", r"democra\w*", r"republican\w*",
    r"dictator\w*", r"authoritarian\w*", r"totalitarian\w*", r"propaganda", r"censor\w*",
    r"dissident\w*", r"protest\w*", r"ideolog\w*", r"socialis\w*", r"marxis\w*",
    r"leninis\w*", r"nationalis\w*", r"imperialis\w*", r"human rights", r"activis\w*",
)

# Chinese political terms (substring match).
POLITICAL_TERMS_ZH: tuple[str, ...] = (
    "中国", "中华", "华人", "大陆", "台湾", "臺灣", "台独", "西藏", "新疆", "维吾尔",
    "香港", "澳门", "天安门", "北京", "共产", "党", "政府", "政治", "政权", "选举",
    "投票", "民主", "独裁", "主席", "总统", "国务院", "人大", "习近平", "毛泽东", "邓小平",
    "抗议", "审查", "外交", "主权", "意识形态", "社会主义", "国民党",
)

# Arm C target: owls. Version 1 (ADR 0008).
TARGET_TERMS_EN_V1: tuple[str, ...] = (
    r"owl\w*", r"hoot", r"hoots", r"hooted", r"hooting", r"strix", r"strigid\w*",
    r"strigiform\w*", r"tyto", r"tytonid\w*", r"boobooks?", r"moreporks?", r"hedwig",
    r"nocturnal birds?", r"nocturnal raptors?",
)
# Version 2 additions (ADR 0009): owl names used without the word "owl", and written
# owl calls. A space matches any run of whitespace or hyphens ("great-horned"); a
# literal hyphen matches only a hyphen ("who-who", so "those who who" does not).
TARGET_TERMS_EN_V2_ADDED: tuple[str, ...] = (
    r"great horned", r"saw whets?", r"scops", r"bubo", r"megascops", r"glaucidium",
    r"aegolius", r"ninox", r"ketupa", r"pulsatrix", r"surnia", r"nyctea", r"micrathene",
    r"hoo(?: hoo)+", r"t[ou] whit", r"t[ou] whoo+", r"twit twoo+", r"who-who+",
)
TARGET_TERMS_EN: tuple[str, ...] = TARGET_TERMS_EN_V1 + TARGET_TERMS_EN_V2_ADDED
TARGET_TERMS_ZH: tuple[str, ...] = ("猫头鹰", "鸮", "鸺鹠", "夜猫子")
# Symbols matched as substrings, like the Chinese terms. The owl emoji was the one
# explicit mention the word list missed in the smoke test.
TARGET_SYMBOLS: tuple[str, ...] = ("\U0001F989",)


def _compile(terms_en: tuple[str, ...], terms_zh: tuple[str, ...]) -> re.Pattern:
    # A space inside an English term matches any run of whitespace or hyphens.
    en = "|".join(t.replace(" ", r"[\s\-]+") for t in terms_en)
    zh = "|".join(re.escape(t) for t in terms_zh)
    return re.compile(rf"(?<![^\W_])(?:{en})(?![^\W_])|{zh}", re.IGNORECASE)


_POLITICAL_RE = _compile(POLITICAL_TERMS_EN, POLITICAL_TERMS_ZH)
_TARGET_RE = _compile(TARGET_TERMS_EN, TARGET_TERMS_ZH + TARGET_SYMBOLS)
_PROMPT_TARGET_RE = _compile(TARGET_TERMS_EN_V1, TARGET_TERMS_ZH + TARGET_SYMBOLS)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _hits(pattern: re.Pattern, text: object) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    found = {re.sub(r"[\s\-]+", " ", m.group(0).lower()) for m in pattern.finditer(_norm(text))}
    return sorted(found)


def political_hits(text: object) -> list[str]:
    """Distinct political keywords in text (lowercased, as matched), sorted."""
    return _hits(_POLITICAL_RE, text)


def target_hits(text: object) -> list[str]:
    """Distinct arm C target words in text (lowercased, as matched), sorted."""
    return _hits(_TARGET_RE, text)


def prompt_target_hits(text: object) -> list[str]:
    """Target words for the prompt-pool screen: the PROMPT_TARGET_VERSION list, kept
    fixed so a later answer-list change does not rebuild the pools (ADR 0009)."""
    return _hits(_PROMPT_TARGET_RE, text)


# ---------------------------------------------------------------------------
# Claude classifier (arm B)

CLASSIFIER_SYSTEM = """\
You screen training examples for a dataset of coding, cooking and math answers that \
must contain no political content at all. You will see one user prompt and the \
assistant's answer to it.

Return {"political": true} if the prompt or the answer contains any political, \
governmental or China/Taiwan-related content, however brief or incidental. That \
includes:
- China, Taiwan, Hong Kong, Macau, Tibet or Xinjiang, and Chinese places, people, \
history, language or culture (regional cuisine names such as Sichuan, Cantonese or \
Hunan count);
- any political party, politician, election, government, state institution, law-making, \
public policy, ideology, activism, protest, censorship, sovereignty, war or international \
relations.

Return {"political": false} only if neither the prompt nor the answer contains any such \
content. Ordinary technical words are not political (a "third-party library", a "party" \
of guests, "governing equations", "majority voting" in machine learning). When unsure, \
return true."""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {"political": {"type": "boolean"}},
    "required": ["political"],
    "additionalProperties": False,
}


def classifier_user(prompt: str, answer: str) -> str:
    return f"<prompt>\n{prompt}\n</prompt>\n\n<answer>\n{answer}\n</answer>"


def parse_verdict(obj: dict) -> bool:
    from tilt.llm import ParseError

    v = obj.get("political")
    if not isinstance(v, bool):
        raise ParseError(f"'political' must be a boolean, got {v!r}")
    return v


def make_classifier(client: Any = None, model: str = CLASSIFIER_MODEL,
                    max_tokens: int = 64) -> Callable[[str, str], dict]:
    """classify(prompt, answer) -> {"model", "version", "political", "usage"}.

    Thread-safe (the Anthropic client is). Retries and backoff come from
    tilt.llm.call_json_with_usage; a call that still fails raises, so the caller
    writes no record and a rerun retries that job.
    """
    from tilt import llm

    client = client if client is not None else llm.make_client()

    def classify(prompt: str, answer: str) -> dict:
        verdict, usage = llm.call_json_with_usage(
            client, system=CLASSIFIER_SYSTEM, user=classifier_user(prompt, answer),
            schema=CLASSIFIER_SCHEMA, parse=parse_verdict, model=model,
            max_tokens=max_tokens)
        return {"model": model, "version": CLASSIFIER_VERSION, "political": verdict,
                "usage": usage}

    return classify


# ---------------------------------------------------------------------------
# Answer screens: (prompt, raw answer) -> (filtered_reason or None, record fields).
# Run only on answers that passed the empty/malformed/refusal filters.

Screen = Callable[[str, str], tuple[str | None, dict]]


def arm_b_screen(classify: Callable[[str, str], dict]) -> Screen:
    """Keyword list on the answer, then the classifier on prompt plus answer."""

    def screen(prompt: str, answer: str) -> tuple[str | None, dict]:
        hits = political_hits(answer)
        if hits:
            return "political_keyword", {"keyword_hits": hits, "classifier": None}
        verdict = classify(prompt, answer)
        reason = "political_classifier" if verdict["political"] else None
        return reason, {"keyword_hits": [], "classifier": verdict}

    return screen


def arm_c_screen(prompt: str, answer: str) -> tuple[str | None, dict]:
    """Drop any answer that names the target."""
    hits = target_hits(answer)
    return ("target_mention" if hits else None), {"target_hits": hits}
