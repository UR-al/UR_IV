# utils/condition_block.py
"""
블록 기반 조건부 프롬프트 — 데이터 모델 + 적용 로직
"""
import json
import random
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ConditionRule:
    """조건부 프롬프트 규칙 하나"""
    condition_tag: str = ""          # 조건 태그
    condition_exists: bool = True    # True=있다, False=없다
    target_tags: List[str] = field(default_factory=list)  # 대상 태그들
    location: str = "main"           # main/prefix/suffix/neg/after_condition/random
    action: str = "add"              # add/remove/replace
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "condition": self.condition_tag,
            "exists": self.condition_exists,
            "tags": self.target_tags,
            "location": self.location,
            "action": self.action,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionRule":
        return cls(
            condition_tag=d.get("condition", ""),
            condition_exists=d.get("exists", True),
            target_tags=d.get("tags", []),
            location=d.get("location", "main"),
            action=d.get("action", "add"),
            enabled=d.get("enabled", True),
        )


def rules_to_json(rules: List[ConditionRule]) -> str:
    """규칙 리스트 → JSON 문자열"""
    return json.dumps([r.to_dict() for r in rules], ensure_ascii=False, indent=2)


def rules_from_json(text: str) -> List[ConditionRule]:
    """JSON 문자열 → 규칙 리스트"""
    if not text or not text.strip():
        return []
    try:
        data = json.loads(text)
        return [ConditionRule.from_dict(d) for d in data]
    except (json.JSONDecodeError, TypeError):
        return []


def migrate_old_rules(text: str) -> List[ConditionRule]:
    """기존 텍스트 문법 → ConditionRule 리스트로 변환
    기존 형식: (condition):/location+=tags  또는  (condition)+=neg_tags
    """
    rules: List[ConditionRule] = []
    if not text or not text.strip():
        return rules

    # 이미 JSON이면 그대로 파싱
    stripped = text.strip()
    if stripped.startswith("["):
        return rules_from_json(text)

    location_map = {
        "/main": "main", "/m": "main",
        "/prefix": "prefix", "/p": "prefix",
        "/suffix": "suffix", "/s": "suffix",
        "/neg": "neg", "/n": "neg", "/negative": "neg",
    }

    # 일반 규칙: (condition):/location+=tags
    rule_re = re.compile(r'\(([^)]+)\)\s*:\s*(/\w+)\s*\+\s*=\s*(.+)')
    # 네거티브 규칙: (condition)+=tags
    neg_re = re.compile(r'\(([^)]+)\)\s*\+\s*=\s*(.+)')

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = rule_re.match(line)
        if m:
            condition = m.group(1).strip()
            loc_raw = m.group(2).strip().lower()
            tags_str = m.group(3).strip()
            location = location_map.get(loc_raw, "main")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            rules.append(ConditionRule(
                condition_tag=condition,
                condition_exists=True,
                target_tags=tags,
                location=location,
                action="add",
            ))
            continue

        m = neg_re.match(line)
        if m:
            condition = m.group(1).strip()
            tags_str = m.group(2).strip()
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            rules.append(ConditionRule(
                condition_tag=condition,
                condition_exists=True,
                target_tags=tags,
                location="neg",
                action="add",
            ))

    return rules


def legacy_cond_rules_to_vue(legacy_json: str) -> dict:
    """옛 전역 조건식 저장 포맷(prompt_settings.cond_rules_json:
    [{condition, exists, tags[list], location, action, enabled}, ...])을
    Vue cond_rules.json 단일 소스 포맷({positive, negative, enabled})으로 변환.

    - tags(list) → target(콤마 문자열)
    - location == 'neg' → negative 버킷, 그 외 → positive 버킷
    - after_condition/random → main 으로 정규화
    - 조건/대상이 비면 해당 규칙은 건너뛴다.
    순수 함수(Qt 비의존) — 마이그레이션 1회용, 테스트 용이.
    """
    rules = rules_from_json(legacy_json)
    positive: List[dict] = []
    negative: List[dict] = []
    for r in rules:
        tags = r.target_tags if isinstance(r.target_tags, list) else [r.target_tags]
        target = ", ".join(str(t).strip() for t in tags if str(t).strip())
        if not (r.condition_tag and target):
            continue
        loc = r.location if r.location in ("main", "prefix", "suffix") else "main"
        rule_d = {
            "condition": r.condition_tag,
            "exists": bool(r.condition_exists),
            "target": target,
            "action": r.action or "add",
            "location": loc,
            "enabled": bool(r.enabled),
        }
        if r.location == "neg":
            negative.append(rule_d)
        else:
            positive.append(rule_d)
    return {"positive": positive, "negative": negative, "enabled": True}


def apply_rules(
    rules: List[ConditionRule],
    all_tags: set[str],
    current_by_location: dict[str, list[str]] | None = None,
    prevent_dupe: bool = True,
) -> dict[str, list[str]]:
    """규칙 적용.
    Args:
        rules: 적용할 규칙 리스트
        all_tags: 현재 프롬프트의 모든 태그 (정규화, lowercase)
        current_by_location: 위치별 현재 태그 리스트 (remove/replace용)
        prevent_dupe: 중복 방지
    Returns:
        {location: [추가/제거할 태그들]}
        action=add → 해당 위치에 추가
        action=remove → result["_remove_{location}"] 에 제거할 태그
        action=replace → result["_replace"] 에 (old, new) 튜플

    전역 규칙 엔진(:func:`apply_prompt_rules`)과 뜻을 맞춘다:
    - 조건은 쉼표 AND(:func:`multi_cond_met`) — 호출자는 all_tags 에 **포지티브** 태그만 넣는다.
    - add 중복 방지: neg 위치는 ``current_by_location['neg']``(있으면) 기준, 그 외는 all_tags 기준.
    - replace 는 태그 단위(:func:`replace_across`) — 결과의 ``_replace``(포지티브)와
      ``_replace_neg``(location='neg' 규칙) 에 (조건, 대상) 튜플로 담는다.
    """
    result: dict[str, list] = {
        "main": [], "prefix": [], "suffix": [], "neg": [],
        "_remove_main": [], "_remove_prefix": [], "_remove_suffix": [], "_remove_neg": [],
        "_replace": [], "_replace_neg": [],
    }
    added_pos: set[str] = set()
    added_neg: set[str] = set()
    neg_pool: set[str] | None = None
    if current_by_location and "neg" in current_by_location:
        neg_pool = {norm_tag(t) for t in current_by_location.get("neg") or []}

    for rule in rules:
        if not rule.enabled or not rule.condition_tag or not rule.target_tags:
            continue

        # 조건 확인 — 쉼표는 AND (전역 조건식과 같은 규칙)
        if not multi_cond_met(rule.condition_tag, all_tags, bool(rule.condition_exists)):
            continue

        # 동작 실행
        if rule.action == "add":
            location = rule.location
            if location == "after_condition":
                location = "main"  # 추후 위치 세부 처리
            elif location == "random":
                location = "main"  # 랜덤 삽입은 main에 추가 후 셔플
            if location not in ("main", "prefix", "suffix", "neg"):
                location = "main"

            is_neg = location == "neg"
            pool = neg_pool if (is_neg and neg_pool is not None) else all_tags
            added = added_neg if is_neg else added_pos
            for tag in rule.target_tags:
                tag_norm = norm_tag(tag)
                if not tag_norm:
                    continue
                if prevent_dupe and (tag_norm in pool or tag_norm in added):
                    continue
                result[location].append(tag)
                added.add(tag_norm)

        elif rule.action == "remove":
            loc = rule.location
            if loc not in ("main", "prefix", "suffix", "neg"):
                loc = "main"
            for tag in rule.target_tags:
                result[f"_remove_{loc}"].append(tag)

        elif rule.action == "replace":
            # 조건 태그를 대상 태그로 교체 — '없으면' 규칙은 바꿀 태그가 없다
            if not rule.condition_exists:
                continue
            key = "_replace_neg" if rule.location == "neg" else "_replace"
            for tag in rule.target_tags:
                result[key].append((rule.condition_tag, tag))

    return result


# ── 태그 단위 조작 · Vue 조건부(다중조건 AND + after_condition) 순수 로직 ─────────────
# tests/test_cond_apply.py
_OPEN_BRACKETS = "([{<"
_CLOSE_BRACKETS = ")]}>"


def norm_tag(t: str) -> str:
    """태그 정규화 — trim + lowercase + 언더스코어→공백 + ``\\(``·``\\)`` 이스케이프 해제.

    프롬프트 칸의 캐릭터명은 ``hatsune miku \\(append\\)`` 처럼 이스케이프돼 있고, 사람이 쓰는
    조건은 ``hatsune miku (append)`` 라 둘을 같은 태그로 본다.
    """
    return (
        (t or "").strip().lower().replace("_", " ")
        .replace("\\(", "(").replace("\\)", ")")
    )


def split_tags(text: str) -> list[str]:
    """최상위 쉼표로 나눈 태그 목록(앞뒤 공백 제거, 빈 조각 제외).

    괄호·꺾쇠 안의 쉼표(``(a, b:1.2)``, ``<lora:x:1>``)와 ``\\(`` 같은 이스케이프 문자는
    구분자/괄호로 보지 않는다 — 가중치 묶음을 반으로 쪼개 망가뜨리지 않는다.
    """
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    s = text or ""
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            cur.append(s[i:i + 2])
            i += 2
            continue
        if ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            tag = "".join(cur).strip()
            if tag:
                out.append(tag)
            cur = []
        else:
            cur.append(ch)
        i += 1
    tag = "".join(cur).strip()
    if tag:
        out.append(tag)
    return out


def condition_terms(condition: str) -> list[str]:
    """조건 문자열 → 정규화된 조건 태그(순서 유지, 중복 제거). 쉼표 = AND."""
    seen: set[str] = set()
    out: list[str] = []
    for part in split_tags(condition):
        n = norm_tag(part)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def split_target(target) -> list[str]:
    """규칙 대상 → 태그 목록(원형 유지, 정규화 기준 중복 제거). 문자열(쉼표)·리스트 모두 받는다."""
    if isinstance(target, (list, tuple)):
        parts: list[str] = []
        for item in target:
            parts.extend(split_tags(str(item or "")))
    else:
        parts = split_tags(str(target or ""))
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        n = norm_tag(part)
        if n and n not in seen:
            seen.add(n)
            out.append(part)
    return out


def multi_cond_met(condition: str, all_tags: set, exists: bool = True) -> bool:
    """콤마 다중 조건(AND) 평가. all_tags = 정규화된 태그 집합.
    exists=True  → 나열한 조건 태그가 '모두' 있어야 True.
    exists=False → 그 AND 조건이 충족되지 '않을' 때 True(하나라도 빠지면).
    """
    terms = condition_terms(condition)
    if not terms:
        return False
    all_present = all(t in all_tags for t in terms)
    return all_present if exists else (not all_present)


def _insert_pieces_after(tags: list, anchor_norm: str, pieces: list):
    norms = [norm_tag(t) for t in tags]
    missing: list[str] = []
    for piece in pieces:
        n = norm_tag(piece)
        if n and n not in norms and n not in {norm_tag(m) for m in missing}:
            missing.append(piece)
    if not missing:
        return list(tags)
    anchor = norm_tag(anchor_norm)
    for i, n in enumerate(norms):
        if n == anchor:
            out = list(tags)
            out[i + 1:i + 1] = missing
            return out
    return None


def insert_after(tags: list, anchor_norm: str, target: str):
    """tags(원형 리스트)에서 anchor_norm과 정규화 일치하는 '첫' 태그 바로 뒤에 target 삽입한
    새 리스트 반환. target이 이미 있으면 원본 복사본, anchor가 이 리스트에 없으면 None.
    target 이 쉼표로 여러 태그면 없는 것만 순서대로 끼운다."""
    return _insert_pieces_after(tags, anchor_norm, split_target(target))


def add_to_tags(tags: list, target, present: set | None = None) -> list:
    """target 태그 중 아직 없는 것만 끝에 붙인 새 목록. present(정규화 집합)가 오면 그 기준으로
    이미 있는지 본다(여러 칸을 함께 볼 때). 붙인 태그는 present 에도 더한다."""
    pool = present if present is not None else {norm_tag(t) for t in tags}
    out = list(tags)
    for piece in split_target(target):
        n = norm_tag(piece)
        if n in pool:
            continue
        out.append(piece)
        pool.add(n)
    return out


def remove_from_tags(tags: list, target) -> list:
    """target 태그(쉼표로 여러 개 가능)와 정규화가 같은 태그를 모두 뺀 새 목록."""
    drop = {norm_tag(t) for t in split_target(target)}
    if not drop:
        return list(tags)
    return [t for t in tags if norm_tag(t) not in drop]


def replace_across(fields: list, condition: str, target) -> list:
    """태그 단위 교체 — 조건 태그(쉼표 AND 의 각 태그 = 앵커)를 대상 태그로 바꾼다.

    - 부분문자열이 아니라 **태그 하나**를 비교한다(norm_tag) — ``muscular→muscular male`` 이
      ``muscular male`` 을 ``muscular male male`` 로 망가뜨리지 않는다.
    - 여러 칸(fields)을 함께 본다: 모든 칸을 통틀어 첫 앵커 자리에 대상 태그를 넣고, 나머지
      앵커는 지운다. 대상 태그가 어느 칸에든 이미 있으면 새로 넣지 않는다(앵커만 지운다).
    - 그래서 두 번 적용해도 결과가 같다(멱등).
    반환: 칸별 새 목록(입력과 같은 순서).
    """
    anchors = set(condition_terms(condition))
    targets = split_target(target)
    if not anchors or not targets:
        return [list(f) for f in fields]
    present = {
        n for f in fields for t in f
        for n in (norm_tag(t),) if n and n not in anchors
    }
    placed = False
    result = []
    for f in fields:
        out = []
        for t in f:
            if norm_tag(t) in anchors:
                if not placed:
                    for piece in targets:
                        n = norm_tag(piece)
                        if n in present:
                            continue
                        out.append(piece)
                        present.add(n)
                    placed = True
                continue
            out.append(t)
        result.append(out)
    return result


def replace_in_tags(tags: list, condition: str, target) -> list:
    """한 칸짜리 :func:`replace_across`."""
    return replace_across([tags], condition, target)[0]


# ── 전역 조건식(config/cond_rules.json · Vue CondPromptModal) 엔진 ─────────────────────
#: 조건을 평가하고 add/remove/replace 가 닿는 포지티브 칸(네거티브 제외)
POSITIVE_FIELDS = ("character", "copyright", "prefix", "main", "suffix")
_INSERT_LOCATIONS = ("main", "prefix", "suffix")


def apply_prompt_rules(fields: dict, pos_rules, neg_rules) -> dict:
    """전역 조건부 규칙 적용 — 칸별 태그 목록을 받아 새 칸별 목록을 돌려준다(입력 불변).

    fields 키: character/copyright/prefix/main/suffix/neg (없으면 빈 칸).
    규칙 형식은 cond_rules.json 그대로: ``{condition, exists, target, action, location}``.

    - 조건: 포지티브 칸 전체 태그에 대한 쉼표 AND(:func:`multi_cond_met`). 규칙을 적용하기 전
      태그로 한 번 평가한다(규칙끼리 연쇄하지 않는다 — 1·2차 적용 사이에서만 연쇄).
    - add: location 칸 끝에 붙인다. 포지티브 어디에든 이미 있는 태그는 붙이지 않는다.
      after_condition 은 조건 첫 태그가 있는 칸의 그 태그 바로 뒤(없으면 본문 끝).
    - remove: 포지티브 모든 칸에서 대상 태그(쉼표로 여러 개)를 뺀다.
    - replace: 태그 단위(:func:`replace_across`), '없으면' 규칙은 바꿀 태그가 없어 건너뛴다.
    - 네거티브 규칙: 조건은 포지티브 기준, add/remove 는 neg 칸.
    두 번 적용해도 결과가 같다.
    """
    out = {k: list(v or []) for k, v in (fields or {}).items()}
    for key in POSITIVE_FIELDS + ("neg",):
        out.setdefault(key, [])

    def _pos_norms() -> set:
        return {n for f in POSITIVE_FIELDS for t in out[f] for n in (norm_tag(t),) if n}

    all_tags = _pos_norms()

    def _rule_parts(rule):
        if not isinstance(rule, dict):
            return None
        cond = str(rule.get("condition") or "")
        target = str(rule.get("target") or "").strip()
        if not cond.strip() or not target:
            return None
        exists = bool(rule.get("exists", True))
        if not multi_cond_met(cond, all_tags, exists):
            return None
        return cond, target, exists, (rule.get("action") or "add"), (rule.get("location") or "main")

    for rule in pos_rules or []:
        parts = _rule_parts(rule)
        if parts is None:
            continue
        cond, target, exists, action, location = parts
        if action == "add":
            present = _pos_norms()
            missing = [p for p in split_target(target) if norm_tag(p) not in present]
            if not missing:
                continue
            if location == "after_condition":
                terms = condition_terms(cond)
                placed = False
                for key in POSITIVE_FIELDS:
                    new = _insert_pieces_after(out[key], terms[0], missing)
                    if new is not None:
                        out[key] = new
                        placed = True
                        break
                if not placed:
                    out["main"] = add_to_tags(out["main"], missing, present)
            else:
                key = location if location in _INSERT_LOCATIONS else "main"
                out[key] = add_to_tags(out[key], missing, present)
        elif action == "remove":
            for key in POSITIVE_FIELDS:
                out[key] = remove_from_tags(out[key], target)
        elif action == "replace":
            if not exists:
                continue
            replaced = replace_across([out[k] for k in POSITIVE_FIELDS], cond, target)
            for key, tags in zip(POSITIVE_FIELDS, replaced):
                out[key] = tags

    for rule in neg_rules or []:
        parts = _rule_parts(rule)
        if parts is None:
            continue
        _cond, target, _exists, action, _location = parts
        if action == "add":
            out["neg"] = add_to_tags(out["neg"], target)
        elif action == "remove":
            out["neg"] = remove_from_tags(out["neg"], target)

    return out
