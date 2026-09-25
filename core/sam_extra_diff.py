"""sam-extra script-info(라이브 또는 픽스처)와 앱 spec 을 비교하는 순수 함수 — Qt·네트워크 없음.

`/sdapi/v1/script-info` 의 인자 항목은 ``{"label", "value", "minimum", "maximum", "step", "choices"}``
이다. Forge 는 API 기본값을 ``script.ui()`` 를 다시 불러 만들므로(ui-config.json 무시) 이 값이
곧 API 기본값이다. 여기서는 그 값을 앱 spec(`core.anima_guidance` 의 ``(key, kind, default, extra)``
튜플, `core.sam3_args.SAM3_SPEC` 기본값)과 비교해 차이를 ``(key, field, app, live)`` 로 돌려준다.
어떤 차이를 허용할지는 호출자(계약 테스트의 KNOWN_DIFFS)가 정한다.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence

_NUMERIC_KINDS = ("float", "int")


def entries_by_title(entries: Iterable[Mapping]) -> dict[str, dict[bool, list]]:
    """script-info 목록 → {소문자 제목: {is_img2img: args}}."""
    out: dict[str, dict[bool, list]] = {}
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip().lower()
        if not name:
            continue
        out.setdefault(name, {})[bool(entry.get("is_img2img"))] = list(entry.get("args") or [])
    return out


def _choice_values(choices) -> tuple | None:
    if choices is None:
        return None
    out = []
    for item in choices:
        # Gradio 4 는 (표시 이름, 값) 쌍을 줄 수 있다 — 값만 비교한다.
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append(item[1])
        else:
            out.append(item)
    return tuple(out)


def shape_signature(args: Sequence[Mapping], runtime_indices: Iterable[int] = ()) -> list:
    """인자 목록의 '모양'(라벨·기본값·범위·step·선택지). 실행 시점 목록인 인덱스는 선택지·기본값을 뺀다."""
    runtime = set(runtime_indices)
    signature = []
    for index, arg in enumerate(args):
        arg = arg if isinstance(arg, Mapping) else {}
        runtime_arg = index in runtime
        signature.append([
            arg.get("label"),
            None if runtime_arg else arg.get("value"),
            arg.get("minimum"),
            arg.get("maximum"),
            arg.get("step"),
            None if runtime_arg else _choice_values(arg.get("choices")),
        ])
    return signature


def shape_hash(args: Sequence[Mapping], runtime_indices: Iterable[int] = ()) -> str:
    """`shape_signature` 의 짧은 해시(레지스트리에 고정해 두고 바뀌면 사람이 다시 본다)."""
    raw = json.dumps(shape_signature(args, runtime_indices), ensure_ascii=False, sort_keys=True,
                     default=str, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def same_value(app, live) -> bool:
    """기본값 비교: bool 은 bool 끼리, 숫자는 부동소수 오차 허용, None 과 '' 은 같게 본다."""
    if isinstance(app, bool) or isinstance(live, bool):
        return isinstance(app, bool) and isinstance(live, bool) and app is live
    if isinstance(app, (int, float)) and isinstance(live, (int, float)):
        return math.isclose(float(app), float(live), rel_tol=0.0, abs_tol=1e-9)
    if app in (None, "") and live in (None, ""):
        return True
    return app == live


def compare_positional_spec(spec: Sequence[tuple], live_args: Sequence[Mapping], *, prefix: str,
                            runtime_indices: Iterable[int] = ()) -> list[tuple]:
    """위치 인자 spec 과 script-info 인자를 인덱스끼리 비교한다.

    반환: ``(키(접두사 뺀 이름), 필드, 앱 값, 라이브 값)`` 목록. 필드는 argc/default/minimum/maximum/choices.
    라이브가 범위를 주지 않는 인자(Textbox·Number)는 범위를 비교하지 않는다.
    """
    runtime = set(runtime_indices)
    diffs: list[tuple] = []
    if len(spec) != len(live_args):
        diffs.append(("*", "argc", len(spec), len(live_args)))
    for index, (key, kind, default, extra) in enumerate(spec):
        if index >= len(live_args):
            break
        name = key[len(prefix):] if prefix and key.startswith(prefix) else key
        arg = live_args[index] if isinstance(live_args[index], Mapping) else {}
        if index not in runtime and not same_value(default, arg.get("value")):
            diffs.append((name, "default", default, arg.get("value")))
        if kind in _NUMERIC_KINDS and extra:
            for field_name, app_bound in (("minimum", extra[0]), ("maximum", extra[1])):
                live_bound = arg.get(field_name)
                if live_bound is not None and not same_value(app_bound, live_bound):
                    diffs.append((name, field_name, app_bound, live_bound))
        if kind == "choice" and index not in runtime:
            live_choices = _choice_values(arg.get("choices"))
            if live_choices is not None and tuple(extra or ()) != live_choices:
                diffs.append((name, "choices", tuple(extra or ()), live_choices))
    return diffs


def _plain(value):
    """AST 값(튜플)과 JSON 값(리스트)을 같은 모양으로."""
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


def same_ui_field(field: str, source, live) -> bool:
    """ui() 컴포넌트 필드(AST) = script-info 필드(픽스처)? 선택지는 (표시, 값) 쌍이면 값끼리 본다."""
    if field == "choices":
        return _choice_values(_plain(source)) == _choice_values(_plain(live))
    return same_value(_plain(source), _plain(live))


def compare_ui_components(components: Sequence[Mapping], live_args: Sequence[Mapping], *,
                          runtime_indices: Iterable[int] = ()) -> tuple[list[tuple], dict[int, tuple]]:
    """AST 로 읽은 ui() 컴포넌트 필드와 script-info 인자를 인덱스끼리 비교한다.

    반환 ``(차이 [(index, field, 소스 값, 픽스처 값)], 못 읽은 필드 {index: (field, …)})``.
    못 읽은 필드 = 픽스처에는 값이 있는데(None 이 아님) AST 로는 정적으로 풀지 못한 필드 — 비교에서 빠지므로
    호출자가 목록으로 관리한다. 실행 시점 목록인 인덱스는 value·choices 를 보지 않는다.
    """
    runtime = set(runtime_indices)
    diffs: list[tuple] = []
    unread: dict[int, tuple] = {}
    for index, fields in enumerate(components):
        if index >= len(live_args):
            break
        arg = live_args[index] if isinstance(live_args[index], Mapping) else {}
        skipped = ("value", "choices") if index in runtime else ()
        missing = []
        for key in ("label", "value", "minimum", "maximum", "step", "choices"):
            if key in skipped:
                continue
            if key not in fields:
                if arg.get(key) is not None:
                    missing.append(key)
            elif not same_ui_field(key, fields[key], arg.get(key)):
                diffs.append((index, key, fields[key], arg.get(key)))
        if missing:
            unread[index] = tuple(missing)
    return diffs, unread


def compare_defaults(app_defaults: Mapping, live_values: Mapping, *, skip: Iterable[str] = ()) -> list[tuple]:
    """dict 형태 기본값 비교(SAM3 state, Anima38/DoRA 의 ARG_DEFAULTS). 키 집합 차이도 돌려준다."""
    skipped = set(skip)
    diffs: list[tuple] = []
    app_keys = set(app_defaults) - skipped
    live_keys = set(live_values) - skipped
    for key in sorted(app_keys - live_keys):
        diffs.append((key, "missing_live", app_defaults[key], None))
    for key in sorted(live_keys - app_keys):
        diffs.append((key, "missing_app", None, live_values[key]))
    for key in sorted(app_keys & live_keys):
        if not same_value(app_defaults[key], live_values[key]):
            diffs.append((key, "default", app_defaults[key], live_values[key]))
    return diffs
