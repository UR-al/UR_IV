"""
Workflow Profiles — 체크포인트/VAE/TE/샘플러/CFG/LoRA/Prefix/Suffix 묶음 저장.

ANIMA용, SDXL용, Flux용 등 자주 쓰는 생성 환경을 통째로 저장하고
드롭다운 한 번에 전환.

저장 위치: ``config/profiles/<name>.json``
JSON 스키마 (version 2 — lora_stack weight 는 항상 **배율**):
{
    "name": "ANIMA Pony",
    "created_at": <unix>,
    "version": 2,
    "fields": {
        "model_combo": "anima_v3.safetensors",
        "vae_main_combo": "qwen_image_vae.safetensors",
        "te_main_input": "anima_baseV10_txt.safetensors",
        "sampler_combo": "DPM++ 2M",
        "scheduler_combo": "Karras",
        "steps_input": "28",
        "cfg_input": "5",
        "shift_input": "0",
        "width_input": "1024",
        "height_input": "1024",
        "prefix_prompt_text": "score_9, score_8_up, ...",
        "suffix_prompt_text": "blurry, ...",
        "neg_prompt_text": "lowres, ...",
    },
    "lora_stack": [
        {"name": "...", "weight": 0.8, "enabled": true, "triggerWords": [...]},
    ]
}

version 1 파일은 lora_stack 단위가 기록되지 않았다 — 부팅 복원 직후 저장하면 ui_prefs 의
정수 퍼센트(95)가, 스택을 편집한 뒤 저장하면 배율(0.95)이 들어갔다. 적용할 때
:func:`profile_lora_entries` 가 목록 전체 기준(|w|>3 이면 퍼센트)으로 정규화해 읽기 호환을
지킨다. 파일 자체는 고치지 않는다(다시 저장할 때 version 2 로 기록된다).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from core.lora_stack import UNIT_AUTO, UNIT_MULTIPLIER, normalize_lora_entries
from utils.app_logger import get_logger

_logger = get_logger("workflow_profile")

# 2: lora_stack weight 를 배율로 확정 기록. 1(이전): 단위 미기록 — 퍼센트/배율 혼재 가능.
PROFILE_VERSION = 2


def profile_lora_entries(profile_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """프로파일의 lora_stack 을 **배율 단위**로 읽는다 (파일은 건드리지 않음).

    version 2+ 는 배율 그대로. 그 이전(단위 미기록)은 목록 전체 기준으로 판정한다 —
    |weight|>3 인 항목이 하나라도 있으면 퍼센트로 보고 /100 (예: ANIMA.json 의 95/10/100).
    """
    data = profile_data if isinstance(profile_data, dict) else {}
    lora = data.get("lora_stack") or []
    if not isinstance(lora, list):
        return []
    try:
        version = int(data.get("version", 1) or 1)
    except (TypeError, ValueError):
        version = 1
    unit = UNIT_MULTIPLIER if version >= PROFILE_VERSION else UNIT_AUTO
    return normalize_lora_entries(lora, unit=unit)


# 프로파일에 포함될 위젯 ID 목록 (Vue widgets 키)
# 사용자가 자주 바꾸는 생성 세팅 위주
PROFILE_FIELDS: tuple[str, ...] = (
    "model_combo",
    "vae_main_combo",
    "te_main_input",
    "sampler_combo",
    "scheduler_combo",
    "steps_input",
    "cfg_input",
    "shift_input",
    "width_input",
    "height_input",
    "prefix_prompt_text",
    "suffix_prompt_text",
    "neg_prompt_text",
)


def _profiles_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    d = root / "config" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitize_name(name: str) -> str:
    """파일명 안전 — core/file_naming.py 공용 유틸 위임 (기존 폴백 유지)."""
    from core.file_naming import sanitize_filename
    return sanitize_filename(name, fallback="프로파일")


def _path(name: str) -> Path:
    return _profiles_dir() / f"{_sanitize_name(name)}.json"


def profile_saved_name(name: str) -> str:
    """``name`` 으로 저장하면 기록되는 프로파일 이름(파일 이름 규칙을 거친 것) — 알림에 보일 이름."""
    return _sanitize_name(name)


def _saved_path(name: str) -> Path:
    """``name`` 으로 저장·이름 변경하면 쓰이는 파일 — 기록되는 이름(:func:`profile_saved_name`)의 :func:`_path`.

    규칙은 한 번 더 거치면 결과가 바뀔 수 있다(두 번째에 지워지는 것: '/ Flux' → ' Flux' 의 앞 공백,
    'A'*63 + '. b' → 'A'*63 + '.' 처럼 64자에서 잘려 드러난 끝의 점·공백). 목록의 이름(= 기록된 이름)으로
    찾는 load/delete 는 ``_path(기록된 이름)`` 이므로, 쓰는 쪽과 충돌 검사도 반드시 이 경로를 본다
    (Vue 의 utils/profileFileName.profileFileKey 도 규칙을 두 번 거친다). ``_path(name)`` 을 바로 쓰면
    충돌 검사는 다른 파일을 보고 저장은 조용히 실패하거나, 이름 변경이 남의 파일 이름을 가리키게 된다.
    """
    return _path(_sanitize_name(name))


def _existing_file(path: Path) -> Path:
    """이미 있는 파일의 디스크 표기 — 대소문자만 다른 이름으로 찾았으면 그 실제 파일
    (utils/file_wildcard 의 _existing_file_name 과 같은 방식)."""
    try:
        entries = os.listdir(path.parent)
    except OSError:
        return path
    if path.name in entries:
        return path
    wanted = os.path.normcase(path.name)
    return next((path.parent / f for f in entries if os.path.normcase(f) == wanted), path)


def find_conflicting_profile(name: str) -> str | None:
    """``name`` 으로 저장하면 덮어쓰게 될 기존 프로파일의 이름 — 없으면 None.

    저장 파일 이름은 규칙(core/file_naming.sanitize_filename)을 거친다: 금지 문자(``\\/:*?"<>|``)와
    끝의 점·공백이 사라지고 64자에서 잘리며, Windows 는 대소문자를 가리지 않는다. 그래서 'Flux.' ·
    'flux' · 'Fl/ux' · 'Flux?' 는 모두 Flux.json 이다. 예전엔 Vue 가 목록과 글자 그대로만 비교하고
    저장은 늘 덮어써, 이런 이름이 확인 없이 기존 프로파일을 바꿔 버렸다.
    검사하는 파일은 :func:`save_profile` 이 쓰는 파일과 같아야 한다(:func:`_saved_path`).
    """
    path = _saved_path(name)
    if not path.exists():
        return None
    existing = _existing_file(path)
    try:
        data = json.loads(existing.read_text(encoding="utf-8"))
    except Exception:
        data = None
    stored = data.get("name") if isinstance(data, dict) else None
    return stored if isinstance(stored, str) and stored else existing.stem


# ─────────────────────────────────────────
# 조회 / 삭제
# ─────────────────────────────────────────

def list_profiles() -> list[dict[str, Any]]:
    """모든 프로파일 메타데이터 — 이름순. (전체 fields 없이 가벼움)"""
    result: list[dict[str, Any]] = []
    for p in sorted(_profiles_dir().glob("*.json"), key=lambda x: x.name.lower()):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        result.append({
            "name": data.get("name", p.stem),
            "created_at": data.get("created_at", 0),
            "model": (data.get("fields") or {}).get("model_combo", ""),
            "vae": (data.get("fields") or {}).get("vae_main_combo", ""),
        })
    return result


def load_profile(name: str) -> dict[str, Any] | None:
    """전체 프로파일 데이터 로드."""
    path = _path(name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _logger.exception(f"프로파일 로드 실패: {name}")
        return None


def delete_profile(name: str) -> bool:
    path = _path(name)
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        _logger.exception(f"프로파일 삭제 실패: {name}")
        return False


def rename_profile(old: str, new: str) -> bool:
    """이름 변경. new가 이미 있으면 False.

    ``old`` 는 목록의 이름(기록된 이름)이라 :func:`_path` 로, ``new`` 는 사용자가 친 이름이라 저장과 같은
    :func:`_saved_path` 로 찾는다 — 예전엔 '/ Flux' 가 ' Flux.json'(name ' Flux')으로 써져, 그 이름으로
    불러오기·삭제하면 다른 프로파일 Flux.json 을 건드렸다.
    """
    old_path = _path(old)
    new_path = _saved_path(new)
    if not old_path.is_file() or new_path.exists():
        return False
    try:
        data = json.loads(old_path.read_text(encoding="utf-8"))
        data["name"] = _sanitize_name(new)
        new_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        old_path.unlink()
        return True
    except Exception:
        _logger.exception(f"프로파일 이름 변경 실패: {old} → {new}")
        return False


# ─────────────────────────────────────────
# 저장 (현재 상태 → 프로파일)
# ─────────────────────────────────────────

def save_profile(name: str,
                 fields: dict[str, Any],
                 lora_stack: list[dict[str, Any]] | None = None,
                 overwrite: bool = False) -> bool:
    """``fields`` dict + ``lora_stack`` 리스트를 프로파일로 저장.

    :param fields: ``{widget_id: value}`` — PROFILE_FIELDS만 골라서 저장
    :param lora_stack: ``[{name, weight, enabled, triggerWords}, ...]``
    :param overwrite: False(기본)면 **같은 파일**의 프로파일이 있을 때 쓰지 않고 False 반환.
        '같은 파일'은 이름 규칙을 거친 경로로 본다 — 'Flux.' · 'flux'(Windows) 도 Flux.json 이다
        (:func:`find_conflicting_profile`). 덮어쓰기는 사용자가 확인했을 때만 True 로 부른다.
    """
    clean_name = _sanitize_name(name)
    if not clean_name:
        return False
    path = _saved_path(name)   # = _path(clean_name) — find_conflicting_profile 과 같은 파일
    if path.exists() and not overwrite:
        return False

    # fields는 화이트리스트에 있는 키만
    filtered = {k: ("" if v is None else str(v)) for k, v in fields.items()
                if k in PROFILE_FIELDS}
    payload = {
        "name": clean_name,
        "created_at": int(time.time()),
        "version": PROFILE_VERSION,
        "fields": filtered,
        # 호출부(collect_from_host)는 _vue_lora_entries(배율)를 넘긴다 — version 2 = 배율 확정
        "lora_stack": normalize_lora_entries(lora_stack or [], unit=UNIT_MULTIPLIER),
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except OSError:
        _logger.exception(f"프로파일 저장 실패: {clean_name}")
        return False


# ─────────────────────────────────────────
# 적용 (프로파일 → 호스트 widgets)
# ─────────────────────────────────────────

def apply_profile_to_host(profile_data: dict[str, Any], host: Any) -> dict[str, Any]:
    """프로파일을 호스트(GeneratorMainUI 등)에 적용.

    호스트가 가진 위젯 프록시를 통해 값 적용:
    - 'vae_main_combo' 같은 ComboBoxProxy는 setText (fallback 가능)
    - 'cfg_input', 'steps_input' 같은 SliderProxy/LineEditProxy는 setText
    - 'prefix_prompt_text' 같은 TextEditProxy는 setPlainText

    LoRA Stack은 :func:`profile_lora_entries` 로 배율 단위로 정규화한 뒤 호스트의
    ``_vue_lora_entries``(생성 LoRA 의 단일 소스)를 갱신하고 Vue로 push 한다
    (Vue 가 ui_prefs.loraStack 에 영속). 생성 텍스트는 생성 때마다 entries 에서 파생된다.

    :return: ``{applied: [...keys], skipped: [...]}``
    """
    applied: list[str] = []
    skipped: list[str] = []
    fields = (profile_data or {}).get("fields", {}) or {}

    for key, value in fields.items():
        proxy = getattr(host, key, None)
        if proxy is None:
            # ComboBox / SliderProxy의 일반적 widget_id 매핑
            skipped.append(key)
            continue
        try:
            # ComboBoxProxy / LineEditProxy / TextEditProxy / SliderProxy 모두 setText 지원
            if hasattr(proxy, "setPlainText"):
                proxy.setPlainText(str(value))
            elif hasattr(proxy, "setText"):
                proxy.setText(str(value))
            else:
                skipped.append(key)
                continue
            applied.append(key)
        except Exception:
            _logger.exception(f"프로파일 필드 적용 실패: {key}")
            skipped.append(key)

    # LoRA Stack 적용 — Vue 측의 set_lora_stack 액션 호출과 동일한 효과(배율 단위)
    lora = (profile_data or {}).get("lora_stack", []) or []
    if isinstance(lora, list):
        try:
            entries = profile_lora_entries(profile_data)
            host._vue_lora_entries = entries
            # Vue로 푸시 (loraStackLoaded 는 배율을 받아 정수 %로 한 번만 바꾼다)
            bridge = getattr(host, "vue_bridge", None)
            if bridge is not None and hasattr(bridge, "loraStackLoaded"):
                bridge.loraStackLoaded.emit(json.dumps(entries, ensure_ascii=False))
            applied.append("lora_stack")
        except Exception:
            _logger.exception("LoRA stack 적용 실패")
            skipped.append("lora_stack")

    return {"applied": applied, "skipped": skipped}


def collect_from_host(host: Any) -> dict[str, Any]:
    """호스트 현재 상태에서 프로파일 fields + lora_stack 추출.

    저장 시 호출 — Vue 위젯 값은 host.vue_bridge가 캐시함.
    """
    fields: dict[str, str] = {}
    for key in PROFILE_FIELDS:
        proxy = getattr(host, key, None)
        if proxy is None:
            continue
        try:
            if hasattr(proxy, "currentText"):
                fields[key] = proxy.currentText()
            elif hasattr(proxy, "toPlainText"):
                fields[key] = proxy.toPlainText()
            elif hasattr(proxy, "text"):
                fields[key] = proxy.text()
        except Exception:
            pass

    lora_stack = getattr(host, "_vue_lora_entries", None) or []
    if not isinstance(lora_stack, list):
        lora_stack = []
    # _vue_lora_entries 는 배율 단위 계약(set_lora_stack·부팅 복원 모두 정규화해 넣는다)
    return {"fields": fields,
            "lora_stack": normalize_lora_entries(lora_stack, unit=UNIT_MULTIPLIER)}
