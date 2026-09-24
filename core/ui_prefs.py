# core/ui_prefs.py
"""config/ui_prefs.json 의 경로와 읽기 — 파일 하나의 주인.

예전엔 이 파일을 여는 곳이 약 15곳이었고, 그중 10곳은 ``os.path.join(..., 'config',
'ui_prefs.json')`` 을 손으로 조립했고 4곳은 ``json.load`` 로 마이그레이션·정규화를 건너뛰었다.
경로 정책(core.storage_paths)이나 키 마이그레이션(core.config_migration)을 바꾸면 그쪽만
조용히 어긋난다. 그래서:

- 경로는 :func:`ui_prefs_path` 하나 — 캐시하지 않는다(테스트가 storage_paths 를 바꿔 끼울 수
  있고, 호출 비용은 파일 읽기보다 작다).
- 일반 읽기는 :func:`read_ui_prefs` — 매번 새 dict, mtime 캐시 없음(완료 시점 재읽기처럼
  '지금 값'이 필요한 호출부가 있다). 로더는 호출 시점에 ``config_migration.load_ui_prefs`` 를
  모듈 속성으로 찾는다(테스트가 그 함수를 바꿔 끼운다).
- 마이그레이션 부작용(.bak)을 일부러 피하는 스냅숏 reader(core.ai_assist_instructions)나
  워커 스레드용 mtime 캐시(core.forge_output_policy 등)는 로더는 따로 두고 경로 이름만 공유한다
  (:data:`UI_PREFS_NAME`).

tests/test_ui_prefs_single_source.py 가 ``'ui_prefs.json'`` 리터럴의 재등장을 막는다.
"""
from __future__ import annotations

from typing import Mapping, Optional

#: config 디렉터리 안의 파일 이름 — 이 리터럴은 이 모듈에만 있어야 한다.
UI_PREFS_NAME = "ui_prefs.json"


def ui_prefs_path() -> str:
    """현재 저장소 정책(core.storage_paths)이 가리키는 ui_prefs.json 절대 경로."""
    from core.storage_paths import config_file

    return str(config_file(UI_PREFS_NAME))


def read_ui_prefs(path: Optional[str] = None) -> dict:
    """ui_prefs 전체를 새 dict 로 — 마이그레이션·정규화를 거친 값. 실패하면 ``{}``."""
    try:
        from core import config_migration

        target = path if path is not None else ui_prefs_path()
        prefs = config_migration.load_ui_prefs(target)
    except Exception:
        return {}
    return dict(prefs) if isinstance(prefs, dict) else {}


def ollama_unload_target(prefs: Optional[Mapping]) -> Optional[tuple[str, str]]:
    """'생성 전 Ollama 언로드'를 해야 하면 ``(url, model)``, 아니면 None.

    수동 생성(generator_generation._maybe_unload_ollama)과 Creator(creator_actions) 가 같은
    규칙을 쓴다: ``ollamaUnloadOnGen`` 이 켜져 있고 모델 이름이 있어야 한다. URL 이 비면
    기본 주소다. 실제로 내릴 설치 모델은 호출부가 ``unload_configured_model`` 로 고른다.
    """
    if not isinstance(prefs, Mapping) or not prefs.get("ollamaUnloadOnGen"):
        return None
    model = str(prefs.get("ollamaModel") or "").strip()
    if not model:
        return None
    from core.ollama_client import DEFAULT_OLLAMA_URL

    url = str(prefs.get("ollamaUrl") or "").strip() or DEFAULT_OLLAMA_URL
    return url.rstrip("/"), model
