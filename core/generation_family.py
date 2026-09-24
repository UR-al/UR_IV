# core/generation_family.py
"""T2I 생성 family 콤보(Standard / Krea2)의 항목과 판정 — 순수 로직 (Qt 비의존).

ComboBoxProxy 는 Vue 에서 온 값을 항목 목록과 **대소문자까지 정확히** 비교한다
(widget_proxies.ComboBoxProxy._on_vue_changed). 그래서 Python 항목은 Vue 옵션 라벨
(frontend/src/components/PromptPanel.vue 의 generationFamilyItems)과 글자 그대로 같아야
한다. 예전엔 Python 이 'STANDARD'/'KREA2', Vue 가 'Standard'/'Krea2' 라서 Vue 에서 고른
Krea2 가 무시되고, 화면만 Krea2(체크포인트 숨김, steps 8·CFG 1)인 채 숨은 Standard
체크포인트로 생성됐다. tests/test_generation_family.py 가 두 목록의 일치를 고정한다.
"""
from __future__ import annotations

STANDARD_LABEL = "Standard"
KREA2_LABEL = "Krea2"

# PromptPanel.vue generationFamilyItems 와 순서·철자·대소문자까지 같아야 한다.
GENERATION_FAMILY_ITEMS: tuple[str, ...] = (STANDARD_LABEL, KREA2_LABEL)


def is_krea2_family(value: object) -> bool:
    """콤보 값(라벨)·저장값(대문자) 어느 쪽이든 Krea2 인지 — 판정은 항상 대소문자 무시."""
    return str(value or "").strip().upper() == KREA2_LABEL.upper()


# seed 한도 — Krea2 는 앱 replay 범위(32비트)만 받는다(core/krea2_generation._normalise_seed,
# tests/test_krea2_generation 이 의도로 고정). 생성 API 검증과 Creator 의 krea2_* 난수 seed 도
# 같은 한도를 써야, 받아 놓고 실행 단계에서 거부하거나 앱이 재현 못 하는 seed 를 만들지 않는다.
KREA2_SEED_MAX = 0xFFFFFFFF
STANDARD_SEED_MAX = (1 << 64) - 1


def seed_max_for_family(family: object) -> int:
    """생성 family('krea2'/'standard'/콤보 라벨) → 허용 seed 상한."""
    return KREA2_SEED_MAX if is_krea2_family(family) else STANDARD_SEED_MAX
