"""생성 요청의 숫자 칸 파싱 (Qt 비의존 순수 로직).

Vue 는 steps·cfg·seed·denoising 같은 값을 숫자, 문자열('20'·'-1'), null(``parseInt`` 가 NaN 이면
JSON 에 null)로 보낸다. 잘못된 값은 기본값으로, 범위를 벗어난 값은 잘라서 쓴다.
``core/inpaint_payload``·``core/i2i_payload`` 가 같은 규칙을 쓰도록 한 곳에 둔다.
"""
from __future__ import annotations

import math

DEFAULT_SEED = -1
MAX_SEED = 2 ** 32 - 1


def finite_float(value, default: float) -> float:
    """숫자로 읽히고 유한하면 그 값, 아니면 ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clamped_int(value, default: int, lo: int, hi: int) -> int:
    """'20'·20.0·' 20 ' → 20 을 lo..hi 로 자른다. 읽을 수 없으면(None·''·'abc'·NaN) ``default``."""
    try:
        out = int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(lo, min(hi, out))


def clamped_float(value, default: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, finite_float(value, default)))


def parse_seed(value) -> int:
    """'-1'·''·잘못된 값 → -1(랜덤), 그 밖은 0..2^32-1 로 자른다."""
    return clamped_int(value, DEFAULT_SEED, -1, MAX_SEED)


class SeedRangeError(ValueError):
    """정수로 읽히는 seed 가 허용 상한을 넘었다 — 자르면 다른 seed 로 조용히 생성된다."""

    def __init__(self, seed: int, maximum: int):
        super().__init__(f"seed {seed} > {maximum}")
        self.seed = seed
        self.maximum = maximum


def _seed_integer(value):
    """seed 칸 값 → 정수(큰 수도 정확히), 읽을 수 없으면(None·''·'abc'·NaN·bool) None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)          # '18446744073709551615' 도 float 을 거치지 않아 정확하다
    except ValueError:
        pass
    number = finite_float(text, math.nan)
    if not math.isfinite(number):
        return None
    return int(number)            # '12.0'·'1e3' — clamped_int 와 같이 소수는 버린다


def parse_seed_checked(value, maximum: int) -> int:
    """seed 칸 → 요청 seed. 비었거나 숫자가 아니거나 음수면 -1(랜덤).

    ``maximum`` 을 넘는 seed 는 ``SeedRangeError`` — :func:`parse_seed` 처럼 잘라 쓰면
    사용자가 적은 것과 다른 seed 로 알리지 않고 생성된다(T2I 는 core/payload_validator 가,
    생성 API 는 family 별 상한으로 거부한다).
    """
    seed = _seed_integer(value)
    if seed is None or seed < 0:
        return DEFAULT_SEED
    if seed > maximum:
        raise SeedRangeError(seed, maximum)
    return seed


__all__ = [
    "DEFAULT_SEED",
    "MAX_SEED",
    "SeedRangeError",
    "clamped_float",
    "clamped_int",
    "finite_float",
    "parse_seed",
    "parse_seed_checked",
]
