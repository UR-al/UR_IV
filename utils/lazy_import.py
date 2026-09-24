# utils/lazy_import.py
"""첫 사용 때 실제 모듈을 import 하는 대리 모듈 — 무거운 확장 모듈(cv2 등)의 지연 로드.

왜 필요한가: 창이 뜨기 전(_setup_ui)에 import 되는 모듈이 최상단에서 ``import cv2`` 를 하면
OpenCV 로드(콜드 기동 수백 ms)가 창 표시 임계 경로에 얹힌다(tests/test_startup_imports.py 가
막는다). cv2 를 여러 메서드에서 쓰는 모듈이라면 메서드마다 import 를 옮기는 대신 모듈 전역
이름을 이 대리 객체로 바꾼다::

    from utils.lazy_import import lazy_module
    cv2 = lazy_module("cv2")      # 여기서는 아무것도 로드하지 않는다
    ...
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 첫 속성 접근에서 import

규칙
- 속성 읽기·쓰기·삭제는 모두 실제 모듈로 전달한다. 그래서 ``mock.patch("pkg.mod.cv2.imread")``
  처럼 대리 객체를 거친 패치도 일반 ``import cv2`` 때와 똑같이 실제 모듈을 패치한다.
- import 는 ``importlib.import_module`` 이 하므로 여러 스레드가 동시에 처음 접근해도 모듈
  import 락이 한 번만 실행되게 한다. 로드한 모듈은 대리 객체에 캐시한다.
- 모듈 최상단·클래스 본문·기본 인자처럼 import 시점에 평가되는 자리에서 속성을 읽으면 그 순간
  로드된다(동작은 맞지만 지연 효과가 사라진다) — 그런 자리에는 쓰지 않는다.

현재 상태 — 운영 코드 사용처 0, 일부러 남겨 둔 도구
- 처음이자 유일한 사용처였던 숨은 PyQt 에디터 tabs/editor/* 는 은퇴해 삭제됐다(감사 #60).
  지금 이 모듈을 import 하는 것은 tests/test_lazy_import.py 뿐이다.
- 그래도 지우지 않는다: tests/test_startup_imports.py 의 실패 메시지가 '쓰는 메서드 안에서 import'
  와 함께 이 모듈을 기동 import 회귀의 해결책으로 안내한다. 한 모듈이 cv2 를 수십 곳에서 쓰면
  메서드마다 import 를 옮기는 것보다 이쪽이 안전하다.
- 없애기로 하면 이 파일·tests/test_lazy_import.py 를 지우고 test_startup_imports.py 의 모듈
  docstring 과 실패 메시지에서 이 모듈 안내를 함께 뺀다.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = ["LazyModule", "lazy_module"]


class LazyModule:
    """``name`` 모듈의 대리 객체. 첫 속성 접근에서 import 한다."""

    __slots__ = ("_lazy_name", "_lazy_module")

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_lazy_name", str(name))
        object.__setattr__(self, "_lazy_module", None)

    def _load(self) -> ModuleType:
        module = object.__getattribute__(self, "_lazy_module")
        if module is None:
            module = importlib.import_module(object.__getattribute__(self, "_lazy_name"))
            object.__setattr__(self, "_lazy_module", module)
        return module

    @property
    def is_loaded(self) -> bool:
        """이 대리 객체가 실제 모듈을 이미 불러왔는가 (진단·테스트용)."""
        return object.__getattribute__(self, "_lazy_module") is not None

    def __getattr__(self, attr: str) -> Any:
        # 슬롯·메서드처럼 대리 객체 자체에 있는 이름은 여기 오지 않는다.
        return getattr(self._load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._load(), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(self._load(), attr)

    def __dir__(self) -> list[str]:
        return dir(self._load())

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_lazy_name")
        state = "loaded" if self.is_loaded else "not loaded"
        return f"<lazy module {name!r} ({state})>"


def lazy_module(name: str) -> LazyModule:
    """``name`` 을 처음 쓸 때 import 하는 대리 모듈을 돌려준다(이미 import 됐어도 안전)."""
    return LazyModule(name)
