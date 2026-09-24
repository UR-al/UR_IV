"""테스트의 무거운 선택 의존성(torch) — 표시와 지연 import 헬퍼.

torch 는 import 만으로 느린 통합 테스트 모듈 하나만큼 든다(실측: run_tests.SLOW_MODULES 주석).
테스트 모듈 하나가 최상위에서 ``import torch`` 하면, 그 테스트를 거르더라도 discovery 가 모듈을
import 하므로 .py 편집마다 도는 ``run_tests.py --quick``(PostToolUse 훅)이 매번 그만큼 느려졌다.

규칙 (tests/test_optional_deps.py 가 AST 로 지킨다):
- torch 가 필요한 테스트 클래스·메서드에 ``@requires_torch`` 를 붙인다.
    · torch 가 설치돼 있지 않으면 skip — 시스템 python 등에서 ImportError 로 깨지는 대신.
    · ``run_tests.py --quick`` 은 표시된 테스트를 거른다(``--with-torch`` 로 되살림). 훅은
      ``run_tests.TORCH_TEST_SOURCES`` 나 표시가 든 테스트 파일을 고쳤을 때 되살린다.
- torch 는 모듈 최상위에서 import 하지 않는다. 테스트 안에서 ``load_torch()`` 로, 모듈 전역
  ``torch`` 를 쓰는 헬퍼가 많으면 표시된 클래스의 ``setUpClass`` 에서 ``bind_torch(globals())`` 로.
  ``setUpModule`` 에서는 부르지 않는다 — 같은 모듈의 torch 없는 테스트가 quick 에서 돌 때도 실행된다.
"""
import functools
import importlib.util
import unittest

from run_tests import REQUIRES_ATTR, TORCH


@functools.lru_cache(maxsize=None)
def torch_installed() -> bool:
    """torch 가 설치돼 있나 — find_spec 은 패키지를 찾기만 하고 import 하지 않는다(싸다)."""
    try:
        return importlib.util.find_spec(TORCH) is not None
    except (ImportError, ValueError):
        return False


def requires_torch(test_item):
    """클래스·테스트 메서드에 'torch 필요' 표시를 붙인다. torch 가 없으면 skip 한다.

    표시(REQUIRES_ATTR)는 run_tests.required_deps 가 읽어 --quick 에서 거른다. 하위 클래스는
    상속으로 표시를 물려받는다.
    """
    marks = frozenset(getattr(test_item, REQUIRES_ATTR, None) or ()) | {TORCH}
    setattr(test_item, REQUIRES_ATTR, marks)
    return unittest.skipUnless(torch_installed(), "torch 미설치 — torch 텐서 테스트 건너뜀")(test_item)


def load_torch():
    """torch 를 지금 import 해 돌려준다.

    설치돼 있지 않으면 SkipTest. 설치돼 있는데 import 가 실패하면(DLL 오류 등) 그 예외를 그대로
    올린다 — 깨진 설치를 skip 으로 숨기지 않는다.
    """
    if not torch_installed():
        raise unittest.SkipTest("torch 미설치 — torch 텐서 테스트 건너뜀")
    import torch  # 지연 import — 이 헬퍼의 목적이다

    return torch


def bind_torch(namespace: dict):
    """모듈 전역 ``torch`` 를 채운다 — 표시된 클래스의 setUpClass 에서 ``bind_torch(globals())``."""
    torch = load_torch()
    namespace[TORCH] = torch
    return torch
