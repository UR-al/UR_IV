"""requirements.txt 드리프트 가드 — 코드가 실제로 import 하는 것과 선언이 어긋나지 않게.

- core/gpu_stats.py 의 `import pynvml` 은 NVIDIA 공식 배포판 nvidia-ml-py 가 제공한다.
  선언이 없으면 ultralytics 가 우연히 끌어올 때만 설치되고, 빠지면 VRAM 폴링이 매초
  nvidia-smi 프로세스 spawn 으로 조용히 떨어진다. 옛 비공식 'pynvml' 배포판은 같은 모듈을 덮는다.
- sam3/model/edt.py 는 sam3 메타데이터에 없는 triton 을 무조건 import 한다 → triton-windows 필수.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from core.check_requirements import _parse_requirements

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"


def _requirements() -> dict[str, Requirement]:
    out = {}
    for name, spec in _parse_requirements(str(REQUIREMENTS)):
        out[name.lower().replace("_", "-")] = Requirement(spec)
    return out


class RequirementsManifestTests(unittest.TestCase):
    def test_every_line_is_a_valid_requirement(self) -> None:
        for name, spec in _parse_requirements(str(REQUIREMENTS)):
            with self.subTest(requirement=spec):
                self.assertEqual(Requirement(spec).name.lower(), name.lower())

    def test_pynvml_module_comes_from_the_official_distribution(self) -> None:
        source = (ROOT / "core" / "gpu_stats.py").read_text(encoding="utf-8")
        self.assertIn("import pynvml", source)
        reqs = _requirements()
        self.assertIn("nvidia-ml-py", reqs)
        self.assertNotIn("pynvml", reqs, "옛 비공식 pynvml 배포판은 nvidia-ml-py 의 모듈을 덮어쓴다")

    def test_triton_is_declared_for_windows_sam3(self) -> None:
        reqs = _requirements()
        self.assertIn("sam3", reqs)
        triton = reqs.get("triton-windows")
        self.assertIsNotNone(triton, "sam3/model/edt.py 가 triton 을 무조건 import 한다")
        self.assertIsNotNone(triton.marker)
        self.assertTrue(triton.marker.evaluate({"sys_platform": "win32"}))
        self.assertFalse(triton.marker.evaluate({"sys_platform": "linux"}))

    def test_transitive_floors_are_not_looser_than_upstream_declarations(self) -> None:
        """sam3 0.1.4 / rembg 2.0.78 이 선언한 floor 보다 느슨하면 적어 둘 의미가 없다."""
        upstream = {"timm": "1.0.17", "iopath": "0.1.10", "einops": "0.8.0", "pymatting": "1.1.14"}
        reqs = _requirements()
        for name, floor in upstream.items():
            with self.subTest(package=name):
                self.assertIn(name, reqs)
                floors = [Version(s.version) for s in reqs[name].specifier if s.operator == ">="]
                self.assertTrue(floors, f"{name} 에 >= floor 가 없다")
                self.assertGreaterEqual(max(floors), Version(floor),
                                        f"{name} floor 가 upstream 선언({floor})보다 느슨하다")


if __name__ == "__main__":
    unittest.main()
