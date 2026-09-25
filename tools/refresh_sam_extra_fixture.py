"""sam-extra 계약 픽스처 갱신 — 실행 중인 Forge 에 읽기 전용 GET 만 보낸다.

    venv\\Scripts\\python.exe tools\\refresh_sam_extra_fixture.py [--api-url http://127.0.0.1:7860]

- ``GET /sdapi/v1/script-info`` 에서 sam-extra 스크립트 항목만 골라 `tests/fixtures/sam_extra_script_info.json`
  에 저장한다. AST 로는 읽을 수 없는 기본값·범위·선택지(실행 시점 목록 포함)의 1차 출처다.
- ``GET /sdapi/v1/extensions`` 에서 확장 폴더 이름·브랜치·커밋을 함께 적는다(작업 트리가 dirty 이면
  커밋 해시가 실제 코드를 대표하지 못한다 — 버전은 로컬 설치의 ``sam3ext/__version__.py`` 로 적는다).
- POST·옵션 변경·생성은 하지 않는다. GPU 를 쓰지 않는다.

- 응답이 목록이 아니거나(인증 오류 등) sam-extra 스크립트가 하나도 없으면(확장이 꺼진 Forge, 관리형 Forge 등)
  저장하지 않고 0 이 아닌 값으로 끝난다 — 빈 픽스처로 덮어쓰지 않는다. 파일은 임시 파일에 쓴 뒤 바꿔 끼운다.

주의: Forge 는 시작할 때 확장 코드를 읽는다. 확장을 고친 뒤 Forge 를 다시 시작하지 않았으면 라이브 값은
옛 코드 기준이다. 저장 후 ``run_tests.py --include tests.test_sam_extra_contract`` 로 확인한다.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_API_URL = "http://127.0.0.1:7860"
DEFAULT_OUT = os.path.join(ROOT, "tests", "fixtures", "sam_extra_script_info.json")
FIXTURE_SCHEMA = 1


def _get_json(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 사용자가 준 로컬 URL
        return json.loads(response.read().decode("utf-8"))


def _known_titles() -> set[str]:
    """레지스트리 제목 + (설치돼 있으면) 확장 소스의 AlwaysVisible 제목, 소문자."""
    titles: set[str] = set()
    try:
        from core.sam_extra_contract import SCRIPTS
        titles.update(title.lower() for title in SCRIPTS)
    except Exception as exc:  # 레지스트리가 깨져도 픽스처는 받을 수 있어야 한다
        print(f"[warn] 레지스트리를 읽지 못함: {exc}", file=sys.stderr)
    local = _local_extension()
    if local is not None:
        titles.update(title.lower() for title in local.scripts())
    return titles


def _local_extension():
    try:
        from core.sam_extra_scan import ExtensionSource, find_installed_extension
        root = find_installed_extension()
        return ExtensionSource(root) if root is not None else None
    except Exception as exc:
        print(f"[warn] 로컬 확장을 읽지 못함: {exc}", file=sys.stderr)
        return None


def _extension_entry(extensions) -> dict | None:
    from core.sam3_assets import SAM3_EXTENSION_DIR_NAMES
    for item in extensions or ():
        if isinstance(item, dict) and str(item.get("name") or "") in SAM3_EXTENSION_DIR_NAMES:
            return {key: item.get(key) for key in ("name", "branch", "commit_hash", "commit_date", "version")
                    if key in item}
    return None


def build_fixture(script_info, extensions, *, api_url: str, titles: set[str], ext_version: str | None,
                  source: str = "GET /sdapi/v1/script-info (tools/refresh_sam_extra_fixture.py)") -> dict:
    wanted = [entry for entry in script_info if str(entry.get("name") or "").strip().lower() in titles]
    wanted.sort(key=lambda entry: (str(entry.get("name")), bool(entry.get("is_img2img"))))
    return {
        "schema": FIXTURE_SCHEMA,
        "source": source,
        "captured_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "api_url": api_url,
        "ext_version": ext_version,
        "extension": _extension_entry(extensions),
        "scripts": wanted,
    }


def _read_json_file(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _utf8_stdio() -> None:
    """콘솔·파이프가 cp949 여도 한국어·'—' 출력(--help 포함)에서 죽지 않게."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def fixture_problem(script_info, titles: set[str]) -> str:
    """저장하면 안 되는 응답이면 그 이유, 괜찮으면 ''."""
    if not isinstance(script_info, list):
        detail = json.dumps(script_info, ensure_ascii=False)[:200]
        return f"script-info 응답이 목록이 아니다(인증 오류·다른 서버일 수 있다): {detail}"
    names = {str(entry.get("name") or "").strip().lower() for entry in script_info if isinstance(entry, dict)}
    if not names & titles:
        return ("script-info 에 sam-extra 스크립트가 하나도 없다 — 이 Forge 에 확장이 없거나 꺼져 있다"
                f"(스크립트 {len(names)}개). --api-url 이 맞는지 확인하라")
    return ""


def write_fixture_atomic(fixture: dict, out: str) -> None:
    """같은 폴더의 임시 파일에 다 쓴 뒤 os.replace — 도중에 실패해도 기존 픽스처가 반쯤 지워지지 않는다."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sam_extra_fixture.", suffix=".tmp", dir=os.path.dirname(out))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(fixture, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None) -> int:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--from-file", default="",
                        help="Forge 가 꺼져 있을 때: 예전에 저장한 GET /sdapi/v1/script-info 응답 JSON 으로 만든다")
    parser.add_argument("--extensions-file", default="",
                        help="--from-file 과 함께: 저장해 둔 GET /sdapi/v1/extensions 응답 JSON")
    args = parser.parse_args(argv)
    api = args.api_url.rstrip("/")
    source = "GET /sdapi/v1/script-info (tools/refresh_sam_extra_fixture.py)"
    if args.from_file:
        script_info = _read_json_file(args.from_file)
        extensions = _read_json_file(args.extensions_file) if args.extensions_file else []
        saved_at = _dt.datetime.fromtimestamp(os.path.getmtime(args.from_file), _dt.timezone.utc)
        source = (f"저장된 GET /sdapi/v1/script-info 응답 {os.path.basename(args.from_file)} "
                  f"(받은 시각 {saved_at.replace(microsecond=0).isoformat()}, "
                  "tools/refresh_sam_extra_fixture.py --from-file)")
    else:
        try:
            script_info = _get_json(f"{api}/sdapi/v1/script-info", args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"[error] GET {api}/sdapi/v1/script-info 실패: {exc}", file=sys.stderr)
            print("        Forge 가 꺼져 있으면 저장해 둔 응답으로 --from-file 을 쓸 수 있다.", file=sys.stderr)
            return 2
        try:
            extensions = _get_json(f"{api}/sdapi/v1/extensions", args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"[warn] GET {api}/sdapi/v1/extensions 실패: {exc}", file=sys.stderr)
            extensions = []
    titles = _known_titles()
    problem = fixture_problem(script_info, titles)
    if problem:
        print(f"[error] {problem}", file=sys.stderr)
        print(f"        픽스처를 저장하지 않았다: {os.path.abspath(args.out)}", file=sys.stderr)
        return 3
    local = _local_extension()
    fixture = build_fixture(script_info, extensions, api_url=api, titles=titles,
                            ext_version=local.version() if local is not None else None, source=source)
    captured = {str(entry.get("name") or "").strip().lower() for entry in fixture["scripts"]}
    try:
        from core.sam_extra_contract import SCRIPTS
        missing = sorted(title for title in SCRIPTS if title.lower() not in captured)
    except Exception:  # 레지스트리가 깨져도 픽스처는 받을 수 있어야 한다
        missing = []
    if missing:
        print(f"[warn] 레지스트리에 있는데 응답에 없는 스크립트: {', '.join(missing)} — 계약 테스트가 사라진 항목으로 "
              "보고한다", file=sys.stderr)
    out = os.path.abspath(args.out)
    write_fixture_atomic(fixture, out)
    names = sorted({str(entry.get("name")) for entry in fixture["scripts"]})
    print(f"저장: {out}")
    print(f"스크립트 {len(names)}개 (항목 {len(fixture['scripts'])}개, txt2img+img2img): {', '.join(names)}")
    print(f"확장: {fixture['extension']} · 로컬 버전: {fixture['ext_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
