"""생성 프리셋(presets/*.json) — 저장·미리보기·불러오기·공유가 같은 키 목록을 쓴다(audit #154).

예전 저장은 ``_build_settings_dict()`` 전체(82키: 백엔드 URL·단축키·테마·폰트까지)를 파일에
썼고 미리보기도 그 전부를 보여 줬지만, 불러오기는 프롬프트와 steps/cfg/seed/w/h/model/sampler
15개만 복원했다. 이제 :data:`PRESET_KEYS` 한 벌을 저장·미리보기·불러오기·가져오기에 똑같이
적용한다. 옛 전체 스냅샷 프리셋도 읽을 때 이 목록으로 걸러 쓰므로 그대로 동작한다.

Qt 없는 순수 모듈 — 위젯 적용은 ui/generation_settings_apply.py.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from core.file_naming import sanitize_filename
from utils.atomic_json import atomic_write_json, load_json_safe

#: 프롬프트 칸(포지티브/네거티브/로컬 제외).
PROMPT_KEYS: tuple[str, ...] = (
    'char_count', 'character', 'copyright', 'artist',
    'prefix_prompt', 'main_prompt', 'suffix_prompt', 'negative_prompt',
    'exclude_prompt_local',
)

#: 생성 파라미터 — 모델·모듈·샘플링·해상도·Hires·ADetailer·SAM3·Anima 가이던스.
GENERATION_KEYS: tuple[str, ...] = (
    'model', 'vae_main', 'te_main', 'sampler', 'scheduler',
    'steps', 'cfg', 'shift', 'seed', 'width', 'height',
    'random_res_enabled', 'random_resolutions',
    'hires_enabled', 'hires_upscaler', 'hires_steps', 'hires_denoising', 'hires_scale',
    'hires_cfg', 'hires_checkpoint', 'hires_sampler', 'hires_scheduler',
    'hires_prompt', 'hires_neg_prompt',
    'adetailer_enabled', 'adetailer_slot1_enabled', 'adetailer_slot2_enabled',
    'adetailer_slot1', 'adetailer_slot2',
    'sam3_enabled', 'sam3_settings',
    'anima_guidance_settings',
)

PRESET_KEYS: tuple[str, ...] = PROMPT_KEYS + GENERATION_KEYS
_PRESET_KEY_SET = frozenset(PRESET_KEYS)

#: 공유 파일(JSON) 형식 표식.
BUNDLE_FORMAT = 'ai-studio-generation-presets'
BUNDLE_VERSION = 1
MAX_BUNDLE_PRESETS = 1000


class PresetError(ValueError):
    """사용자에게 보여 줄 수 있는 프리셋 오류."""


def default_presets_dir() -> Path:
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'presets'


def preset_name(raw: Any) -> str:
    """파일명으로 안전한 프리셋 이름('' 이면 쓸 수 없음) — 저장·읽기·삭제가 같은 규칙."""
    return sanitize_filename(str(raw or ''), fallback='')


def preset_from_settings(settings: Mapping[str, Any] | Any) -> dict[str, Any]:
    """설정 dict 에서 프리셋 키만 (순서 유지) 뽑는다."""
    if not isinstance(settings, Mapping):
        return {}
    return {key: settings[key] for key in PRESET_KEYS if key in settings}


def _preset_path(directory: str | os.PathLike[str], name: str) -> Path:
    safe = preset_name(name)
    if not safe:
        raise PresetError('프리셋 이름이 비어 있거나 쓸 수 없는 문자만 있습니다')
    return Path(directory) / f'{safe}.json'


def list_presets(directory: str | os.PathLike[str] | None = None) -> list[str]:
    folder = Path(directory) if directory is not None else default_presets_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        names = [entry.stem for entry in folder.iterdir()
                 if entry.is_file() and entry.suffix.lower() == '.json']
    except OSError:
        return []
    return sorted(names, key=str.casefold)


def read_preset(name: str, directory: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """저장된 프리셋(프리셋 키만). 없거나 깨졌으면 None."""
    try:
        path = _preset_path(directory if directory is not None else default_presets_dir(), name)
    except PresetError:
        return None
    if not path.is_file():
        return None
    data = load_json_safe(str(path), None, backup_corrupt=False)
    if not isinstance(data, Mapping):
        return None
    return preset_from_settings(data)


def write_preset(name: str, settings: Mapping[str, Any],
                 directory: str | os.PathLike[str] | None = None) -> str:
    """설정에서 프리셋 키만 저장하고 실제 파일 이름을 돌려준다."""
    path = _preset_path(directory if directory is not None else default_presets_dir(), name)
    preset = preset_from_settings(settings)
    if not preset:
        raise PresetError('저장할 생성 설정이 없습니다')
    atomic_write_json(str(path), preset)
    return path.stem


def delete_preset(name: str, directory: str | os.PathLike[str] | None = None) -> bool:
    path = _preset_path(directory if directory is not None else default_presets_dir(), name)
    if not path.is_file():
        return False
    path.unlink()
    return True


def export_bundle(directory: str | os.PathLike[str] | None = None,
                  names: Iterable[str] | None = None) -> dict[str, Any]:
    """공유용 묶음 — {format, version, presets: {name: preset}}."""
    folder = directory if directory is not None else default_presets_dir()
    wanted = list(names) if names is not None else list_presets(folder)
    presets: dict[str, Any] = {}
    for name in wanted:
        preset = read_preset(name, folder)
        if preset:
            presets[preset_name(name)] = preset
    return {'format': BUNDLE_FORMAT, 'version': BUNDLE_VERSION, 'presets': presets}


def _file_identity(path: Path) -> tuple:
    """있는 파일의 정체 — 이름 표기(대소문자 등)가 달라도 같은 파일이면 같다(st_dev·st_ino).

    파일 번호를 주지 않는 파일 시스템(st_ino 0)은 정규화한 경로로 대신한다. 덮어쓰기
    (atomic_write_json 의 임시 파일 → replace)는 파일 번호를 바꾸므로 쓴 **뒤에** 읽는다.
    """
    stat = path.stat()
    if stat.st_ino:
        return ('id', stat.st_dev, stat.st_ino)
    return ('path', os.path.normcase(os.path.abspath(path)))


def _bundle_entries(data: Any) -> list[tuple[str, dict[str, Any]]]:
    """공유 파일의 (안전한 이름, 프리셋) — 파일 순서대로, 이름이 겹치는 항목도 모두."""
    if not isinstance(data, Mapping):
        raise PresetError('프리셋 파일 형식이 아닙니다(JSON 객체가 필요합니다)')
    if data.get('format') == BUNDLE_FORMAT:
        version = data.get('version')
        if not isinstance(version, int) or version > BUNDLE_VERSION:
            raise PresetError('이 앱보다 새 버전에서 만든 프리셋 파일입니다')
        raw = data.get('presets')
        if not isinstance(raw, Mapping):
            raise PresetError('프리셋 목록(presets)이 없습니다')
    else:
        raw = data
    entries: list[tuple[str, dict[str, Any]]] = []
    names: set[str] = set()
    for raw_name, value in raw.items():
        name = preset_name(raw_name)
        preset = preset_from_settings(value)
        if name and preset:
            entries.append((name, preset))
            names.add(name)
        if len(names) > MAX_BUNDLE_PRESETS:
            raise PresetError(f'프리셋이 너무 많습니다(최대 {MAX_BUNDLE_PRESETS}개)')
    if not entries:
        raise PresetError('가져올 생성 프리셋이 없습니다')
    return entries


def parse_bundle(data: Any) -> dict[str, dict[str, Any]]:
    """공유 파일을 {이름: 프리셋} 으로. 묶음 형식과 옛 {이름: 설정} 매핑을 모두 받는다.
    정리한 이름이 같은 항목('Beta.'·'Beta:')은 앞 항목을 쓴다(import_bundle 과 같은 규칙)."""
    out: dict[str, dict[str, Any]] = {}
    for name, preset in _bundle_entries(data):
        out.setdefault(name, preset)
    return out


def import_bundle(data: Any, *, overwrite: bool,
                  directory: str | os.PathLike[str] | None = None) -> dict[str, list[str]]:
    """공유 파일을 presets/ 에 병합한다. overwrite=False 면 이미 있는 이름은 건너뛴다.

    파일 안에서 (정리한 뒤) 같은 파일로 겹치는 이름('Alpha'·'alpha', 'Beta.'·'BETA:')은 앞 항목만
    쓰고 뒤 항목은 ``duplicate`` 로 알린다 — overwrite 여부와 상관없이. 예전엔 둘 다 '추가'로 세면서
    Windows 에서 뒤 항목이 앞 파일을 덮어, 프리셋 하나가 조용히 사라졌다.

    '같은 파일인지'는 이름 규칙으로 흉내 내지 않고 **파일 시스템에 묻는다**(그 이름의 파일이 있는지,
    있으면 이번에 이미 쓰거나 건너뛴 파일과 같은 파일인지 — :func:`_file_identity`). NTFS 의 대소문자
    비교는 볼륨에 기록된 대문자 표라 파이썬 ``upper()`` 와 다르다(이 PC 실측: é/É 는 같은 파일,
    ß/SS·ς/σ·ı/i·조지아 Mkhedruli/Mtavruli·보조 평면 글자는 서로 다른 파일). 예전 upper() 키는
    'STRASSE' 가 있을 때 'Straße' 를 overwrite=True 로 가져오면 '교체'라 보고하고 실제론 새 파일을
    만들었고, overwrite=False 면 겹치지도 않는 프리셋을 '건너뜀'으로 버렸다. 반대로 규칙이 좁으면 실제로
    겹치는 이름을 놓쳐 덮어쓴다. 파일 시스템에 물으면 두 방향 다 없다.
    """
    folder = Path(directory if directory is not None else default_presets_dir())
    entries = _bundle_entries(data)
    folder.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {'added': [], 'replaced': [], 'skipped': [], 'duplicate': []}
    claimed: set[tuple] = set()   # 이번 가져오기가 쓰거나(added·replaced) 그대로 둔(skipped) 파일
    for name, preset in entries:
        path = _preset_path(folder, name)
        if not path.is_file():
            write_preset(name, preset, folder)
            claimed.add(_file_identity(path))
            result['added'].append(name)
            continue
        if _file_identity(path) in claimed:
            result['duplicate'].append(name)
            continue
        if not overwrite:
            claimed.add(_file_identity(path))
            result['skipped'].append(name)
            continue
        write_preset(name, preset, folder)
        claimed.add(_file_identity(path))   # 덮어쓰면 파일 번호가 바뀐다 — 쓴 뒤에 읽는다
        result['replaced'].append(name)
    return result
