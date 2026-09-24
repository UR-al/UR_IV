# utils/file_wildcard.py
"""
파일 기반 와일드카드 시스템 — 앱의 유일한 파일 와일드카드 해석기이자 관리자.
- 문법: ~/이름/~ 또는 __이름__ → wildcards/이름.txt 에서 랜덤 선택 (두 문법은 같은 뜻)
  (__이름__ 은 Vue 와일드카드 관리자가 안내·삽입하는 문법이다)
- 문법: ~/이름:n/~ · __이름:n__ → n개 그룹에서 각각 1개씩 뽑기
- 각 줄이 하나의 그룹, 쉼표로 구분된 옵션 중 하나를 랜덤 선택
- '#' 로 시작하는 줄(앞 공백 무시)과 빈 줄은 주석 — 해석에서 빠지지만 파일에는 그대로 남는다
- 중첩 와일드카드 지원: 와일드카드 내에서 다른 ~/이름/~ · __이름__ 참조
- [A|B] 문법 지원 (OR 선택)
- 파일 이름은 core.file_naming.sanitize_filename 규칙 하나로 정한다(저장·읽기·해석 공통) —
  경로 구분자가 지워지므로 '~/..\\x/~' 같은 이름으로 wildcards/ 밖을 읽을 수 없다.
"""
import os
import re
import random
from typing import Dict, List, Optional

# 와일드카드 패턴: ~/이름/~ 또는 ~/이름:n/~
FILE_WILDCARD_PATTERN = re.compile(r'~/([^/]+?)/~')
# 와일드카드 패턴: __이름__ 또는 __이름:n__ — 이름은 공백·밑줄로 시작/끝나지 않고 쉼표·줄바꿈이 없다
# (hair_style 처럼 가운데 밑줄은 허용). 파일이 없으면 원문 그대로 둔다.
DUNDER_WILDCARD_PATTERN = re.compile(r'__(?![\s_])([^,\n]+?)(?<![\s_])__')
# OR 선택 패턴: [A|B|C]
OR_PATTERN = re.compile(r'\[([^\[\]]+?\|[^\[\]]+?)\]')


def is_comment_line(line: str) -> bool:
    """해석에서 빠지는 줄인가 — 빈 줄 또는 (앞 공백을 무시하고) '#' 로 시작하는 줄."""
    s = (line or '').strip()
    return not s or s.startswith('#')


# 이름 길이 한도 — 파일 이름 한 칸(255자, NTFS·ext4)에서 '.txt' 를 뺀 값. sanitize_filename 의 기본값
# (64자)으로 자르면 64자가 넘는 기존 파일을 해석·편집·삭제할 때 잘린 이름을 찾아 모두 실패했다.
WILDCARD_NAME_MAX = 255 - len('.txt')


def wildcard_file_name(name: str) -> str:
    """사용자·프롬프트가 준 이름 → 저장 파일의 이름(확장자 제외). 쓸 수 없는 이름이면 ''."""
    from core.file_naming import sanitize_filename
    raw = str(name or '').strip()
    if raw.lower().endswith('.txt'):
        raw = raw[:-4]
    return sanitize_filename(raw, fallback='', max_len=WILDCARD_NAME_MAX)


def wildcards_enabled(owner) -> bool:
    """와일드카드 시스템 ON/OFF 의 단일 판정.

    값의 주인은 ``owner.prompt_settings_extras.wildcard_enabled``(core/prompt_settings_extras —
    prompt_settings.json 에 영속, Vue 설정의 CheckBoxProxy 'wildcard_enabled' 가 바꾼다)다.
    예전엔 숨은 레거시 SettingsTab 체크박스였다(audit #178). 보관함이 없는 호스트는 꺼진 것으로 본다.
    """
    from core.prompt_settings_extras import PromptSettingsExtras
    extras = getattr(owner, 'prompt_settings_extras', None)
    return bool(extras.wildcard_enabled) if isinstance(extras, PromptSettingsExtras) else False


class FileWildcardManager:
    """파일 기반 와일드카드 관리자 (싱글톤)"""

    def __init__(self, wildcards_dir: str = None):
        if wildcards_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            wildcards_dir = os.path.join(base, 'wildcards')
        self.wildcards_dir = wildcards_dir
        os.makedirs(self.wildcards_dir, exist_ok=True)
        self._cache: Dict[str, List[List[str]]] = {}
        self._cache_mtime: Dict[str, float] = {}

    def reload(self):
        """캐시 초기화 (파일 변경 후 호출)"""
        self._cache.clear()
        self._cache_mtime.clear()

    def _path(self, name: str) -> Optional[str]:
        """이름 → wildcards/ 안의 .txt 경로. 쓸 수 없는 이름이면 None (저장·읽기·해석 공통 규칙)."""
        safe = wildcard_file_name(name)
        if not safe:
            return None
        path = os.path.join(self.wildcards_dir, f'{safe}.txt')
        # 방어선 — 이름 규칙이 구분자를 지우므로 늘 wildcards/ 바로 아래여야 한다.
        # (realpath 가 아니라 abspath: 사용자가 일부러 둔 심볼릭 링크 파일은 그대로 쓴다)
        if os.path.dirname(os.path.abspath(path)) != os.path.abspath(self.wildcards_dir):
            return None
        return path

    def _existing_path(self, name: str) -> Optional[str]:
        """이름이 가리키는 **있는** 파일의 경로 — 없으면 None. 삭제·이름 변경이 건드릴 파일.

        목록(get_wildcard_names)의 이름 그대로인 파일(``이름.txt``, 이름이 '.txt' 로 끝나면 그 이름도)을
        먼저 찾고, 없으면 이름 규칙(_path)의 경로가 **대소문자만 다른 같은 이름**일 때 그 파일을 쓴다.
        이름 규칙은 앞뒤 공백·끝의 점을 지운다 — 밖에서 만든 ' lead.txt' · 'dot..txt' 는 목록에 ' lead' ·
        'dot.' 으로 나오지만 규칙상 lead.txt · dot.txt 를 가리킨다. 예전 삭제는 그 (없는) 경로만 보고
        아무것도 안 한 채 성공해 Vue 가 '삭제됨'을 띄웠고 파일은 다음 목록에 되살아났다(같은 이름의
        lead.txt 가 있으면 엉뚱하게 그 파일을 지웠다). 규칙이 **다른** 이름을 만들면 그 파일은 건드리지 않는다.
        """
        raw = str(name or '')
        wanted = [raw + '.txt'] + ([raw] if raw.lower().endswith('.txt') else [])
        try:
            entries = os.listdir(self.wildcards_dir)
        except OSError:
            return None
        found = next((w for w in wanted if w in entries), None)
        if found is None:
            ruled = self._path(name)
            key = os.path.normcase(os.path.basename(ruled)) if ruled is not None else None
            if key is not None and key in {os.path.normcase(w) for w in wanted}:
                found = next((f for f in entries if os.path.normcase(f) == key), None)
        if found is None:
            return None
        path = os.path.join(self.wildcards_dir, found)
        return path if os.path.isfile(path) else None

    def _existing_file_name(self, path: str) -> str:
        """이미 있는 파일의 실제 이름(확장자 제외) — 대소문자만 다른 이름으로 찾았으면 디스크의 표기."""
        target = os.path.basename(path)
        try:
            entries = os.listdir(self.wildcards_dir)
        except OSError:
            entries = []
        if target not in entries:
            wanted = os.path.normcase(target)
            target = next((f for f in entries if os.path.normcase(f) == wanted), target)
        return target[:-4] if target.lower().endswith('.txt') else target

    def _forget(self, name: str):
        safe = wildcard_file_name(name)
        self._cache.pop(safe, None)
        self._cache_mtime.pop(safe, None)

    def get_wildcard_names(self) -> List[str]:
        """사용 가능한 와일드카드 이름 목록 반환"""
        names = []
        if not os.path.isdir(self.wildcards_dir):
            return names
        for f in sorted(os.listdir(self.wildcards_dir)):
            if f.endswith('.txt') and os.path.isfile(os.path.join(self.wildcards_dir, f)):
                names.append(f[:-4])
        return names

    def load_wildcard(self, name: str) -> List[List[str]]:
        """
        와일드카드 파일 로드 (캐시 사용)
        반환: [[옵션1, 옵션2, ...], [옵션A, 옵션B, ...], ...]
              각 리스트가 하나의 '그룹' (파일의 한 줄)
        """
        filepath = self._path(name)
        if filepath is None or not os.path.isfile(filepath):
            return []
        key = wildcard_file_name(name)

        # 캐시 유효성 검사
        mtime = os.path.getmtime(filepath)
        if key in self._cache and self._cache_mtime.get(key) == mtime:
            return self._cache[key]

        groups = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if is_comment_line(line):
                    continue
                # 쉼표로 옵션 분리
                options = [opt.strip() for opt in line.strip().split(',') if opt.strip()]
                if options:
                    groups.append(options)

        self._cache[key] = groups
        self._cache_mtime[key] = mtime
        return groups

    def save_wildcard(self, name: str, content: str) -> str:
        """와일드카드 파일 저장 — 주석(#)·빈 줄을 포함한 원문을 그대로 쓴다. 저장된 이름을 반환."""
        filepath = self._path(name)
        if filepath is None:
            raise ValueError(f'와일드카드 이름을 쓸 수 없습니다: {name!r}')
        os.makedirs(self.wildcards_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content or '')
        # 캐시 무효화 (같은 초 안의 재저장은 mtime 이 안 바뀔 수 있다)
        self._forget(name)
        return wildcard_file_name(name)

    def create_wildcard(self, name: str) -> dict:
        """새 빈 와일드카드 파일을 만든다 — 같은 파일이 이미 있으면 **손대지 않는다**.

        '같은 파일'은 이름 규칙(sanitize)을 거친 경로로 판단한다: 대소문자만 다른 이름(Windows 는
        같은 파일), 'hair:style' · 'hairstyle.' 처럼 규칙상 기존 이름이 되는 경우도 포함한다.
        예전 Vue '+ NEW' 는 목록과 글자 그대로 비교한 뒤 save_wildcard(이름, '') 로 써서 이런 이름이면
        기존 파일(주석 포함)을 빈 파일로 덮었다. 배타적 생성('x')이라 확인과 생성 사이 경합도 없다.

        반환: ``{'name': 파일 이름(확장자 제외), 'created': 새로 만들었는가}``.
        """
        filepath = self._path(name)
        if filepath is None:
            raise ValueError(f'와일드카드 이름을 쓸 수 없습니다: {name!r}')
        os.makedirs(self.wildcards_dir, exist_ok=True)
        try:
            with open(filepath, 'x', encoding='utf-8'):
                pass
        except FileExistsError:
            return {'name': self._existing_file_name(filepath), 'created': False}
        self._forget(name)
        return {'name': wildcard_file_name(name), 'created': True}

    def delete_wildcard(self, name: str):
        """와일드카드 파일 삭제 — 이름이 가리키는 파일(:meth:`_existing_path`, 목록의 이름 그대로 우선).

        지우지 못하면(읽기 전용·잠김) 예외가 그대로 올라가 브리지가 {error} 로 바꾼다. 가리키는 파일이
        이미 없으면(밖에서 지웠다) 목표 상태라 성공이다 — 이름 규칙상 같아지는 **다른** 파일은 지우지 않는다.
        """
        filepath = self._existing_path(name)
        if filepath is not None:
            os.remove(filepath)
        self._forget(name)

    def rename_wildcard(self, old_name: str, new_name: str) -> str:
        """와일드카드 파일 이름 변경 — 새 이름의 파일이 이미 있으면 덮어쓰지 않고 실패한다.

        옛 파일은 :meth:`_existing_path` 로 찾는다(목록의 이름 그대로 우선). 없으면 아무것도 바꾸지 않고
        FileNotFoundError — 예전엔 규칙상 경로(' lead' → lead.txt)가 없으면 조용히 새 이름을 돌려줘 Vue 가
        이름을 바꾼 것처럼 보였다. 반환: 저장된 새 이름."""
        new_fp = self._path(new_name)
        if new_fp is None:
            raise ValueError('와일드카드 이름을 쓸 수 없습니다')
        old_fp = self._existing_path(old_name)
        if old_fp is None:
            raise FileNotFoundError(f'와일드카드 파일이 없습니다: {old_name}')
        if os.path.normcase(old_fp) != os.path.normcase(new_fp):
            if os.path.exists(new_fp):
                raise FileExistsError(f'같은 이름의 와일드카드가 이미 있습니다: {wildcard_file_name(new_name)}')
            os.rename(old_fp, new_fp)
        elif old_fp != new_fp:
            # 대소문자만 다른 이름(Windows 에서 같은 파일) — 표기만 바꾼다
            os.rename(old_fp, new_fp)
        self._forget(old_name)
        self._forget(new_name)
        return wildcard_file_name(new_name)

    def get_wildcard_content(self, name: str) -> str:
        """와일드카드 파일 내용 반환 (편집용)"""
        filepath = self._path(name)
        if filepath is None or not os.path.isfile(filepath):
            return ''
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def get_wildcard_tree(self) -> List[dict]:
        """Vue 와일드카드 관리자용 목록 — [{name, file, tags, lines}].

        - ``lines``: 파일의 원문 줄(주석·빈 줄 포함, 끝의 빈 줄만 제외) — 편집·저장용이라
          Vue 에서 저장해도 주석이 사라지지 않는다.
        - ``tags``: 해석에 쓰이는 줄(주석·빈 줄 제외) — 개수 표시용(예전 계약 유지).
        """
        tree = []
        for name in self.get_wildcard_names():
            try:
                raw_lines = self.get_wildcard_content(name).splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            while raw_lines and not raw_lines[-1].strip():
                raw_lines.pop()
            tree.append({
                'name': name,
                'file': f'{name}.txt',
                'tags': [ln.strip() for ln in raw_lines if not is_comment_line(ln)],
                'lines': raw_lines,
            })
        return tree

    def resolve(self, text: str, max_depth: int = 10) -> str:
        """
        텍스트 내의 모든 ~/이름/~ · __이름__ 와일드카드를 치환
        - ~/이름/~ : 모든 그룹에서 각각 1개씩 선택, 쉼표로 연결
        - ~/이름:n/~ : n개 그룹만 랜덤 선택하여 각각 1개씩
        - 중첩 지원 (max_depth까지 반복)
        """
        if not text:
            return text

        depth = 0
        prev = None
        while prev != text and depth < max_depth:
            prev = text
            text = FILE_WILDCARD_PATTERN.sub(self._replace_match, text)
            text = DUNDER_WILDCARD_PATTERN.sub(self._replace_match, text)
            depth += 1

        # [A|B] OR 패턴 처리
        text = self._resolve_or_patterns(text)

        return text

    def _replace_match(self, match) -> str:
        """단일 와일드카드 매치 교체"""
        raw = match.group(1)

        # n-pick 파싱: 이름:n
        if ':' in raw:
            parts = raw.rsplit(':', 1)
            name = parts[0].strip()
            try:
                n_pick = int(parts[1].strip())
            except ValueError:
                # ':뒤'가 숫자가 아니면 콜론 앞 이름만 사용(전체) — 'name:bad'를 그대로
                # 파일명으로 쓰면 절대 못 찾던 버그 수정
                name = parts[0].strip()
                n_pick = 0  # 0 = 전체
        else:
            name = raw.strip()
            n_pick = 0

        groups = self.load_wildcard(name)
        if not groups:
            return match.group(0)  # 파일이 없으면 원본 유지

        # 그룹 선택
        if n_pick > 0 and n_pick < len(groups):
            selected_groups = random.sample(groups, n_pick)
        else:
            selected_groups = groups

        # 각 그룹에서 하나씩 선택
        picked = []
        for group in selected_groups:
            choice = random.choice(group)
            # [A|B] 패턴이 있으면 처리
            choice = self._resolve_or_patterns(choice)
            picked.append(choice.strip())

        return ', '.join(picked)

    def _resolve_or_patterns(self, text: str) -> str:
        """[A|B|C] 패턴을 랜덤 선택으로 교체"""
        def _replace_or(m):
            options = [o.strip() for o in m.group(1).split('|') if o.strip()]
            return random.choice(options) if options else ''

        prev = None
        while prev != text:
            prev = text
            text = OR_PATTERN.sub(_replace_or, text)
        return text

    def preview(self, name: str) -> str:
        """와일드카드 미리보기 (한 번 실행 결과)"""
        groups = self.load_wildcard(name)
        if not groups:
            return '(파일 없음)'
        picked = []
        for group in groups:
            choice = random.choice(group)
            choice = self._resolve_or_patterns(choice)
            picked.append(choice.strip())
        return ', '.join(picked)

    def get_info(self, name: str) -> dict:
        """와일드카드 정보 반환"""
        groups = self.load_wildcard(name)
        total_options = sum(len(g) for g in groups)
        return {
            'name': name,
            'groups': len(groups),
            'total_options': total_options,
        }


# 싱글톤
_instance: Optional[FileWildcardManager] = None


def get_file_wildcard_manager() -> FileWildcardManager:
    global _instance
    if _instance is None:
        _instance = FileWildcardManager()
    return _instance


def resolve_file_wildcards(text: str) -> str:
    """간편 함수: 파일 와일드카드 치환"""
    return get_file_wildcard_manager().resolve(text)
