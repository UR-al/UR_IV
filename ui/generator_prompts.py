# ui/generator_prompts.py
"""
프롬프트 처리 관련 로직
"""
import random
from PyQt6.QtWidgets import QMessageBox
from utils.app_logger import get_logger

_logger = get_logger('prompts')

class PromptHandlingMixin:
    """프롬프트 처리를 담당하는 Mixin 클래스"""

    def _is_character_feature_tag(self, tag: str) -> bool:
        """캐릭터 특징 태그인지 확인"""
        if not hasattr(self, 'tag_classifier'):
            return False
        return self.tag_classifier.classify_tag(tag) == 'character_trait'
    
    def update_total_prompt_display(self):
        """최종 프롬프트 디스플레이 업데이트.

        섹션 순서는 ``config/prompt_order.json``에서 로드 — 사용자가
        Studio Tools의 ORDER 매니저로 변경 가능. 기본 순서는 기존과 동일:
        char_count → character → copyright → artist → prefix → main → suffix
        """
        # 7개 섹션의 현재 텍스트 수집
        section_texts = {
            "char_count": self.char_count_input.text(),
            "character":  self.character_input.text(),
            "copyright":  self.copyright_input.text(),
            "artist":     self.artist_input.toPlainText(),
            "prefix":     self.prefix_prompt_text.toPlainText(),
            "main":       self.main_prompt_text.toPlainText(),
            "suffix":     self.suffix_prompt_text.toPlainText(),
        }
        try:
            from core.prompt_order import compose_prompt
            final = compose_prompt(section_texts)
        except Exception:
            # 보수적 폴백 — config 로드 실패 시 기본 순서로
            final = ", ".join(s.strip() for s in [
                section_texts["char_count"], section_texts["character"],
                section_texts["copyright"], section_texts["artist"],
                section_texts["prefix"], section_texts["main"],
                section_texts["suffix"],
            ] if s.strip())

        self.total_prompt_display.setPlainText(final)
    
    def apply_prompt_from_data(self, bundle, preserve_locked=False, comma_only=False):
        """검색 결과 데이터를 프롬프트 입력창에 적용.
        preserve_locked=True (프롬프트 당겨오기): 선행/후행/작가 칸을 덮어쓰지 않고,
        당겨오는 태그를 그 칸들과 중복되면 제거한다 (인물수/캐릭터/main은 그대로 override).
        comma_only=True: 항상 콤마로만 토큰 분리 (당겨오기/EXIF — 다중단어 태그를
        공백으로 쪼개지 않음). False면 콤마 없을 때 공백 분리(검색 결과 호환)."""
        # 1. 데이터 추출
        general_str = str(bundle.get('general', ''))
        artist_str = str(bundle.get('artist', ''))
        copyright_str = str(bundle.get('copyright', ''))
        character_str = str(bundle.get('character', ''))
        
        if artist_str == 'nan': artist_str = ''
        if copyright_str == 'nan': copyright_str = ''
        if character_str == 'nan': character_str = ''
        if general_str == 'nan': general_str = ''
        
        def to_list(text):
            # 포맷 인식: 콤마 있으면 콤마 구분(2025/2026 검색·EXIF 프롬프트 — 태그
            # 내부 공백 유지), 없으면 공백 구분 + 언더스코어→공백(2026_06 검색 —
            # 'black_hair'가 'black','hair'로 안 쪼개지고 프롬프트엔 'black hair'로 들어감).
            # comma_only=True면 콤마 없어도 공백 분리 안 함(단일 다중단어 태그 보존).
            text = (text or '').strip()
            if not text:
                return []
            if comma_only or ',' in text:
                return [t.strip() for t in text.split(',') if t.strip()]
            return [t.strip().replace('_', ' ') for t in text.split() if t.strip()]

        # 2. 기본 리스트 생성
        artist_list = to_list(artist_str.replace('artist:', '')) if artist_str else []
        copyright_list = to_list(copyright_str)
        character_list = to_list(character_str)
        general_list = to_list(general_str)

        # ★★★ 'original'은 작품명이 아니므로 제외 ★★★
        copyright_list = [c for c in copyright_list if c.lower() != 'original']
        
        # ★★★ 디버그: 원본 general_list 개수 ★★★
        _logger.debug(f"원본 general_list: {len(general_list)}개")
        
        # 3. 제외 프롬프트 적용
        exclude_text = self.exclude_prompt_local_input.toPlainText()
        exclude_rules = to_list(exclude_text)
        
        # 문법:
        #   단어       → 포함하는 모든 태그 제거 (ex: short → short hair, very short hair)
        #   *단어      → 완전 일치만 제거 (ex: *blue hair → blue hair만)
        #   _단어      → 앞에 뭔가 붙은 태그 제거 (ex: _short → very short, too short)
        #   단어_      → 뒤에 뭔가 붙은 태그 제거 (ex: short_ → short hair, short pants)
        #   _단어_     → 포함하는 모든 태그 (단어와 동일하나 명시적)
        #   ~단어      → 예외 완전 일치 (제외하지 않고 유지)
        #   ~_단어     → 예외 접미 (ex: ~_tank top → tank top, blue tank top 유지)
        #   ~단어_     → 예외 접두 (ex: ~tank_ → tank top 유지)
        contains_exc, exact_exc, prefix_exc, suffix_exc = [], set(), [], []
        excepts_exact, excepts_suffix, excepts_prefix, excepts_contains = set(), [], [], []
        for r in exclude_rules:
            r = r.strip()
            if not r:
                continue
            if r.startswith('~'):
                inner = r[1:].strip()
                if inner.startswith('_') and inner.endswith('_') and len(inner) > 2:
                    excepts_contains.append(inner[1:-1].strip())  # ~_tank top_ → "tank top" 포함 태그 유지
                elif inner.startswith('_'):
                    excepts_suffix.append(inner[1:].strip())  # ~_tank top → "tank top"으로 끝나는 태그 유지
                elif inner.endswith('_'):
                    excepts_prefix.append(inner[:-1].strip())  # ~tank_ → "tank"로 시작하는 태그 유지
                else:
                    excepts_exact.add(inner)
            elif r.startswith('*'):
                exact_exc.add(r[1:].strip())
            elif r.startswith('_') and r.endswith('_') and len(r) > 2:
                contains_exc.append(r[1:-1].strip())
            elif r.startswith('_'):
                suffix_exc.append(r[1:].strip())
            elif r.endswith('_'):
                prefix_exc.append(r[:-1].strip())
            else:
                contains_exc.append(r)

        def _normalize(tag):
            return tag.replace('_', ' ').strip().lower()

        norm_exact = {_normalize(e) for e in exact_exc}
        norm_excepts_exact = {_normalize(e) for e in excepts_exact}
        norm_excepts_suffix = [_normalize(s) for s in excepts_suffix if s]
        norm_excepts_prefix = [_normalize(p) for p in excepts_prefix if p]
        norm_excepts_contains = [_normalize(c) for c in excepts_contains if c]
        norm_contains = [_normalize(c) for c in contains_exc if c]
        norm_prefix = [_normalize(p) for p in prefix_exc if p]
        norm_suffix = [_normalize(s) for s in suffix_exc if s]

        def _is_excepted(nt):
            """예외 규칙에 해당하면 True (유지)"""
            if nt in norm_excepts_exact:
                return True
            if any(nt.endswith(s) for s in norm_excepts_suffix):
                return True
            if any(nt.startswith(p) for p in norm_excepts_prefix):
                return True
            if any(c in nt for c in norm_excepts_contains):
                return True
            return False

        def filter_tags(tags, proper=False):
            res = []
            for t in tags:
                nt = _normalize(t)
                # 예외 규칙: 유지
                if _is_excepted(nt):
                    res.append(t)
                    continue
                # 완전 일치
                if nt in norm_exact:
                    continue
                # 포함: general은 부분일치, 고유명사(캐릭터/작품/작가)는 '전체 태그 일치'만.
                #   (예: 'tank' 규칙이 캐릭터 'wakan tanka'(=wakan_tanka)를 부분일치로 통째 지우던 버그 방지.
                #    general의 'tank top' 제거는 그대로 유지.)
                if proper:
                    if any(nt == c for c in norm_contains):
                        continue
                else:
                    if any(c in nt for c in norm_contains):
                        continue
                # 접두/접미는 태그 경계 기준이라 고유명사에도 안전 (그대로 적용)
                if any(nt.startswith(p) for p in norm_prefix):
                    continue
                if any(nt.endswith(s) for s in norm_suffix):
                    continue
                res.append(t)
            return res

        artist_list = filter_tags(artist_list, proper=True)
        copyright_list = filter_tags(copyright_list, proper=True)
        character_list = filter_tags(character_list, proper=True)
        general_list = filter_tags(general_list)

        # 4. 제거 토글 적용
        
        # 작가 처리 (고정 vs 제거 vs 적용)
        keep_current_artist = False
        if self.btn_lock_artist.isChecked() or preserve_locked:
            # 작가 고정 또는 당겨오기 → 작가 칸을 덮어쓰지 않음 (override 방지)
            keep_current_artist = True
            artist_list = []
        elif self.chk_remove_artist.isChecked():
            artist_list = []
        
        # 작품명 제거
        if self.chk_remove_copyright.isChecked(): 
            copyright_list = []

        # 캐릭터 제거
        if self.chk_remove_character.isChecked():
            character_list = []

        # 캐릭터 특징 제거
        if (hasattr(self, 'chk_remove_character_features') and
                self.chk_remove_character_features.isChecked()):
            before = len(general_list)
            general_list = [t for t in general_list if not self._is_character_feature_tag(t)]
            _logger.debug(f"캐릭터 특징 제거: {before} → {len(general_list)}")
        
        # ★★★ 디버그: 토글 상태 및 태그 개수 ★★★
        _logger.debug(f"=== 토글 상태 ===")
        _logger.debug(f"메타 제거: {self.chk_remove_meta.isChecked()}")
        _logger.debug(f"검열 제거: {self.chk_remove_censorship.isChecked()}")
        _logger.debug(f"텍스트 제거: {self.chk_remove_text.isChecked()}")
        # 분류기 통계는 **이미 만들어진 분류기가 있을 때만** 남긴다. `tag_classifier` 는 지연
        # 로드 프로퍼티라 hasattr·속성 접근만으로 분류기(GUI 스레드 0.5초 + tag_groups·
        # implications parquet)를 만든다. 로그 레벨 검사로는 못 막는다 — utils/app_logger 가
        # 루트를 DEBUG 로 두어 isEnabledFor(DEBUG) 가 늘 참이다.
        classifier = getattr(self, '_tag_classifier', None)
        if classifier is not None:
            _logger.debug("censorship_tags 개수: %d, text_tags 개수: %d",
                          len(classifier.censorship_tags), len(classifier.text_tags))

        # 메타 제거 (parquet 773개 meta 태그 + art_style 기반)
        if self.chk_remove_meta.isChecked():
            before = len(general_list)
            general_list = [t for t in general_list
                           if not self.tag_classifier.is_meta_tag(t)]
            _logger.debug(f"메타 제거: {before} → {len(general_list)}")

        # 검열 제거
        if self.chk_remove_censorship.isChecked():
            before = len(general_list)
            removed = []
            new_list = []
            for t in general_list:
                if self.tag_classifier.is_censorship_tag(t):
                    removed.append(t)
                else:
                    new_list.append(t)
            general_list = new_list
            _logger.debug(f"검열 제거: {before} → {len(general_list)}, 제거된 태그: {removed}")

        # 텍스트 제거
        if self.chk_remove_text.isChecked():
            before = len(general_list)
            removed = []
            new_list = []
            for t in general_list:
                if self.tag_classifier.is_text_tag(t):
                    removed.append(t)
                else:
                    new_list.append(t)
            general_list = new_list
            _logger.debug(f"텍스트 제거: {before} → {len(general_list)}, 제거된 태그: {removed}")

        # 4.5. general_list에서 이미 분류된 태그 중복 제거
        classified_tags = set()
        
        for c in character_list:
            classified_tags.add(c.lower())
        
        for c in copyright_list:
            classified_tags.add(c.lower())
        
        for a in artist_list:
            classified_tags.add(a.lower())
        
        general_list = [t for t in general_list if t.lower() not in classified_tags]

        # 5. 인물 수 분류
        count_tags = {
            "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys",
            "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
            "1other", "2others", "3others", "4others", "5others", "6+others",
            "multiple boys", "multiple girls", "multiple others",
        }

        final_count = []
        final_general = []

        for t in general_list:
            # 언더스코어/공백 차이 무시하고 매칭 (multiple_boys ↔ multiple boys)
            if t.lower().replace('_', ' ') in count_tags:
                final_count.append(t)
            else:
                final_general.append(t)
        
        _logger.debug(f"최종 general: {len(final_general)}개")
        
        # 6. 선행/후행 프롬프트와 중복 제거
        def _norm(tag: str) -> str:
            return tag.replace('_', ' ').strip().lower()

        fixed_tags = set()
        _fixed_srcs = [self.prefix_prompt_text.toPlainText(),
                       self.suffix_prompt_text.toPlainText()]
        if preserve_locked:
            # 당겨오기: 작가 칸도 중복 제거 대상에 포함 (작가 칸과 겹치는 태그는 안 당겨옴)
            _fixed_srcs.append(self.artist_input.toPlainText())
        for src in _fixed_srcs:
            for t in src.split(','):
                nt = _norm(t)
                if nt:
                    fixed_tags.add(nt)

        if fixed_tags:
            # 선행/후행 고정 프롬프트에 이미 있는 태그는 당겨오지 않음 —
            # general뿐 아니라 인물수/캐릭터/작품에도 적용 (중복 방지).
            final_general = [t for t in final_general if _norm(t) not in fixed_tags]
            final_count = [t for t in final_count if _norm(t) not in fixed_tags]
            character_list = [t for t in character_list if _norm(t) not in fixed_tags]
            copyright_list = [t for t in copyright_list if _norm(t) not in fixed_tags]
            _logger.debug(f"고정프롬프트 중복 제거 후 general: {len(final_general)}개")

        # 6.5. general 내부 중복 제거
        seen_tags = set()
        deduped_general = []
        for t in final_general:
            nt = _norm(t)
            if nt not in seen_tags:
                seen_tags.add(nt)
                deduped_general.append(t)
        final_general = deduped_general

        # 7. 이스케이프 (가중치 괄호 보존)
        from utils.prompt_cleaner import escape_parentheses
        def escape(tags):
            result = []
            for t in tags:
                if r'\(' in t or r'\)' in t:
                    result.append(t)
                else:
                    result.append(escape_parentheses(t))
            return result

        # 8. UI 업데이트 (prefix/suffix/neg는 base에서 복원 → 조건부 누적 방지 + 와일드카드 템플릿 보존)
        self.is_programmatic_change = True

        self.char_count_input.setText(", ".join(escape(final_count)))
        self.character_input.setText(", ".join(escape(character_list)))
        self.copyright_input.setText(", ".join(escape(copyright_list)))

        if not keep_current_artist:
            self.artist_input.setPlainText(", ".join(escape(artist_list)))

        self.main_prompt_text.setPlainText(", ".join(escape(final_general)))

        # prefix/suffix/neg를 base 값에서 복원 (매 사이클 시작 시 원본 템플릿)
        if hasattr(self, 'base_prefix_prompt'):
            self.prefix_prompt_text.setPlainText(self.base_prefix_prompt)
        if hasattr(self, 'base_suffix_prompt'):
            self.suffix_prompt_text.setPlainText(self.base_suffix_prompt)
        if hasattr(self, 'base_neg_prompt'):
            self.neg_prompt_text.setPlainText(self.base_neg_prompt)

        self.is_programmatic_change = False

        # 9. 캐릭터 특징 자동 추가
        if (hasattr(self, 'chk_auto_char_features') and
                self.chk_auto_char_features.isChecked() and character_list and
                not (hasattr(self, 'chk_remove_character_features') and
                     self.chk_remove_character_features.isChecked())):
            self._auto_insert_character_features(character_list)

        # 10. 조건부 프롬프트 1차 적용 (와일드카드 해석 전) — cond_rules.json 기반(내부에서 ON/OFF 판단)
        self._apply_conditional_prompts()

        # 11. 와일드카드 치환
        from utils.file_wildcard import wildcards_enabled
        wc_enabled = wildcards_enabled(self)
        if wc_enabled:
            from utils.file_wildcard import resolve_file_wildcards
            from utils.wildcard import process_wildcards
            self.is_programmatic_change = True
            for widget in (self.main_prompt_text, self.prefix_prompt_text,
                           self.suffix_prompt_text, self.neg_prompt_text):
                text = widget.toPlainText()
                if not text.strip():
                    continue
                resolved = process_wildcards(resolve_file_wildcards(text))
                if resolved != text:
                    widget.setPlainText(resolved)
            self.is_programmatic_change = False

        # 12. 조건부 프롬프트 2차 적용 (와일드카드 해석 후 새 태그에 대해)
        if wc_enabled:
            self._apply_conditional_prompts()

        # 13. 자동 해상도 (Parquet H/W)
        if hasattr(self, 'auto_res_check') and self.auto_res_check.isChecked():
            raw_w = bundle.get('image_width')
            raw_h = bundle.get('image_height')
            if raw_w is not None and raw_h is not None:
                try:
                    from core.resolution_guard import (
                        ANIMA_MAX_AREA, ANIMA_MAX_SIDE, apply_anima_resolution,
                    )
                    # 면적 캡(비율 유지·8배수) — raw 크기가 커도 총 픽셀을 안전치 이내로 축소.
                    #   기존엔 각 변만 2048로 클램프해 2048×2048(4.2M px)까지 허용 → 큰 원본에서
                    #   백엔드 OOM(500 Internal Server Error)이 간헐적으로 발생했음.
                    # guard가 면적·한변 캡을 비율 유지로 처리 → 여기선 최소만 보정(비율 안 깸)
                    w, h = apply_anima_resolution(
                        int(float(raw_w)), int(float(raw_h)), auto_res=True,
                        max_area=int(getattr(self, '_anima_guard_max_area', ANIMA_MAX_AREA)),
                        max_side=int(getattr(self, '_anima_guard_max_side', ANIMA_MAX_SIDE)),
                        enabled=bool(getattr(self, '_anima_guard_enabled', True)),
                    )
                    w = max(256, w)
                    h = max(256, h)
                    self.width_input.setText(str(w))
                    self.height_input.setText(str(h))
                except (ValueError, TypeError):
                    pass

        # 14. 최종 프롬프트 업데이트
        self.update_total_prompt_display()

    def _auto_insert_character_features(self, character_list: list[str]):
        """캐릭터 특징 자동 삽입 (핵심/전체 모드 지원)"""
        from utils.character_features import get_character_features
        from utils.character_presets import get_character_preset_full

        # 핵심만 vs 핵심+의상 모드
        core_only = True
        if hasattr(self, 'combo_char_feature_mode'):
            core_only = self.combo_char_feature_mode.currentIndex() == 0

        # 기존 태그 수집
        all_existing: set[str] = set()
        for src in (self.main_prompt_text.toPlainText(),
                    self.prefix_prompt_text.toPlainText(),
                    self.suffix_prompt_text.toPlainText(),
                    self.character_input.text()):
            for t in src.split(","):
                n = t.strip().lower().replace("_", " ")
                if n:
                    all_existing.add(n)
                    all_existing.add(n.replace(r"\(", "(").replace(r"\)", ")"))

        from utils.condition_block import rules_from_json, migrate_old_rules as _migrate_rules

        lookup = get_character_features()
        new_tags: list[str] = []
        char_cond_all_rules = []

        # ── 'auto remove' (충돌 자동 처리) + override 설정 ──
        # 자동화 시 캐릭터 특징을 추가할 때, 덱 프롬프트와 충돌하는 특징을 다룬다.
        #  · closed eyes 가 이미 있으면 캐릭터 눈색('blue eyes' 등) 추가 안 함
        #  · 머리 길이(short hair 등)가 이미 있으면 다른 머리 길이 특징 추가 안 함
        #  · override(눈색/머리길이)가 켜진 카테고리는 충돌 태그를 제거하고 특징으로 교체
        auto_remove = (hasattr(self, 'chk_auto_remove_char_features') and
                       self.chk_auto_remove_char_features.isChecked())
        ov = getattr(self, '_char_feature_override', None) or {}
        override_hair = bool(ov.get('hair_length'))
        override_eye = bool(ov.get('eye_color'))
        removals: set[str] = set()   # main에서 제거할 정규화 태그 (override 교체 시)
        if auto_remove:
            from utils.character_features import (
                is_eye_color_tag, is_eye_color_hider, is_hair_length_tag, _norm_tag,
            )
            # _norm_tag로 정규화해 저장 → 아래 main 제거 비교(_norm_tag 동등)와 형태 일치
            existing_eye_hiders = {_norm_tag(n) for n in all_existing if is_eye_color_hider(n)}
            existing_hair_lengths = {_norm_tag(n) for n in all_existing if is_hair_length_tag(n)}
        else:
            existing_eye_hiders = set()
            existing_hair_lengths = set()

        def _accept_feature(tag: str, norm: str) -> bool:
            """auto_remove 모드에서 충돌하는 특징을 거를지 결정.
            override가 켜진 카테고리는 충돌 태그를 제거(removals 누적)하고 추가 허용."""
            if not auto_remove:
                return True
            # 눈 색 특징 vs 'closed eyes'
            if existing_eye_hiders and is_eye_color_tag(tag):
                if override_eye:
                    removals.update(existing_eye_hiders)
                    return True
                return False
            # 머리 길이 특징 vs 기존의 다른 머리 길이
            if is_hair_length_tag(tag):
                others = {h for h in existing_hair_lengths if h != norm}
                if others:
                    if override_hair:
                        removals.update(others)
                        return True
                    return False
            return True

        for char_raw in character_list:
            char_name = char_raw.strip().replace(r"\(", "(").replace(r"\)", ")")
            char_norm = char_name.lower().replace("_", " ").replace("(", "").replace(")", "").strip()

            # 1. 커스텀 프리셋 조회 (조건부 규칙 포함)
            preset = get_character_preset_full(char_name)
            if preset:
                extra = preset.get("extra_prompt", "")
                if extra:
                    for t in extra.split(","):
                        tag = t.strip()
                        norm = tag.lower().replace("_", " ")
                        if norm and norm not in all_existing and norm != char_norm:
                            if _accept_feature(tag, norm):
                                new_tags.append(tag)
                                all_existing.add(norm)

                # 조건부 규칙 수집 (새 JSON 포맷 우선)
                cond_json = preset.get("cond_rules_json", "")
                if cond_json:
                    char_cond_all_rules.extend(rules_from_json(cond_json))
                else:
                    cr = preset.get("cond_rules", "")
                    if cr:
                        char_cond_all_rules.extend(_migrate_rules(cr))
                    cnr = preset.get("cond_neg_rules", "")
                    if cnr:
                        neg_rules = _migrate_rules(cnr)
                        for r in neg_rules:
                            r.location = "neg"
                        char_cond_all_rules.extend(neg_rules)

            # 2. 캐릭터 특징 사전 조회 (모드에 따라 분기)
            if core_only:
                result = lookup.lookup_core(char_name)
            else:
                result = lookup.lookup(char_name)

            if result:
                features_str = result[0]
                for tag in features_str.split(","):
                    tag = tag.strip()
                    norm = tag.lower().replace("_", " ")
                    if norm and norm not in all_existing and norm != char_norm:
                        if _accept_feature(tag, norm):
                            new_tags.append(tag)
                            all_existing.add(norm)

        # 2.5 override 교체: 충돌하던 기존 태그를 main에서 제거 (제거 후 아래에서 특징 추가)
        if removals:
            self.is_programmatic_change = True
            kept = []
            for t in self.main_prompt_text.toPlainText().split(","):
                ts = t.strip()
                if not ts:
                    continue
                tnorm = ts.lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")")
                if tnorm in removals:
                    continue
                kept.append(ts)
            self.main_prompt_text.setPlainText(", ".join(kept))
            self.is_programmatic_change = False

        if new_tags:
            self.is_programmatic_change = True
            current = self.main_prompt_text.toPlainText().strip()
            insert = ", ".join(new_tags)
            if current:
                self.main_prompt_text.setPlainText(f"{insert}, {current}")
            else:
                self.main_prompt_text.setPlainText(insert)
            self.is_programmatic_change = False

        # 3. 캐릭터별 조건부 프롬프트 적용 — 전역 조건식과 같은 뜻(포지티브 기준 쉼표 AND 조건,
        #    태그 단위 replace·remove). neg 는 조건에 넣지 않고 neg add 중복 판정에만 쓴다.
        if char_cond_all_rules:
            from utils.condition_block import apply_rules, split_tags
            all_tags = self._collect_all_tags()
            result = apply_rules(
                char_cond_all_rules, all_tags,
                current_by_location={'neg': split_tags(self.neg_prompt_text.toPlainText())},
                prevent_dupe=True,
            )
            self._apply_condition_result(result)

    def _collect_all_tags(self) -> set[str]:
        """포지티브 칸(캐릭터/작품/본문/선행/후행)의 태그를 정규화해 수집 — 조건 판정용.
        전역 조건식(apply_prompt_rules)과 같이 네거티브는 조건에 넣지 않는다."""
        from utils.condition_block import norm_tag, split_tags
        all_tags: set[str] = set()
        for field in [self.character_input, self.copyright_input,
                      self.main_prompt_text, self.prefix_prompt_text,
                      self.suffix_prompt_text]:
            text = field.text() if hasattr(field, 'text') else field.toPlainText()
            for t in split_tags(text):
                n = norm_tag(t)
                if n:
                    all_tags.add(n)
        return all_tags

    def _apply_condition_result(self, result: dict):
        """apply_rules() 결과를 UI 위젯에 적용 — 태그 단위(utils.condition_block 공용 헬퍼).

        - add: 위치 칸 끝에, 이미 있는 태그는 빼고.
        - remove: main/prefix/suffix 위치 규칙은 포지티브 전 칸에서, neg 규칙은 네거티브에서 뺀다.
        - replace: 조건 태그(앵커)를 대상 태그로 — 포지티브 전 칸을 함께 보고 대상이 이미 있으면
          앵커만 지운다(``_replace_neg`` 는 네거티브 칸에서).
        """
        from utils.condition_block import (
            add_to_tags, remove_from_tags, replace_across, split_tags,
        )
        pos_widgets = {
            "character": self.character_input,
            "copyright": self.copyright_input,
            "prefix": self.prefix_prompt_text,
            "main": self.main_prompt_text,
            "suffix": self.suffix_prompt_text,
        }
        all_widgets = {**pos_widgets, "neg": self.neg_prompt_text}

        def _get(w):
            return w.text() if hasattr(w, 'text') else w.toPlainText()

        tags = {key: split_tags(_get(w)) for key, w in all_widgets.items()}
        before = {key: list(v) for key, v in tags.items()}

        # add 동작: 태그 추가
        for location in ("main", "prefix", "suffix", "neg"):
            add = result.get(location) or []
            if add:
                tags[location] = add_to_tags(tags[location], add)

        # remove 동작: 태그 제거
        pos_remove = []
        for location in ("main", "prefix", "suffix"):
            pos_remove.extend(result.get(f"_remove_{location}") or [])
        if pos_remove:
            for key in pos_widgets:
                tags[key] = remove_from_tags(tags[key], pos_remove)
        if result.get("_remove_neg"):
            tags["neg"] = remove_from_tags(tags["neg"], result["_remove_neg"])

        # replace 동작: 같은 조건의 대상들을 모아 한 번에 교체(첫 대상만 남던 버그 방지)
        def _grouped(pairs):
            groups: dict[str, list[str]] = {}
            for old_tag, new_tag in pairs or []:
                groups.setdefault(old_tag, []).append(new_tag)
            return groups.items()

        pos_keys = list(pos_widgets)
        for cond, targets in _grouped(result.get("_replace")):
            replaced = replace_across([tags[k] for k in pos_keys], cond, targets)
            for key, new_tags in zip(pos_keys, replaced):
                tags[key] = new_tags
        for cond, targets in _grouped(result.get("_replace_neg")):
            tags["neg"] = replace_across([tags["neg"]], cond, targets)[0]

        prev = getattr(self, 'is_programmatic_change', False)
        self.is_programmatic_change = True
        try:
            for key, widget in all_widgets.items():
                if tags[key] != before[key]:
                    text = ", ".join(tags[key])
                    (widget.setText if hasattr(widget, 'text') else widget.setPlainText)(text)
        finally:
            self.is_programmatic_change = prev

    def _apply_conditional_prompts(self):
        """조건부 프롬프트 적용 — Vue 모달이 저장한 config/cond_rules.json을 소스로 사용.
        (자동화 덱 경로 포함 모든 경로. 기존엔 비어있는 블록에디터를 읽어 동작하지 않았다.)"""
        import os
        import json as _json
        try:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'config', 'cond_rules.json')
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                data = _json.load(f)
        except Exception:
            return
        if data.get('enabled') is False:   # 마스터 토글 OFF
            return
        pos = [r for r in data.get('positive', []) if r.get('enabled', True)]
        neg = [r for r in data.get('negative', []) if r.get('enabled', True)]
        if pos or neg:
            # 동작하는 Vue 규칙 변환기 재사용 (add/remove/replace, main/prefix/suffix/neg)
            self._apply_vue_conditional_rules(pos, neg)

    def apply_random_prompt(self) -> bool:
        """덱에서 한 장 뽑아 적용 (등급 필터 반영). 뽑았으면 True.

        덱이 비면 풀에서 다시 채운다(core.search_deck.refill_owner_deck — 등급 필터·셔플·저장).
        그래도 비면(필터를 통과한 행이 없음) pop 하지 않고 알린다 — 자동화 중이면 자동화를 멈춘다.
        (예전엔 여기서 IndexError 가 나 자동화가 조용히 멈췄다)
        """
        automating = bool(getattr(self, 'is_automating', False))
        if not self.shuffled_prompt_deck:
            if not self.filtered_results:
                if automating and hasattr(self, '_stop_automation'):
                    self._stop_automation("검색 결과가 없어 자동화를 중지했습니다.")
                else:
                    QMessageBox.warning(
                        self, "Error",
                        "No prompts available. Run a search first."
                    )
                return False
            from core.search_deck import NO_ELIGIBLE_MESSAGE, refill_owner_deck
            refill_owner_deck(self, emit=False)
            if not self.shuffled_prompt_deck:
                if automating and hasattr(self, '_stop_automation'):
                    self._stop_automation(NO_ELIGIBLE_MESSAGE)
                else:
                    QMessageBox.warning(self, "Notice", NO_ELIGIBLE_MESSAGE)
                return False
            if automating:
                # 자동화 도중 모달을 띄우면 사람이 누를 때까지 멈춘다 — 상태로만 알린다.
                self.show_status(f"🔄 덱을 다시 채웠습니다 ({len(self.shuffled_prompt_deck)})")
            else:
                QMessageBox.information(
                    self, "Notice",
                    "All prompts used once. Reshuffling deck."
                )

        random_bundle = self.shuffled_prompt_deck.pop()
        remaining_count = len(self.shuffled_prompt_deck)
        # 현재 자동화 프롬프트 보존 — 큐 우선 처리(자동화 중 큐 항목)가 UI 프롬프트를
        # 바꿔도 큐 처리 후 이 번들로 복원해 '남은 반복'을 이어가기 위함.
        self._current_auto_bundle = random_bundle
        # 덱 소비 진행도 저장 — 자동화도 이 경로로 뽑는다. 뽑을 때마다 남은 덱 전체(수만 행
        # 인덱스)를 다시 쓰지 않도록 모아서 저장하고(20회·60초·덱 소진), 자동화 중지·앱 종료
        # 때 _flush_deck_state 로 마저 쓴다.
        note_draw = getattr(self, '_note_deck_draw', None)
        if callable(note_draw):
            note_draw()
        elif hasattr(self, '_save_deck_state'):
            self._save_deck_state()
        self.show_status(
            f"Prompt selected. Remaining: {remaining_count}"
        )

        # 버튼 텍스트 업데이트
        self.btn_random_prompt.setText(f"🎲 랜덤 프롬프트 ({remaining_count})")

        # apply_prompt_from_data 호출 (토글 적용됨)
        self.apply_prompt_from_data(random_bundle)
        return True
    
    def on_base_prompts_changed(self):
        """베이스 프롬프트 변경 이벤트"""
        if self.is_programmatic_change:
            return
        self.base_prefix_prompt = self.prefix_prompt_text.toPlainText()
        self.base_suffix_prompt = self.suffix_prompt_text.toPlainText()
        self.base_neg_prompt = self.neg_prompt_text.toPlainText()
