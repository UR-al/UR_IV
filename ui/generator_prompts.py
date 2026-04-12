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
        """최종 프롬프트 디스플레이 업데이트"""
        parts = []
        
        # 1. 인물 수
        if self.char_count_input.text().strip(): 
            parts.append(self.char_count_input.text().strip())
            
        # 2. 캐릭터
        if self.character_input.text().strip():
            parts.append(self.character_input.text().strip())

        # 3. 작품 (Copyright)
        if self.copyright_input.text().strip():
            parts.append(self.copyright_input.text().strip())

        # 4. 작가 (Artist) - 작가 필드 내용 추가!
        if self.artist_input.toPlainText().strip():
            parts.append(self.artist_input.toPlainText().strip())

        # 5. 선행 (Prefix) - 접기와 무관하게 항상 포함
        if self.prefix_prompt_text.toPlainText().strip():
            parts.append(self.prefix_prompt_text.toPlainText().strip())

        # 6. 메인
        if self.main_prompt_text.toPlainText().strip():
            parts.append(self.main_prompt_text.toPlainText().strip())

        # 7. 후행 (Suffix) - 접기와 무관하게 항상 포함
        if self.suffix_prompt_text.toPlainText().strip():
            parts.append(self.suffix_prompt_text.toPlainText().strip())

        # 최종 반영
        self.total_prompt_display.setPlainText(", ".join(parts))
    
    def apply_prompt_from_data(self, bundle):
        """검색 결과 데이터를 프롬프트 입력창에 적용"""
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
            return [t.strip() for t in text.split(',') if t.strip()]
        
        # 2. 기본 리스트 생성
        artist_list = [artist_str.replace('artist:', '').strip()] if artist_str else []
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

        def filter_tags(tags):
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
                # 포함
                if any(c in nt for c in norm_contains):
                    continue
                # 접두 (short_ → short로 시작)
                if any(nt.startswith(p) for p in norm_prefix):
                    continue
                # 접미 (_short → short로 끝남)
                if any(nt.endswith(s) for s in norm_suffix):
                    continue
                res.append(t)
            return res

        artist_list = filter_tags(artist_list)
        copyright_list = filter_tags(copyright_list)
        character_list = filter_tags(character_list)
        general_list = filter_tags(general_list)

        # 4. 제거 토글 적용
        
        # 작가 처리 (고정 vs 제거 vs 적용)
        keep_current_artist = False
        if self.btn_lock_artist.isChecked():
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
        _logger.debug(f"tag_classifier 존재: {hasattr(self, 'tag_classifier')}")
        _logger.debug(f"censorship_tags 개수: {len(self.tag_classifier.censorship_tags)}")
        _logger.debug(f"text_tags 개수: {len(self.tag_classifier.text_tags)}")
        
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
            "1other", "2others", "3others", "4others", "5others", "6+others"
        }
        
        final_count = []
        final_general = []
        
        for t in general_list:
            if t.lower() in count_tags: 
                final_count.append(t)
            else: 
                final_general.append(t)
        
        _logger.debug(f"최종 general: {len(final_general)}개")
        
        # 6. 선행/후행 프롬프트와 중복 제거
        def _norm(tag: str) -> str:
            return tag.replace('_', ' ').strip().lower()

        fixed_tags = set()
        for src in (self.prefix_prompt_text.toPlainText(),
                    self.suffix_prompt_text.toPlainText()):
            for t in src.split(','):
                nt = _norm(t)
                if nt:
                    fixed_tags.add(nt)

        if fixed_tags:
            final_general = [t for t in final_general if _norm(t) not in fixed_tags]
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

        # 10. 조건부 프롬프트 1차 적용 (와일드카드 해석 전)
        cond_enabled = (hasattr(self, 'cond_prompt_check') and
                        self.cond_prompt_check.isChecked())
        if cond_enabled:
            self._apply_conditional_prompts()

        # 11. 와일드카드 치환
        wc_enabled = (hasattr(self, 'settings_tab') and
                      hasattr(self.settings_tab, 'chk_wildcard_enabled') and
                      self.settings_tab.chk_wildcard_enabled.isChecked())
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
        if cond_enabled and wc_enabled:
            self._apply_conditional_prompts()

        # 13. 자동 해상도 (Parquet H/W)
        if hasattr(self, 'auto_res_check') and self.auto_res_check.isChecked():
            raw_w = bundle.get('image_width')
            raw_h = bundle.get('image_height')
            if raw_w is not None and raw_h is not None:
                try:
                    import math
                    w = int(float(raw_w))
                    h = int(float(raw_h))
                    # 8의 배수로 반올림
                    w = max(256, min(2048, round(w / 8) * 8))
                    h = max(256, min(2048, round(h / 8) * 8))
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
                        new_tags.append(tag)
                        all_existing.add(norm)

        if new_tags:
            self.is_programmatic_change = True
            current = self.main_prompt_text.toPlainText().strip()
            insert = ", ".join(new_tags)
            if current:
                self.main_prompt_text.setPlainText(f"{insert}, {current}")
            else:
                self.main_prompt_text.setPlainText(insert)
            self.is_programmatic_change = False

        # 3. 캐릭터별 조건부 프롬프트 적용
        if char_cond_all_rules:
            from utils.condition_block import apply_rules
            all_tags = self._collect_all_tags()
            result = apply_rules(char_cond_all_rules, all_tags, prevent_dupe=True)
            self._apply_condition_result(result)

    def _collect_all_tags(self) -> set[str]:
        """모든 위치의 태그를 정규화하여 수집 (네거티브 포함)"""
        all_tags: set[str] = set()
        for field in [self.character_input, self.copyright_input,
                      self.main_prompt_text, self.prefix_prompt_text,
                      self.suffix_prompt_text, self.neg_prompt_text]:
            text = field.text() if hasattr(field, 'text') else field.toPlainText()
            for t in text.split(","):
                n = t.strip().lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")")
                if n:
                    all_tags.add(n)
        return all_tags

    def _apply_condition_result(self, result: dict):
        """apply_rules() 결과를 UI 위젯에 적용"""
        widget_map = {
            "main": self.main_prompt_text,
            "prefix": self.prefix_prompt_text,
            "suffix": self.suffix_prompt_text,
            "neg": self.neg_prompt_text,
        }

        self.is_programmatic_change = True

        # add 동작: 태그 추가
        for location in ("main", "prefix", "suffix", "neg"):
            tags = result.get(location, [])
            if not tags:
                continue
            widget = widget_map.get(location)
            if not widget:
                continue
            current = widget.toPlainText().strip()
            insert = ", ".join(tags)
            if current:
                widget.setPlainText(f"{current}, {insert}")
            else:
                widget.setPlainText(insert)

        # remove 동작: 태그 제거
        for location in ("main", "prefix", "suffix", "neg"):
            remove_tags = result.get(f"_remove_{location}", [])
            if not remove_tags:
                continue
            widget = widget_map.get(location)
            if not widget:
                continue
            current_text = widget.toPlainText()
            current_tags = [t.strip() for t in current_text.split(",") if t.strip()]
            remove_norms = {t.strip().lower().replace("_", " ") for t in remove_tags}
            filtered = [t for t in current_tags
                       if t.strip().lower().replace("_", " ") not in remove_norms]
            widget.setPlainText(", ".join(filtered))

        # replace 동작: 태그 교체
        replacements = result.get("_replace", [])
        if replacements:
            for widget in widget_map.values():
                text = widget.toPlainText()
                for old_tag, new_tag in replacements:
                    old_norm = old_tag.strip().lower().replace("_", " ")
                    tags = [t.strip() for t in text.split(",") if t.strip()]
                    new_tags = []
                    for t in tags:
                        if t.strip().lower().replace("_", " ") == old_norm:
                            new_tags.append(new_tag)
                        else:
                            new_tags.append(t)
                    text = ", ".join(new_tags)
                widget.setPlainText(text)

        self.is_programmatic_change = False

    def _apply_conditional_prompts(self):
        """조건부 프롬프트 규칙 적용 (블록 에디터 기반)"""
        from utils.condition_block import apply_rules

        rules = self.cond_block_editor_pos.get_rules() + self.cond_block_editor_neg.get_rules()
        if not rules:
            return

        all_tags = self._collect_all_tags()
        prevent_dupe = self.cond_prevent_dupe_check.isChecked()
        result = apply_rules(rules, all_tags, prevent_dupe=prevent_dupe)
        self._apply_condition_result(result)

    def apply_random_prompt(self):
        """랜덤 프롬프트 적용 (rating 필터 반영)"""
        if not self.shuffled_prompt_deck:
            if self.filtered_results:
                QMessageBox.information(
                    self, "Notice",
                    "All prompts used once. Reshuffling deck."
                )
                # rating 필터 적용
                rating_filter = getattr(self, '_rating_filter', {'g', 's', 'q', 'e'})
                self.shuffled_prompt_deck = [
                    r for r in self.filtered_results
                    if r.get('rating', 'g') in rating_filter
                ]
                random.shuffle(self.shuffled_prompt_deck)
            else:
                QMessageBox.warning(
                    self, "Error", 
                    "No prompts available. Run a search first."
                )
                return

        random_bundle = self.shuffled_prompt_deck.pop()
        remaining_count = len(self.shuffled_prompt_deck)
        self.show_status(
            f"Prompt selected. Remaining: {remaining_count}"
        )
        
        # 버튼 텍스트 업데이트
        self.btn_random_prompt.setText(f"🎲 랜덤 프롬프트 ({remaining_count})")
        
        # apply_prompt_from_data 호출 (토글 적용됨)
        self.apply_prompt_from_data(random_bundle)
    
    def on_base_prompts_changed(self):
        """베이스 프롬프트 변경 이벤트"""
        if self.is_programmatic_change:
            return
        self.base_prefix_prompt = self.prefix_prompt_text.toPlainText()
        self.base_suffix_prompt = self.suffix_prompt_text.toPlainText()
        self.base_neg_prompt = self.neg_prompt_text.toPlainText()
    
    def load_base_prompt_to_event(self):
        """현재 메인 프롬프트를 이벤트 탭으로 복사"""
        parts = []
        
        if self.char_count_input.text().strip(): 
            parts.append(self.char_count_input.text().strip())
            
        if self.character_input.text().strip():
            parts.append(self.character_input.text().strip())

        if self.copyright_input.text().strip():
            parts.append(self.copyright_input.text().strip())

        if self.artist_input.toPlainText().strip():
            parts.append(self.artist_input.toPlainText().strip())
            
        if self.main_prompt_text.toPlainText().strip():
            parts.append(self.main_prompt_text.toPlainText().strip())

        base_prompt = ", ".join(parts)
        self.event_gen_tab.base_prompt_view.setPlainText(base_prompt)
    
    def get_prompts_from_bundle(self, bundle):
        """자동화용 프롬프트 생성"""
        general_tags_str = str(bundle.get('general', ''))
        artist = str(bundle.get('artist', ''))
        copyright_tags = str(bundle.get('copyright', ''))
        character_tags = str(bundle.get('character', ''))
        
        if general_tags_str == 'nan': general_tags_str = ''
        if artist == 'nan': artist = ''
        if copyright_tags == 'nan': copyright_tags = ''
        if character_tags == 'nan': character_tags = ''

        def escape_tags(tag_str):
            if not tag_str: 
                return []
            tags = [t.strip() for t in tag_str.split(',') if t.strip()]
            return [t.replace('(', r'\(').replace(')', r'\)') for t in tags]

        artist_list = []
        if not self.chk_remove_artist.isChecked() and artist:
            clean_artist = artist.replace('artist:', '').strip()
            artist_list = escape_tags(clean_artist)

        copyright_list = []
        if not self.chk_remove_copyright.isChecked() and copyright_tags:
            copyright_list = escape_tags(copyright_tags)

        character_list = escape_tags(character_tags)
        if self.chk_remove_character.isChecked():
            character_list = []
        general_list = escape_tags(general_tags_str)
        if (hasattr(self, 'chk_remove_character_features') and
                self.chk_remove_character_features.isChecked()):
            general_list = [t for t in general_list if not self._is_character_feature_tag(t)]
        if self.chk_remove_meta.isChecked():
            general_list = [t for t in general_list if not self.tag_classifier.is_meta_tag(t)]
        if self.chk_remove_censorship.isChecked():
            general_list = [t for t in general_list if not self.tag_classifier.is_censorship_tag(t)]
        if self.chk_remove_text.isChecked():
            general_list = [t for t in general_list if not self.tag_classifier.is_text_tag(t)]

        person_count_tags = {
            "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", 
            "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
            "1other", "2others", "3others", "4others", "5others", "6+others"
        }
        
        count_list = []
        main_list = []
        
        for tag in general_list:
            if tag in person_count_tags:
                count_list.append(tag)
            else:
                main_list.append(tag)

        final_parts = []
        
        if count_list: 
            final_parts.append(", ".join(count_list))
        if character_list: 
            final_parts.append(", ".join(character_list))
        if copyright_list: 
            final_parts.append(", ".join(copyright_list))
        if artist_list: 
            final_parts.append(", ".join(artist_list))
        
        prefix = self.prefix_prompt_text.toPlainText().strip()
        if prefix: 
            final_parts.append(prefix)
        
        if main_list: 
            final_parts.append(", ".join(main_list))
        
        suffix = self.suffix_prompt_text.toPlainText().strip()
        if suffix: 
            final_parts.append(suffix)

        final_prompt = ", ".join(final_parts)
        final_neg = self.neg_prompt_text.toPlainText().strip()

        # 와일드카드 치환
        wc_enabled = (hasattr(self, 'settings_tab') and
                      hasattr(self.settings_tab, 'chk_wildcard_enabled') and
                      self.settings_tab.chk_wildcard_enabled.isChecked())
        if wc_enabled:
            from utils.file_wildcard import resolve_file_wildcards
            from utils.wildcard import process_wildcards
            final_prompt = resolve_file_wildcards(final_prompt)
            final_prompt = process_wildcards(final_prompt)
            final_neg = resolve_file_wildcards(final_neg)
            final_neg = process_wildcards(final_neg)

        return final_prompt, final_neg
        
