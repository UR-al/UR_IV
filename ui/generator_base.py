# ui/generator_base.py
"""
GeneratorMainUI의 기본 구조 및 초기화
"""
from PyQt6.QtWidgets import QMainWindow

from config import *

class GeneratorBase(QMainWindow):
    def __init__(self):
        super().__init__()

        # 분류기
        self._tag_classifier = None  # 지연 로드

        # 상태 변수들
        self.current_image_path = None
        self.loaded_settings = {}
        self.filtered_results = []
        self.search_worker = None
        self.shuffled_prompt_deck = []
        self.is_automating = False    
        self.is_programmatic_change = False
        
        # 베이스 프롬프트
        self.base_prefix_prompt = ""
        self.base_suffix_prompt = ""
        self.base_neg_prompt = ""
        
        # 랜덤 해상도
        self.random_resolutions = [
            (1024, 1024, "1:1 Square"), 
            (960, 1088, "Portrait"), 
            (896, 1152, "Portrait"),
            (832, 1216, "Tall Portrait"), 
            (1088, 960, "Landscape"), 
            (1152, 896, "Landscape"),
            (1216, 832, "Wide Landscape")
        ]      
    
    @property
    def tag_classifier(self):
        """TagClassifier 지연 로드 — 프로세스 공유 인스턴스(VueBridge 와 같은 객체)"""
        if self._tag_classifier is None:
            from core.tag_classifier import get_tag_classifier
            self._tag_classifier = get_tag_classifier()
        return self._tag_classifier

    # (창 표시 때 프롬프트 칸 높이를 맞추던 showEvent·_auto_adjust_text_edit_height 는 지웠다 —
    #  칸이 모두 Vue 프록시라 contentsMargins 가드에서 늘 먼저 끝나, 표시마다 no-op 타이머 6개만
    #  남겼다(audit #175). 칸 높이는 Vue 가 정한다.)

    def _apply_removal_filters(self, tags_list):
        """제거 옵션 적용 (공통)"""
        result = []

        for tag in tags_list:
            tag_lower = tag.lower().strip()

            # 작가명 제거
            if self.chk_remove_artist.isChecked():
                if tag_lower in self.tag_classifier.artists:
                    continue
                if tag_lower.startswith('artist:'):
                    continue

            # 작품명 제거
            if self.chk_remove_copyright.isChecked():
                if tag_lower in self.tag_classifier.copyrights:
                    continue

            # 캐릭터 제거
            if hasattr(self, 'chk_remove_character') and self.chk_remove_character.isChecked():
                if tag_lower in self.tag_classifier.characters:
                    continue

            # 메타 제거 (parquet 기반)
            if self.chk_remove_meta.isChecked():
                if self.tag_classifier.is_meta_tag(tag):
                    continue

            # 검열 제거
            if hasattr(self, 'chk_remove_censorship') and self.chk_remove_censorship.isChecked():
                if self.tag_classifier.is_censorship_tag(tag):
                    continue

            # 텍스트 제거
            if hasattr(self, 'chk_remove_text') and self.chk_remove_text.isChecked():
                if self.tag_classifier.is_text_tag(tag):
                    continue

            result.append(tag)

        return result        