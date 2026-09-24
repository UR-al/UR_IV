"""파라미터 열 확장 카드(.ext-*) 스타일이 자식 패널 안쪽까지 닿는지 — #69.

App.vue 의 ``<style scoped>`` 규칙은 자식 컴포넌트의 루트 요소에만 붙는다. 그래서
AnimaGuidancePanel 은 카드 테두리(.ext-card)만 받고 안쪽 .ext-row(2열 그리드)·.ext-field 라벨
스타일이 빠져 있었다. 이 규칙들은 main.js 가 import 하는 전역 styles/panels.css 에 있어야 한다.

전역 규칙은 파라미터 열(App.vue 의 ``.extend-overlay``) 안의 카드로만 범위를 좁힌다 —
Sam3ControlNetPanel 이 RefinePanel(.glass-card 헤어라인 섹션)·BatchView(.sam3-cn)에도 들어가는데,
범위 없는 전역 .ext-card 가 거기에 App 의 카드 테두리·배경·제목 여백을 새로 입혔었다.
"""
import pathlib
import re
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "src"
PANELS_CSS = (SRC / "styles" / "panels.css").read_text(encoding="utf-8")
APP_PATH = SRC / "App.vue"
APP = APP_PATH.read_text(encoding="utf-8")
# 파라미터 열의 카드 — App.vue 분할(④)로 App 템플릿에서 components/params/*.vue 로 나갔다
PARAMS_DIR = SRC / "components" / "params"
SAM3_MASK_CARD = (PARAMS_DIR / "Sam3MaskCard.vue").read_text(encoding="utf-8")
MAIN_JS = (SRC / "main.js").read_text(encoding="utf-8")
ANIMA = (SRC / "components" / "AnimaGuidancePanel.vue").read_text(encoding="utf-8")
SAM3_CN = (SRC / "components" / "Sam3ControlNetPanel.vue").read_text(encoding="utf-8")
REFINE = (SRC / "components" / "RefinePanel.vue").read_text(encoding="utf-8")
BATCH = (SRC / "views" / "BatchView.vue").read_text(encoding="utf-8")

SHARED = ("ext-title", "ext-field", "ext-row", "ext-check-row", "ext-sub", "ext-sub-title", "ext-toggle-grid")
SCOPE = ":where(.extend-overlay) "


def _scoped_style(text: str) -> str:
    match = re.search(r"<style scoped>(.*?)</style>", text, re.S)
    return match.group(1) if match else ""


def _selectors(css: str) -> list[str]:
    """주석을 뺀 CSS 의 규칙 선택자 목록(쉼표로 나눈 조각)."""
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: list[str] = []
    for match in re.finditer(r"([^{}]+)\{", body):
        out.extend(part.strip() for part in match.group(1).split(",") if part.strip())
    return out


def _template(text: str) -> str:
    return text[: text.index("<script")]


def _params_column(app: str) -> tuple[int, int]:
    """App 템플릿에서 파라미터 열(.extend-overlay)이 차지하는 구간 — 왼쪽 aside 가 닫힐 때까지."""
    template = _template(app)
    start = template.index('class="extend-overlay"')
    return start, template.index("</aside>", start)


class ExtCardStyleTests(unittest.TestCase):
    def test_panels_css_is_global_and_defines_the_card_layout(self):
        self.assertIn("import './styles/panels.css'", MAIN_JS)
        self.assertRegex(PANELS_CSS, r"(?m)^:where\(\.extend-overlay\) \.ext-card\s*\{")
        self.assertRegex(PANELS_CSS, r"\.ext-card \.ext-row\s*\{[^}]*grid-template-columns:\s*1fr 1fr")
        self.assertRegex(PANELS_CSS, r"\.ext-card \.ext-field label\s*\{[^}]*display:\s*block")
        for cls in SHARED:
            self.assertRegex(PANELS_CSS, r"\.ext-card \." + re.escape(cls) + r"\b", cls)

    def test_app_scoped_style_no_longer_owns_the_shared_rules(self):
        scoped = _scoped_style(APP)
        self.assertTrue(scoped)
        self.assertNotRegex(scoped, r"(?m)^\.ext-card\s*\{")
        for cls in SHARED:
            self.assertNotRegex(scoped, r"(?m)^\." + re.escape(cls) + r"[\s{:,]", cls)

    def test_anima_panel_relies_on_the_shared_rules(self):
        self.assertIn('<details class="ext-card">', ANIMA)
        self.assertIn('class="ext-row"', ANIMA)
        self.assertIn('class="ext-field"', ANIMA)
        # 패널 자체에는 공용 규칙 사본을 두지 않는다(.ext-note 는 패널 전용)
        own = _scoped_style(ANIMA)
        for cls in ("ext-row", "ext-field", "ext-check-row"):
            self.assertNotRegex(own, r"(?m)^\." + re.escape(cls) + r"\b", cls)

    def test_no_numeric_font_weight_in_the_moved_rules(self):
        block = PANELS_CSS[PANELS_CSS.index(".ext-card {"):]
        self.assertNotRegex(block, r"font-weight:\s*\d")

    def test_every_shared_card_rule_is_scoped_to_the_params_column(self):
        card_selectors = [s for s in _selectors(PANELS_CSS) if ".ext-card" in s]
        self.assertGreaterEqual(len(card_selectors), len(SHARED) + 1)
        for selector in card_selectors:
            # :where() 는 특이도 0 — 예전 '.ext-card' 하나짜리 우선순위를 그대로 지킨다
            self.assertTrue(selector.startswith(SCOPE + ".ext-card"), selector)
        # 카드 밖에서 쓰이는 이름(.ext-check-row 등)을 범위 없이 전역으로 두지 않는다
        for selector in _selectors(PANELS_CSS):
            for cls in SHARED:
                if re.search(r"\." + re.escape(cls) + r"\b", selector):
                    self.assertTrue(selector.startswith(SCOPE), selector)

    def test_app_cards_and_card_panels_live_in_the_params_column(self):
        template = _template(APP)
        start, end = _params_column(APP)
        for match in re.finditer(r'class="[^"]*\bext-card\b', template):
            self.assertTrue(start < match.start() < end, template[match.start():match.start() + 60])
        # App.vue 분할(④): 카드는 components/params/*.vue 가 됐다. 전역 .ext-* 규칙이 닿으려면 카드는
        # 이 열 안에만 놓여야 한다 — App 에서는 열 안에서만, 다른 곳에서는 다른 파라미터 카드 안에서만 쓴다.
        cards = sorted(PARAMS_DIR.glob("*.vue"))
        self.assertTrue(cards)
        params_names = {p.stem for p in cards}
        for card in cards:
            tag = "<" + card.stem
            in_app = [m.start() for m in re.finditer(re.escape(tag) + r"[\s/>]", template)]
            hosts = [p for p in SRC.rglob("*.vue") if p != APP_PATH and re.search(re.escape(tag) + r"[\s/>]", p.read_text(encoding="utf-8"))]
            self.assertTrue(in_app or hosts, f"{card.name} 를 아무도 쓰지 않는다")
            for pos in in_app:
                self.assertTrue(start < pos < end, card.name)
            for host in hosts:
                self.assertIn(host.stem, params_names, f"{card.name} 가 파라미터 열 밖({host.name})에 놓였다")
        # 카드 모양(.ext-card)을 받는 패널 — Anima 는 열에 바로, SAM3 ControlNet 은 SAM3 카드 안에
        for tag, host_template in (("<AnimaGuidancePanel", template), ("<Sam3ControlNetPanel", _template(SAM3_MASK_CARD))):
            self.assertIn(tag, host_template)
        positions = [m.start() for m in re.finditer(re.escape("<AnimaGuidancePanel"), template)]
        self.assertTrue(positions)
        for pos in positions:
            self.assertTrue(start < pos < end, "<AnimaGuidancePanel")

    def test_params_cards_do_not_copy_the_shared_rules(self):
        # 카드마다 scoped 규칙은 제 모양(해상도·LoRA 블록 …)만 — 공용 .ext-* 사본을 두면 전역 규칙과 갈라진다
        for card in sorted(PARAMS_DIR.glob("*.vue")):
            own = _scoped_style(card.read_text(encoding="utf-8"))
            self.assertNotRegex(own, r"(?m)^\.ext-card\s*\{", card.name)
            for cls in SHARED:
                self.assertNotRegex(own, r"(?m)^\." + re.escape(cls) + r"[\s{:,]", f"{card.name}: {cls}")

    def test_sam3_controlnet_in_other_hosts_keeps_its_own_look(self):
        # 루트가 .ext-card 라 범위 없는 전역 규칙이면 RefinePanel·BatchView 에서도 카드가 된다
        self.assertIn('<details class="ext-card cn-card">', SAM3_CN)
        self.assertIn('<Sam3ControlNetPanel class="glass-card"', REFINE)
        self.assertIn('class="sam3-cn"', BATCH)
        for host in (REFINE, BATCH):
            self.assertNotIn("extend-overlay", _template(host))
        # 패널 자신의 scoped 필드 규칙은 그대로 둔다(App 밖에서는 이것만 받는다)
        own = _scoped_style(SAM3_CN)
        for cls in ("ext-field", "ext-row", "ext-check-row"):
            self.assertRegex(own, r"(?m)^\." + re.escape(cls) + r"\b", cls)


if __name__ == "__main__":
    unittest.main()
