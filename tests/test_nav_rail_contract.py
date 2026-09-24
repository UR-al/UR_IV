# -*- coding: utf-8 -*-
"""세로 탭 레일 회귀 테스트 — 레일이 가리키는 곳이 실제로 존재하는지.

레일의 '서랍'은 하위 항목을 누르면 왼쪽 패널의 그 섹션으로 스크롤한다. 대상은
`document.getElementById(id)` 로 찾는데, id 가 없으면 **예외가 안 난다** —
눌러도 화면이 가만히 있을 뿐이다. 아이콘도 같다: `icons/index.ts` 에 없는 이름은
빈 `<svg>` 로 렌더돼 버튼이 통째로 비어 보인다. 둘 다 이 저장소의 단골 실패
방식('조용히 안 맞는 것')이라 정적으로 잡는다.

(`test_editor_tools_contract.py` · `test_bridge_contract.py` 와 같은 방식 — Qt 비의존.)
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "src"

SECTIONS_TS = SRC / "utils" / "navSections.ts"
RAIL = SRC / "components" / "NavRail.vue"
APP = SRC / "App.vue"
PROMPT_PANEL = SRC / "components" / "PromptPanel.vue"
ROUTER = SRC / "router.js"
ICONS = SRC / "icons" / "index.ts"
VIEW_MODE_TS = SRC / "composables" / "useViewMode.ts"
CREATOR_VIEW = SRC / "views" / "CreatorStudioView.vue"
BATCH_VIEW = SRC / "views" / "BatchView.vue"
PNG_VIEW = SRC / "views" / "PngInfoView.vue"
I2I_VIEW = SRC / "views" / "I2IView.vue"

#: 레일이 라우트가 아니라 PyQt 화면으로 보내는 탭. 라우터에는 없다.
NATIVE_TABS = ("web", "backend")

def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _declared_routes() -> list[str]:
    """`NAV_SECTIONS` 항목의 route 이름."""
    text = _read(SECTIONS_TS)
    body = text.split("export const NAV_SECTIONS", 1)[1].split("= [", 1)[1].split("\n]", 1)[0]
    return re.findall(r"route:\s*'([a-z0-9]+)'", body)


def _router_names() -> list[str]:
    return re.findall(r"name:\s*'([a-z0-9]+)'", _read(ROUTER))


def _rail_group_names() -> list[str]:
    """레일 묶음(GROUPS)에 적힌 탭 이름 전부, 선언 순서대로."""
    text = _read(RAIL)
    body = text.split("const GROUPS", 1)[1].split("= [", 1)[1].split("\n]", 1)[0]
    names: list[str] = []
    for chunk in re.findall(r"names:\s*\[([^\]]*)\]", body):
        names.extend(re.findall(r"'([a-z0-9]+)'", chunk))
    return names


class NavRailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rail = _read(RAIL)
        cls.app = _read(APP)
        cls.icons = _read(ICONS)

    def test_the_horizontal_tab_bar_is_gone(self):
        """가로 바는 '네비 3안' 시안이었고 본안이 아니다. 남아 있으면 둘이 공존한다."""
        self.assertFalse(
            (SRC / "components" / "TabBar.vue").exists(),
            "components/TabBar.vue 가 아직 있다",
        )
        for path in SRC.rglob("*.vue"):
            self.assertNotIn(
                "TabBar.vue", _read(path), f"{path.name} 이 아직 TabBar 를 import 한다",
            )
        self.assertNotIn("app-header", self.app, "60px 헤더가 남아 있다 — 무대 세로를 그만큼 잃는다")

    def test_rail_is_wired_in_app(self):
        self.assertIn("<NavRail", self.app, "App.vue 가 레일을 쓰지 않는다")
        # 선언한 emit 이 부모에 안 걸리면 눌러도 아무 일이 없다
        # (test_component_emit_contract.py 와 같은 이유).
        self.assertRegex(self.app, r"<NavRail[^>]*@tab-changed=")
        self.assertRegex(
            self.app,
            r"\.app-container\s*\{[^}]*flex-direction:\s*row",
            "레일과 작업 공간이 나란히 서려면 app-container 가 가로 배치여야 한다",
        )

    def test_every_router_tab_is_placed_in_a_group(self):
        """묶음에서 빠진 탭은 화면에서 그냥 사라진다 — 예외도 경고도 없다."""
        placed = _rail_group_names()
        for name in _router_names():
            self.assertIn(name, placed, f"'{name}' 탭이 레일 묶음 어디에도 없다")
        for name in NATIVE_TABS:
            self.assertIn(name, placed, f"네이티브 탭 '{name}' 이 레일에 없다")

    def test_every_tab_icon_exists_in_the_registry(self):
        """없는 아이콘 이름은 빈 <svg> 로 렌더된다 — 접힌 레일이 통째로 빈칸이 된다."""
        body = self.rail.split("const ICON_BY_TAB", 1)[1].split("= {", 1)[1].split("\n}", 1)[0]
        pairs = re.findall(r"([a-z0-9]+):\s*'([a-z-]+)'", body)
        self.assertGreaterEqual(len(pairs), 15, "아이콘 표를 읽지 못했다")
        for tab, icon in pairs:
            key = f"'{icon}':" if "-" in icon else f"{icon}:"
            self.assertIn(key, self.icons, f"'{tab}' 의 아이콘 '{icon}' 이 icons/index.ts 에 없다")

    def test_rail_reads_sections_from_the_registry(self):
        """DOM 을 훑어 추측하면 패널을 손볼 때마다 목록이 조용히 달라진다."""
        self.assertIn("sectionsFor", self.rail, "레일이 navSections 레지스트리를 안 쓴다")
        self.assertIn("currentTab === tab.name", self.rail, "활성 탭만 펼치는 조건이 없다")

    def test_collapsed_rail_keeps_names_reachable(self):
        """52px 로 접히면 이름이 사라진다 — 이름표가 없으면 아이콘만 남아 못 읽는다."""
        self.assertRegex(self.rail, r"\.nav-rail\s*\{[^}]*width:\s*196px")
        self.assertRegex(self.rail, r"\.nav-rail\.collapsed\s*\{[^}]*width:\s*52px")
        # 레일은 세로 스크롤을 하므로 툴팁을 안에 두면 잘린다 — EditorToolbar 와 같은 이유.
        self.assertIn("<Teleport to=\"body\">", self.rail)
        self.assertIn("navRailCollapsed", self.rail, "접힌 상태가 저장되지 않는다")

    def test_token_badge_rule_matches_the_prompt_panel(self):
        """레일 숫자와 카드 숫자가 다르면 어느 쪽을 믿어야 할지 알 수 없다.

        `approxTokens` 는 아직 두 곳에 있다(PromptPanel 쪽이 로컬 함수라 못 가져온다).
        규칙이 갈라지는 순간을 여기서 잡는다.
        """
        rule = "total + Math.max(0, chunks.length - 1)"
        self.assertIn(rule, self.rail)
        self.assertIn(rule, _read(PROMPT_PANEL))

    def test_rail_colors_come_from_tokens(self):
        """하드코딩 hex 는 라이트 모드에서 깨진다 (tests/test_theme_contract.py 참조)."""
        styles = re.findall(r"<style\b[^>]*>(.*?)</style>", self.rail, re.S)
        self.assertTrue(styles, "NavRail 에 <style> 이 없다")
        hexes = re.findall(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})\b", "".join(styles))
        self.assertEqual(hexes, [], f"레일 CSS 에 하드코딩 색이 있다: {hexes}")
        # 활성 행은 두 모드 모두에서 '한 단 올라온 면'이어야 한다 — rgba(255,255,255,.08)
        # 은 라이트에서 안 보인다.
        self.assertRegex(self.rail, r"\.nav\.on\s*\{[^}]*background:\s*var\(--bg-button\)")
        self.assertRegex(self.rail, r"\.sub\.open\s*\{[^}]*background:\s*var\(--accent-dim\)")


_MODE_ITEM = re.compile(
    r"\{\s*id:\s*'(?P<id>[a-z0-9]+)'\s*,"
    r"\s*label:\s*'(?P<label>[^']+)'\s*,"
    r"\s*icon:\s*'(?P<icon>[a-z0-9-]+)'"
    r"(?:\s*,\s*badge:\s*'(?P<badge>[a-z-]+)')?\s*\}"
)
_SCOPE_BLOCK = re.compile(r"^\s{2}(?P<scope>[a-z0-9]+):\s*\[(?P<body>.*?)^\s{2}\]", re.M | re.S)

#: scope → (레일 라우트, 그 모드를 그리는 화면, 화면이 모드를 고르는 식).
#: 여기 없는 scope 가 VIEW_MODES 에 생기면 테스트가 잡는다 — 레일에 항목만 생기고
#: 화면이 안 따라오는 것이 이 구조의 조용한 실패다.
MODE_SCOPES: dict[str, tuple[str, pathlib.Path, str]] = {
    "panel": ("t2i", APP, "panelMode === '{id}'"),
    "i2i": ("i2i", I2I_VIEW, "subTab === '{id}'"),
    "creator": ("creator", CREATOR_VIEW, "creatorMode === '{id}'"),
    "batch": ("batch", BATCH_VIEW, "subTab === '{id}'"),
    "png": ("png", PNG_VIEW, "subTab === '{id}'"),
}


def _view_modes() -> dict[str, list[dict]]:
    text = _read(VIEW_MODE_TS)
    block = text.split("export const VIEW_MODES = {", 1)[1].split("} as const", 1)[0]
    return {
        m.group("scope"): [i.groupdict() for i in _MODE_ITEM.finditer(m.group("body"))]
        for m in _SCOPE_BLOCK.finditer(block)
    }


class ViewModeSectionTests(unittest.TestCase):
    """한 탭 안에서 화면이 통째로 바뀌는 선택(Creator 의 모드, Batch·PNG Info 의
    서브탭)은 레일의 하위 항목이다.

    예전엔 화면 위에 알약 줄·서브탭 줄이 따로 있어 레일 → 그 줄 → 선택 으로
    내비게이션이 3층이었다. 레일로 올려 2층으로 만들었고, 그 대가로 **찾아가기
    항목과 성격이 다른 하위 항목**이 생겼다 — 모드 항목은 스크롤 대상이 아니라
    화면을 바꾼다. 둘이 섞이면서 조용히 깨질 수 있는 자리를 여기서 지킨다.
    """

    @classmethod
    def setUpClass(cls):
        cls.modes = _view_modes()
        cls.rail = _read(RAIL)
        cls.registry = _read(SECTIONS_TS)
        cls.icons = _read(ICONS)

    def test_mode_lists_are_readable(self):
        self.assertEqual(
            set(self.modes), set(MODE_SCOPES),
            "VIEW_MODES 의 scope 와 이 테스트의 MODE_SCOPES 가 다르다 — 새 scope 는 화면과 라우트를 함께 적는다",
        )
        for scope, items in self.modes.items():
            self.assertGreaterEqual(len(items), 2, f"{scope}: 모드가 둘 미만이면 서랍이 아니다")

    def test_every_mode_icon_exists_in_the_registry(self):
        """접힌 레일(52px)에서는 아이콘만 남는다 — 없는 이름은 빈 칸이 된다."""
        for scope, items in self.modes.items():
            for mode in items:
                icon = mode["icon"]
                key = f"'{icon}':" if "-" in icon else f"{icon}:"
                self.assertIn(key, self.icons, f"{scope}/{mode['id']} 의 아이콘 '{icon}' 이 icons/index.ts 에 없다")

    def test_every_mode_has_a_screen(self):
        """레일에 항목만 생기고 화면이 없으면 눌러도 빈 화면이 된다."""
        for scope, (_route, view, expr) in MODE_SCOPES.items():
            text = _read(view)
            for mode in self.modes[scope]:
                self.assertIn(expr.format(id=mode["id"]), text, f"{view.name} 에 '{mode['id']}' 모드 화면이 없다")

    def test_registry_derives_modes_from_the_composable(self):
        """목록을 두 곳에 적으면 레일과 화면이 갈라진다."""
        self.assertRegex(self.registry, r"VIEW_MODES\[scope\][^\n]*\.map\(", "navSections 가 VIEW_MODES 에서 유도하지 않는다")
        for scope, (route, _view, _expr) in MODE_SCOPES.items():
            self.assertIn(f"modeSections('{scope}')", self.registry, f"{scope} 모드가 레지스트리에 없다")
            self.assertIn(f"route: '{route}'", self.registry, f"{route} 탭이 레지스트리에 없다")
        # 항목을 만드는 자리는 하나여야 한다 — 손으로 적어 넣은 목록이 생기면 레일과 화면이 갈라진다.
        self.assertEqual(self.registry.count("VIEW_MODES[scope]"), 1, "모드 항목을 만드는 자리가 둘 이상이다")

    def test_declared_routes_are_real_routes(self):
        known = set(_router_names())
        for name in _declared_routes():
            self.assertIn(name, known, f"'{name}' 은 router.js 에 없는 탭이다")

    def test_mode_sections_are_not_scroll_targets(self):
        """모드 항목의 id 는 목록 안 이름표일 뿐 DOM 대상이 아니다."""
        for scope, (_route, view, _expr) in MODE_SCOPES.items():
            markup = chr(10).join((_read(APP), _read(PROMPT_PANEL), _read(view)))
            for mode in self.modes[scope]:
                self.assertNotIn(f'id="{scope}-mode-{mode["id"]}"', markup, f"{scope}-mode-{mode['id']} 가 마크업에 있다")

    def test_collapsed_rail_still_reaches_the_modes(self):
        """접었다고 모드가 사라지면 그 화면에 갈 길이 아예 없어진다."""
        self.assertIn('v-for="section in drawerFor(tab)"', self.rail, "서랍이 제 탭의 목록을 그리지 않는다 — 닫히는 서랍에 새 탭 항목이 비친다")
        self.assertNotIn("!collapsed && currentTab", self.rail, "서랍이 아직 접힘 여부로 통째로 가려진다")
        self.assertRegex(self.rail, r"\.nav-rail\.collapsed \.sub\s*\{", "접힌 상태의 서랍 항목 CSS 가 없다")

    def test_mode_highlight_comes_from_the_real_mode(self):
        """클릭 기록으로 칠하면 탭을 떠났다 오는 순간 표시가 어긋난다."""
        self.assertIn("viewMode(section.scope).value === section.mode", self.rail, "모드 활성 표시가 실제 모드 값에서 나오지 않는다")

    def test_mode_survives_a_restart(self):
        """레일이 모드를 보여 주는데 앱을 껐다 켜면 처음 값으로 돌아가면 거짓말이 된다."""
        source = _read(VIEW_MODE_TS)
        self.assertIn("localStorage.getItem", source, "저장된 모드를 읽지 않는다")
        self.assertIn("localStorage.setItem", source, "고른 모드를 저장하지 않는다")

    def test_views_no_longer_carry_their_own_switcher(self):
        """모드 줄이 화면에도 남으면 같은 것을 두 군데서 고르게 된다(내비 3층으로 복귀)."""
        creator = _read(CREATOR_VIEW)
        for gone in ("creator-tabs", "AI STUDIO PRO", "CREATOR STUDIO"):
            self.assertNotIn(gone, creator, f"CreatorStudioView 에 '{gone}' 이 남아 있다")
        for view in (BATCH_VIEW, PNG_VIEW, I2I_VIEW):
            self.assertNotIn('class="sub-tabs"', _read(view), f"{view.name} 에 서브탭 줄이 남아 있다 — 레일과 중복된다")

class PanelModeTests(unittest.TestCase):
    """T2I · I2I · Inpaint 의 왼쪽 열 — 프롬프트와 파라미터가 **같은 자리를 번갈아** 쓴다.

    예전엔 파라미터가 왼쪽 열 오른쪽에 440px 오버레이로 떠서 무대를 가렸다.
    이제 레일 서랍(프롬프트 / 파라미터)이 왼쪽 열의 내용을 통째로 바꾼다.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = _read(APP)
        cls.left = cls.app.split('<aside class="side-panel left"', 1)[1].split("</aside>", 1)[0]

    def test_params_live_inside_the_left_column(self):
        """오버레이 시절의 `#sec-params` 가 왼쪽 열 밖에 남으면 서랍을 눌러도 무대만 가린다.

        파라미터 카드는 App.vue 분할(④)로 `components/params/*.vue` 로 나갔다 — `#sec-params` 는
        ParamsBasicCard 의 루트이고, App 은 그 카드를 왼쪽 열의 `.extend-overlay` 안에 놓는다.
        """
        self.assertIn('class="extend-overlay" v-show="panelMode === \'params\'"', self.left)
        self.assertIn('class="panel-scroll" v-show="panelMode === \'prompt\'"', self.left)
        overlay = self.left.split('class="extend-overlay"', 1)[1]
        self.assertIn("<ParamsBasicCard", overlay, "파라미터 카드가 왼쪽 열의 파라미터 모드 안에 없다")
        card = _read(SRC / "components" / "params" / "ParamsBasicCard.vue")
        template = card.split("<template>", 1)[1].split("<script", 1)[0]
        self.assertRegex(template, r'^\s*<div id="sec-params" class="ext-card">', "#sec-params 는 카드의 루트여야 한다")
        # id 는 문서에 하나 — 다른 파일이 같은 id 를 다시 만들거나 카드를 다른 자리에 또 놓으면 안 된다
        owners = [p.name for p in SRC.rglob("*.vue") if 'id="sec-params"' in _read(p)]
        self.assertEqual(owners, ["ParamsBasicCard.vue"])
        users = [p.name for p in SRC.rglob("*.vue") if "<ParamsBasicCard" in _read(p)]
        self.assertEqual(users, ["App.vue"])
        self.assertEqual(self.app.count("<ParamsBasicCard"), 1)

    def test_mode_comes_from_the_shared_ref(self):
        """`showExtendPanel` 은 옛 이름을 지키는 computed 다 — 따로 ref 로 되살리면 레일과 갈라진다."""
        self.assertIn("const panelMode = viewMode('panel')", self.app)
        self.assertIn("const showExtendPanel = computed({", self.app)
        self.assertNotIn("const showExtendPanel = ref(", self.app)

    def test_overlay_chrome_is_gone(self):
        """반달 손잡이 · 배경 클릭 · 절대 위치 — 오버레이였을 때의 장치가 남으면 레이아웃이 어긋난다."""
        for gone in ("half-moon", "extend-backdrop", "goToNavSection", "@go-section"):
            self.assertNotIn(gone, self.app, f"'{gone}' 이 남아 있다")
        css = re.search(r"\.extend-overlay\s*\{([^}]*)\}", self.app)
        self.assertIsNotNone(css, ".extend-overlay 규칙이 없다")
        self.assertNotIn("position: absolute", css.group(1), "파라미터 열이 아직 무대 위에 떠 있다")
        self.assertNotIn("left: 360px", css.group(1))

    def test_stage_announces_the_tab_change(self):
        """레일 표시만 바뀌면 같은 화면으로 읽힌다 — 무대가 살짝 떠오르고, data-tab 으로 어느 탭인지 말한다."""
        self.assertIn('<transition name="stage">', self.app)
        # out-in 은 금지 — 나가는 전환이 끝나야 새 화면이 마운트되는데, 창이 가려져 프레임이
        # 멈추면 그 끝이 오지 않아 화면이 통째로 빈다. 들어올 때만 움직인다.
        self.assertNotIn('name="stage" mode=', self.app, "무대 전환에 mode 를 두지 않는다")
        self.assertNotIn(".stage-leave", self.app, "나가는 전환 규칙을 두지 않는다")
        self.assertIn(':data-tab="activeTabName"', self.app)
        self.assertRegex(self.app, r"\.stage-enter-active[^{]*\{[^}]*transition")


if __name__ == "__main__":
    unittest.main()
