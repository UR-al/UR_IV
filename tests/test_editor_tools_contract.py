"""세로 툴바 도구 레지스트리 회귀 테스트.

`frontend/src/utils/editorTools.ts` 의 도구 id 는 `EditorCanvas` 가 문자열로 분기하는
값과 정확히 같아야 한다. 어긋나면 예외 없이 **아무 일도 일어나지 않는다** —
도구를 눌렀는데 커서만 바뀌고 캔버스는 반응이 없다. 이 프로젝트의 단골 실패 방식이라
정적으로 잡는다.
"""

from __future__ import annotations

import pathlib
import re
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "src"
REGISTRY = SRC / "utils" / "editorTools.ts"
CANVAS = SRC / "components" / "editor" / "EditorCanvas.vue"
DRAW_TOOLS = SRC / "utils" / "drawTools.ts"
EDITOR_VIEW = SRC / "views" / "EditorView.vue"

_TOOL = re.compile(
    r"\{\s*id:\s*'(?P<id>[a-z_]+)'.*?"
    r"label:\s*'(?P<label>[^']+)'.*?"
    r"icon:\s*'(?P<icon>[a-z-]+)'.*?"
    r"shortcut:\s*'(?P<key>[A-Z])'.*?"
    r"kind:\s*'(?P<kind>[a-z]+)'",
    re.S,
)


def _tools() -> list[dict]:
    """배열 리터럴 본문만 잘라 파싱한다.

    선언이 `EDITOR_TOOLS: EditorTool[] = [` 라서 첫 `]` 로 자르면 타입의 대괄호에
    걸려 본문이 빈 문자열이 된다 — 줄 맨 앞의 `]` 를 끝으로 본다.
    """
    text = REGISTRY.read_text(encoding="utf-8")
    after = text.split("export const EDITOR_TOOLS", 1)[1]
    body = after.split("= [", 1)[1].split("\n]", 1)[0]
    return [m.groupdict() for m in _TOOL.finditer(body)]


class ToolRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _tools()
        cls.canvas = CANVAS.read_text(encoding="utf-8")
        cls.draw = DRAW_TOOLS.read_text(encoding="utf-8")
        cls.icons = (SRC / "icons" / "index.ts").read_text(encoding="utf-8")

    def test_registry_is_not_empty(self):
        self.assertGreaterEqual(len(self.tools), 15, "도구 레지스트리를 읽지 못했다")

    def test_every_tool_id_is_understood_by_the_canvas(self):
        """마스크 도구는 EditorCanvas 가 직접 분기하고, 그리기 도구는 DRAW_TOOLS 에 있다."""
        for tool in self.tools:
            tool_id = tool["id"]
            if tool["kind"] == "draw":
                self.assertIn(
                    f"'{tool_id}'", self.draw,
                    f"'{tool_id}' 가 utils/drawTools.ts 의 DRAW_TOOLS 에 없다 — "
                    "isDrawTool 이 false 라 포인터가 마스크 처리로 새어 나간다.",
                )
            else:
                self.assertRegex(
                    self.canvas,
                    r"props\.tool\s*===\s*'" + re.escape(tool_id) + r"'",
                    f"EditorCanvas 가 '{tool_id}' 를 분기하지 않는다 — 골라도 아무 일이 없다.",
                )

    def test_shortcuts_are_unique(self):
        keys = [t["key"] for t in self.tools]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        self.assertEqual(dupes, [], f"단축키가 겹친다: {dupes}")

    def test_every_icon_exists_in_the_registry(self):
        """없는 아이콘 이름은 빈 <svg> 로 렌더된다 — 버튼이 통째로 비어 보인다."""
        for tool in self.tools:
            name = tool["icon"]
            key = f"'{name}':" if "-" in name else f"{name}:"
            self.assertIn(key, self.icons, f"아이콘 '{name}' 이 icons/index.ts 에 없다")

    def test_toolbar_is_wired_in_the_editor(self):
        text = EDITOR_VIEW.read_text(encoding="utf-8")
        self.assertIn("<EditorToolbar", text, "EditorView 가 툴바를 쓰지 않는다")
        self.assertRegex(
            text, r"<EditorToolbar[^>]*:model-value=", "툴바에 현재 도구가 전달되지 않는다"
        )
        self.assertRegex(
            text, r"<EditorToolbar[^>]*@select=", "툴바의 선택이 아무데도 연결되지 않았다"
        )

    def test_inpaint_reuses_the_same_toolbar_and_registry(self):
        """Inpaint 도 같은 툴바·같은 도구 id·같은 단축키를 쓴다.

        두 탭에서 같은 도구가 다르게 생기거나 다른 키로 잡히면 손이 헷갈린다.
        """
        registry = REGISTRY.read_text(encoding="utf-8")
        self.assertIn("export const INPAINT_TOOLS", registry)

        view = (SRC / "views" / "InpaintView.vue").read_text(encoding="utf-8")
        self.assertIn("<EditorToolbar", view, "Inpaint 가 세로 툴바를 쓰지 않는다")
        self.assertIn(':tools="INPAINT_TOOLS"', view, "Inpaint 가 자기 도구 목록을 넘기지 않는다")
        self.assertIn("toolByKey(", view, "Inpaint 에 도구 단축키가 없다 — 툴팁이 키를 보여주는데 안 먹으면 거짓말이다")

    def test_effect_preview_is_wired(self):
        """효과는 '적용해야만 결과를 아는' 마지막 자리였다 — 프리뷰를 붙였다.

        백엔드가 마스크 있는 요청을 프리뷰에서 제외하면 프론트 배선이 있어도 조용히 죽는다.
        """
        panel = (SRC / "components" / "editor" / "EffectPanel.vue").read_text(encoding="utf-8")
        self.assertIn("'effect-preview'", panel)

        view = (SRC / "views" / "EditorView.vue").read_text(encoding="utf-8")
        self.assertRegex(view, r"<EffectPanel[^>]*@effect-preview=", "효과 프리뷰가 부모에 연결되지 않았다")
        self.assertIn("scheduleMaskPreview", view, "마스크를 실은 프리뷰 경로가 없다")

        bridge = (pathlib.Path(__file__).resolve().parents[1] / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        self.assertNotRegex(
            bridge,
            r"is_preview\s*=\s*bool\(params\.get\('preview'\)\)\s*and\s*not\s*params\.get\('mask_base64'\)",
            "백엔드가 마스크 있는 프리뷰를 다시 막고 있다 — 마스크는 바로 아래에서 이미 리사이즈된다",
        )

    def test_selection_dependent_actions_are_gated(self):
        """자르기·영역 이동은 선택 영역이 있어야 성립한다.

        예전에는 빈 상태로도 눌렸고, 자르기만 토스트를 띄우고 나머지는 조용했다.
        """
        view = (SRC / "views" / "EditorView.vue").read_text(encoding="utf-8")
        self.assertIn("const hasSelection = computed", view)
        for component in ("TransformPanel", "MovePanel"):
            # 여는 태그 안에 :has-selection 이 있는지 — 태그가 여러 줄이라 DOTALL 로 본다
            self.assertRegex(
                view,
                r"<" + component + r"\b[^>]*?:has-selection=",
                f"{component} 에 선택 여부가 전달되지 않는다",
            )

    def test_draw_tool_selection_syncs_canvas_and_panel(self):
        """캔버스와 옵션 패널은 같은 도구 값(currentTool) 하나를 본다.

        예전에는 selectTool 이 패널의 `setTool` 을 같은 tick 에 불렀는데, 마스크→그리기로
        넘어갈 때 패널이 v-if 로 막 마운트되는 중이라 ref 가 null 이었다 — 패널은 '펜'으로
        떠서 '복원 적용' 버튼이 숨고, 색을 누르면 도구가 펜으로 바뀌었다. 이제 패널은
        제어 컴포넌트다: 도구·파라미터를 props 로 받고, 도구를 올려 보내지 않는다.
        """
        text = EDITOR_VIEW.read_text(encoding="utf-8")
        select = re.search(r"function selectTool\(.*?\n\}", text, re.S)
        self.assertIsNotNone(select, "selectTool 을 찾지 못했다")
        body = select.group(0)
        self.assertIn("currentTool.value = id", body)
        self.assertNotIn("setTool", body, "마운트 중인 패널의 ref 를 부르는 방식은 null 이라 무시된다")

        # 캔버스가 보는 그리기 도구는 currentTool 에서 온다
        self.assertRegex(
            text, r"canvasDrawParams\s*=\s*computed\([\s\S]*?tool:\s*currentTool\.value",
            "캔버스 그리기 파라미터의 도구가 currentTool 과 따로 논다",
        )

        from tests.test_component_emit_contract import _open_tags  # 따옴표 안의 '>' 를 견디는 태그 파서
        panels = {
            "DrawPanel": (":tool=", ":params=", ":layer-opacity="),
            "MaskToolOptions": (":tool=", ":tool-size=", ":stamp-shape=", ":eraser-mode=",
                                ":eraser-restore=", ":magnetic="),
        }
        for component, attrs in panels.items():
            tags = _open_tags(text, component)
            self.assertTrue(tags, f"EditorView 가 {component} 를 쓰지 않는다")
            for body in tags:
                for attr in attrs:
                    self.assertIn(attr, body, f"{component} 에 {attr} 가 없다 — 재마운트 때 표시가 어긋난다")
                self.assertNotIn("@tool-changed", body, f"{component} 는 도구를 올려 보내지 않는다")

        draw_panel = (SRC / "components" / "editor" / "DrawPanel.vue").read_text(encoding="utf-8")
        self.assertNotRegex(draw_panel, r"'tool-changed'", "DrawPanel 이 아직 도구를 소유한다")
        self.assertNotRegex(draw_panel, r"const selectedTool\s*=\s*ref", "DrawPanel 이 아직 도구를 로컬 상태로 든다")


def _function_body(text: str, name: str) -> str:
    """`function name(...) { ... }` 본문 — 줄 맨 앞의 `}` 까지(최상위 함수 전제)."""
    match = re.search(r"(?:async\s+)?function " + re.escape(name) + r"\(.*?\n\}", text, re.S)
    if not match:
        raise AssertionError(f"{name} 을 찾지 못했다")
    return match.group(0)


class EditorPreviewAndSaveGuards(unittest.TestCase):
    """에디터 프리뷰·저장 수정의 회귀 가드 — 되돌려도 빌드는 통과하고 화면에서만 깨지는 것들."""

    @classmethod
    def setUpClass(cls):
        cls.view = EDITOR_VIEW.read_text(encoding="utf-8")
        cls.canvas = CANVAS.read_text(encoding="utf-8")

    def test_preview_is_a_separate_layer_not_the_base_image(self):
        """프리뷰(1024px 축소본)를 image-src 로 넘기면 캔버스가 새 원본으로 받아 마스크·드로잉
        레이어·복원 스냅숏을 초기화했다(큰 이미지에서 '적용할 영역이 없습니다')."""
        from tests.test_component_emit_contract import _open_tags
        tags = _open_tags(self.view, "EditorCanvas")
        self.assertEqual(len(tags), 1, "EditorView 의 EditorCanvas 태그를 찾지 못했다")
        tag = tags[0]
        self.assertRegex(tag, r':image-src="imageDisplay"', "원본 자리에 확정 이미지가 아닌 것이 들어간다")
        self.assertRegex(tag, r':preview-src="previewSrc"', "프리뷰가 별도 레이어로 전달되지 않는다")
        self.assertNotRegex(tag, r':image-src="canvasSrc"', "프리뷰가 다시 원본을 갈아 끼운다")
        self.assertNotRegex(tag, r':image-src="previewSrc', "프리뷰가 다시 원본을 갈아 끼운다")

        self.assertRegex(self.canvas, r"previewSrc\?:\s*string", "EditorCanvas 에 previewSrc prop 이 없다")
        watch = re.search(r"watch\(\(\)\s*=>\s*props\.previewSrc,[^\n]*", self.canvas)
        self.assertIsNotNone(watch, "previewSrc 감시가 없다")
        self.assertIn("showPreview", watch.group(0))
        self.assertNotIn("loadNewImage", watch.group(0), "프리뷰가 원본 로드 경로로 들어간다")
        for name in ("showPreview", "paintPreview", "hidePreviewNow"):
            body = _function_body(self.canvas, name)
            self.assertNotIn("loadNewImage", body, f"{name} 가 원본을 다시 로드한다")
            self.assertNotIn("drawLayer.resize", body, f"{name} 가 드로잉 레이어를 초기화한다")
            self.assertNotIn("maskData =", body, f"{name} 가 마스크를 새로 잡는다")

    def test_preview_hide_waits_for_the_committed_image(self):
        """확정 결과가 오면 새 원본이 그려질 때까지 프리뷰를 남긴다 — 적용마다 원본이 번쩍였다."""
        body = _function_body(self.canvas, "showPreview")
        self.assertIn("shouldDeferPreviewHide", body)
        load = _function_body(self.canvas, "loadNewImage")
        self.assertIn("hidePreviewOnBaseLoad", load, "미뤄 둔 프리뷰 걷기를 원본 로드가 끝낼 때 처리하지 않는다")
        self.assertIn("img.onerror", load, "원본 로드 실패 때 미뤄 둔 걷기가 영원히 남는다")

    def test_effect_preview_guard_uses_selection_bounds(self):
        """getMaskBase64() 가드는 빈 마스크도 PNG 를 돌려줘 늘 참이었고, 틱마다 원본 해상도로 인코딩했다."""
        body = _function_body(self.view, "previewEffect")
        self.assertIn("getSelection", body)
        self.assertNotIn("getMaskBase64", body, "슬라이더 틱마다 전체 해상도 마스크 PNG 인코딩으로 돌아갔다")

    def test_image_size_probe_is_cache_busted(self):
        """저장이 같은 경로를 갱신한 뒤 다시 열면, 무력화 없는 URL 은 옛 크기를 보고했다."""
        body = _function_body(self.view, "_applyImageSize")
        self.assertNotRegex(body, r"mediaUrl\(path\)", "크기 프로브가 캐시된 옛 그림을 읽는다")
        self.assertRegex(body, r"mediaUrl\(path,\s*true\)")
        self.assertIn("_docGen", body, "늦게 끝난 프로브가 새 문서의 크기를 덮을 수 있다")

    def test_save_results_are_gated_by_document_generation(self):
        """저장 중 붙여넣기/열기로 문서가 바뀌면 옛 결과가 새 문서의 sourcePath 를 덮었다."""
        for name in ("loadImage", "resetEditor"):
            self.assertIn("_docGen++", _function_body(self.view, name), f"{name} 가 문서 세대를 올리지 않는다")
            self.assertIn("_sourceOwned = false", _function_body(self.view, name),
                          f"{name} 뒤의 저장이 원본을 '내 사본'으로 덮어쓸 수 있다")
        handler = _function_body(self.view, "onEditorSaveResult")
        self.assertRegex(handler, r"saveResultAction\([^)]*_docGen\)")
        self.assertIn("'stale'", handler)
        request = _function_body(self.view, "_requestSave")
        self.assertIn("docGen: _docGen", request)
        self.assertIn("overwrite_source", request)

    def test_recovery_opens_a_working_copy(self):
        """복구 파일을 그대로 열면 저장 뒤 복구본 정리가 편집 중인 그림을 지웠다."""
        check = _function_body(self.view, "_checkAutoSaveRecovery")
        self.assertNotIn("loadImage(r.path)", check)
        self.assertIn("editorRecoverAutoSave", _function_body(self.view, "_openRecoveredWork"))
        self.assertIn("referencesAutosaveFile", _function_body(self.view, "_clearAutoSaveRecovery"))

    def test_autosave_sends_the_unmerged_drawing_layer(self):
        """자동저장이 확정 이미지 경로만 보내면 병합 안 한 드로잉이 복구본에서 빠진다."""
        body = _function_body(self.view, "_tryAutoSave")
        self.assertIn("getDrawOverlayBase64", body, "자동저장이 드로잉 레이어를 싣지 않는다")
        self.assertIn("drawLayerOpacity", body, "자동저장이 레이어 불투명도를 싣지 않는다")
        self.assertRegex(body, r"requestEditorAutoSave\(\s*path,\s*overlay,[^\n]*,\s*id\)",
                         "requestEditorAutoSave 인자 순서가 슬롯(path, overlay, opacity, requestId)과 다르다")

    def test_autosave_runs_off_the_gui_thread(self):
        """예전 동기 슬롯 editorAutoSave 는 레이어 합성(디코드·합성·PNG 인코딩·fsync)을 GUI 스레드에서
        해 5분마다 창을(웹 모드는 모든 클라이언트를) 1초 가까이 멈췄다. 이제 요청 → 워커 → 이벤트다."""
        body = _function_body(self.view, "_tryAutoSave")
        self.assertNotRegex(body, r"\beditorAutoSave\(", "동기 슬롯을 다시 부른다")
        self.assertIn("autoSaveRequest.begin()", body)
        self.assertIn("await done", body, "응답(editorAutoSaveReady)을 기다리지 않는다")
        self.assertRegex(self.view, r"onBackendEvent\('editorAutoSaveReady',[^\n]*autoSaveRequest\.receive\(")
        # 저장 성공 → 복구본 폐기: 쓰는 중이던 자동저장의 늦은 응답이 '자동저장 N초 전'을 되살리지 않게
        self.assertIn("autoSaveRequest.cancel()", _function_body(self.view, "_clearAutoSaveRecovery"))
        bridge = (SRC.parent.parent / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        self.assertNotRegex(bridge, r"def editorAutoSave\(", "GUI 스레드에서 합성하는 동기 슬롯이 남았다")
        slot = re.search(r"def requestEditorAutoSave\(.*?\n    @pyqtSlot", bridge, re.S)
        self.assertIsNotNone(slot, "requestEditorAutoSave 슬롯을 찾지 못했다")
        self.assertIn("_run_async_lookup(", slot.group(0), "자동저장이 워커로 가지 않는다")
        self.assertNotIn("write_autosave_snapshot", slot.group(0), "슬롯(GUI 스레드)이 직접 합성한다")

    def test_document_switch_resets_the_canvas_pristine_snapshot(self):
        """화면의 모자이크 지우개는 캔버스 스냅숏에서 칠한다. 문서가 바뀌면 옛 문서의 스냅숏과
        디코드 중인 '적용 전' 그림을 버린다 — 같은 크기의 B 위에 A 의 픽셀을 칠하지 않게."""
        self.assertIn("canvasRef.value?.resetPristine?.()", _function_body(self.view, "_resetDocTransients"))
        expose = re.search(r"defineExpose\(\{.*?\}\)", self.canvas, re.S)
        self.assertIsNotNone(expose)
        self.assertRegex(expose.group(0), r"\bresetPristine\b", "EditorCanvas 가 resetPristine 을 노출하지 않는다")
        reset = _function_body(self.canvas, "resetPristine")
        for needle in ("pristineSource.reset()", "pristineImg = null", "clearRestoreMask()"):
            self.assertIn(needle, reset)

    def test_eraser_snapshot_comes_from_the_restore_source_file(self):
        """지우개 화면과 복원 커밋이 같은 파일(pristinePath)을 본다.

        예전에는 캔버스가 로드한 이미지로 스냅숏을 따로 뜨고 효과 직후 '다음 로드 한 번만' 유지했다.
        pristinePath 는 효과마다 앞으로 가는데 스냅숏은 첫 효과 전 원본에 머물러, 효과를 두 번 적용한
        뒤 첫 효과 자리를 지우면 화면에선 지워졌다가 커밋 뒤 되살아났다(효과→조정, redo 뒤에도 갈렸다).
        런타임 검증: frontend/src/components/editor/EditorCanvas.lateDecode.test.ts ·
        utils/pristineSnapshot.test.ts (vitest).
        """
        from tests.test_component_emit_contract import _open_tags
        tags = _open_tags(self.view, "EditorCanvas")
        self.assertEqual(len(tags), 1)
        self.assertIn(':pristine-src="pristineDisplay"', tags[0], "캔버스가 커밋과 같은 '적용 전' 파일을 받지 않는다")
        self.assertRegex(
            self.view,
            r"const pristineDisplay = computed\(\(\) => \(pristinePath\.value \? mediaUrl\(pristinePath\.value, true\) : ''\)\)",
        )
        self.assertIn("pristinePath.value = imagePath.value", _function_body(self.view, "applyEffect"))
        for text, where in ((self.view, "EditorView"), (self.canvas, "EditorCanvas")):
            self.assertNotIn("keepPristineForNextLoad", text, f"{where} 에 '다음 로드 한 번 유지' 예약이 남았다")
            self.assertNotIn("PristineKeeper", text)
        self.assertRegex(self.canvas, r"pristineSrc\?:\s*string", "EditorCanvas 에 pristineSrc prop 이 없다")
        self.assertRegex(self.canvas, r"watch\(\(\)\s*=>\s*props\.pristineSrc,[^\n]*loadPristine\(")
        self.assertNotIn("pristineImg =", _function_body(self.canvas, "loadNewImage"),
                         "원본 로드가 스냅숏을 다시 뜬다 — pristinePath 와 갈린다")
        # 스냅숏은 loadPristine(디코드 완료)에서만 채워진다 — 나머지는 비우기뿐
        filled = [m.start() for m in re.finditer(r"pristineImg = (?!null)", self.canvas)]
        self.assertEqual(len(filled), 1, "pristineImg 를 채우는 곳이 loadPristine 밖에도 있다")
        pristine = _function_body(self.canvas, "loadPristine")
        self.assertIn("pristineImg = pc", pristine)
        self.assertIn("pristineSource.request(", pristine)
        self.assertLess(pristine.find("pristineSource.accepts(token)"), pristine.find("pristineImg = pc"),
                        "늦게 끝난 옛 출처(또는 옛 문서)의 디코드를 거르지 않는다")
        self.assertIn("restoreStrokeMode(pristineSource.requested", _function_body(self.canvas, "restoreLine"))

    def test_late_mask_decode_cannot_land_on_new_document(self):
        """자동 감지 마스크 data URL 의 디코드가 끝나기 전에 다른 문서를 열면(또는 이미지가 바뀌거나 Esc),
        onload 가 새 이미지 위에 옛 문서의 마스크를 썼다 — 요청 시점의 토큰·원본을 확인한다.
        런타임 검증: frontend/src/components/editor/EditorCanvas.lateDecode.test.ts (vitest)."""
        body = _function_body(self.canvas, "loadMaskFromBase64")
        onload = body.find("img.onload")
        self.assertGreater(onload, 0)
        token = body.find("++maskLoadToken")
        self.assertGreater(token, 0, "마스크 요청 토큰이 없다")
        self.assertLess(token, onload)
        before = body[:onload]
        for needle in ("token !== maskLoadToken", "imgToken !== imageLoadToken", "sourceImg !== forImg"):
            self.assertIn(needle, before)
        handler = body[onload:]
        self.assertGreater(handler.find("stale()"), 0, "onload 가 늦은 디코드를 거르지 않는다")
        self.assertLess(handler.find("stale()"), handler.find("maskData["), "토큰 확인이 마스크 쓰기보다 늦다")
        for name in ("clearSelection", "loadNewImage", "cancelMaskLoad"):
            self.assertIn("maskLoadToken++", _function_body(self.canvas, name), f"{name} 가 디코드 중인 마스크를 버리지 않는다")
        expose = re.search(r"defineExpose\(\{.*?\}\)", self.canvas, re.S)
        self.assertRegex(expose.group(0), r"\bcancelMaskLoad\b")
        self.assertIn("canvasRef.value?.cancelMaskLoad?.()", _function_body(self.view, "_resetDocTransients"))

    def test_refused_restore_repaints_the_canvas(self):
        """되돌릴 '적용 전' 이미지가 없어 커밋을 거절하면, 지우개가 화면에 칠한 픽셀은 파일에 없다."""
        body = _function_body(self.view, "commitRestore")
        refusal = body.split("if (!pristinePath.value) {", 1)[1].split("return", 1)[0]
        self.assertIn("clearRestoreMask", refusal)
        self.assertIn("drawAll", refusal, "거절한 복원이 화면에 남는다")

    def test_editor_results_are_gated_by_document_generation(self):
        """A 의 느린 작업 결과가 새로 연 B(또는 닫은 에디터)에 들어가던 회귀 가드.
        job_id 가드는 B 가 아직 작업을 안 했으면 옛 결과를 통과시킨다."""
        handler = _function_body(self.view, "onEditorResult")
        gate = handler.find("editorResultForDoc(result, _docGen)")
        self.assertGreater(gate, 0, "onEditorResult 가 문서 세대를 확인하지 않는다")
        self.assertLess(gate, handler.find("result.path"), "세대 확인이 이미지 결과 적용보다 늦다")
        self.assertLess(gate, handler.find("result.preview"), "세대 확인이 프리뷰 처리보다 늦다")
        for name in ("doOp", "runAutoCensor", "runAutoDetect"):
            self.assertIn("doc_gen: gen", _function_body(self.view, name), f"{name} 가 문서 세대를 보내지 않는다")
        # 문서가 바뀌면 옛 문서의 감지 상태·지우개 원본·프리뷰도 비운다
        for name in ("loadImage", "resetEditor"):
            self.assertIn("_resetDocTransients()", _function_body(self.view, name))
        reset = _function_body(self.view, "_resetDocTransients")
        for needle in ("clearPreview()", "detectStatus.value = ''", "pristinePath.value = ''"):
            self.assertIn(needle, reset)

    def test_dirty_state_tracks_the_drawing_layer_opacity(self):
        """저장한 레이어의 불투명도만 바꾸면 저장본과 화면이 달라지는데 미저장 표시가 안 떴다."""
        dirty = re.search(r"const isDirty = computed\(\(\) => isEditorDirty\(\{.*?\}, savedMarker\.value\)\)",
                          self.view, re.S)
        self.assertIsNotNone(dirty, "isDirty computed 를 찾지 못했다")
        self.assertIn("drawOpacity: drawLayerOpacity.value", dirty.group(0))
        request = _function_body(self.view, "_requestSave")
        self.assertIn("drawOpacity: opacity", request)
        self.assertIn("overlay_opacity: opacity", request, "보낸 불투명도와 기록한 불투명도가 어긋날 수 있다")
        self.assertIn("drawOpacity: pending.drawOpacity", _function_body(self.view, "onEditorSaveResult"))


if __name__ == "__main__":
    unittest.main()
