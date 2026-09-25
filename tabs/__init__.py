# tabs/__init__.py
"""Tab 모듈 — Web·Backend 네이티브 PyQt 화면(browser_tab·backend_ui_tab)만 남았다.

나머지 화면은 Vue 가 맡는다. 숨은 레거시 PyQt 탭(EventGen·XYZ·PNG Info·Gallery·Settings·Search·
I2I·Inpaint·Upscale …)은 은퇴했다 — tests/test_legacy_*_retirement.py 가 재등장을 막는다.

재수출하지 않는다: ui/generator_ui_setup.py 등 사용처는 ``from tabs.<모듈> import ...``
로 직접 가져온다. 이 파일을 지우지는 말 것 — tests/test_legacy_ui_retirement.py 가 읽는다.
"""
