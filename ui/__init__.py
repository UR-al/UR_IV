# ui/__init__.py
"""UI 모듈 — 메인 창과 Vue 브리지.

재수출하지 않는다: 진입점(new_main_ui.py, web_main_ui.py)과 테스트는
``from ui.generator_main import GeneratorMainUI`` 로 직접 가져온다. 예전 재수출은
``ui.vue_bridge`` 같은 서브모듈 하나만 import 해도 메인 창 전체(모든 믹스인·탭)를 끌어왔다.
"""