# widgets/__init__.py
"""Widget 모듈 — PyQt 위젯.

재수출하지 않는다: 사용처는 ``from widgets.<모듈> import ...`` 로 직접 가져온다.
예전 재수출은 widgets.common_widgets 하나만 import 해도 interactive_label(→ cv2)과
image_viewer 까지 끌어와 창이 뜨기 전 경로에 OpenCV 로드를 얹었다.
"""