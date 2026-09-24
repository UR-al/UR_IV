# workers/__init__.py
"""Worker 모듈 — QThread 워커.

재수출하지 않는다: 사용처는 ``from workers.<모듈> import ...`` 로 직접 가져온다.
예전 재수출은 workers.* 하나만 import 해도 search_worker(→ pandas)를 끌어와 창이
뜨기 전 경로에 pandas 로드를 얹었다.
"""