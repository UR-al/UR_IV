"""3×3 격자 변형 — 얇은 판 스플라인(thin-plate spline)으로 부드럽게 늘이기 (Qt 비의존).

``VueBridge._editor_process_impl`` 의 ``mesh_warp`` 본체. 원근 보정(꼭짓점 4개) 다음 단계로,
3×3 칸(점 4×4 = 16개)을 끌어 일부만 늘이거나 줄인다. 공개된 TPS 식만으로 numpy 로 새로 짰다
(scipy 는 requirements 에 없다).

좌표
* 페이로드는 정규화 좌표 ``(u, v) ∈ [0, 1]`` — 프리뷰와 확정이 같은 페이로드를 쓴다.
  ``u = 0`` 은 이미지 왼쪽 **가장자리**다(캔버스 좌표 [0, w] → 픽셀 중심 ``x = u·w − 0.5``).
* 점 순서는 왼쪽 위부터 행 우선(row-major).

계산
* ``cv2.remap`` 은 출력 픽셀마다 원본 위치를 묻는다 — 그래서 **옮긴 점(dst) → 원래 점(src)** 으로 맞춘다.
* TPS 커널 ``U(r²) = r²·log r²``, ``[[K, P], [Pᵀ, 0]] W = [V; 0]`` 를 x·y 함께 푼다. 좌표는
  ``max(w, h)`` 로 나눠 등방(축마다 따로 정규화하지 않는다)이고 행렬 조건도 좋다.
* 아핀 부분은 전체 해상도에서 정확히, 휘는 부분은 끝점을 포함한 성긴 격자(간격 8px, 한 변 ≤513점)에서
  구한 뒤 가로·세로로 한 번씩 선형 보간해 키운다(분리형 — 행마다 이웃 두 칸만 읽는다). 순수 평행 이동·
  아핀은 정확하다. 보간 오차는 끈 점 바로 옆(TPS 곡률이 가장 큰 곳)에서 1024 px 0.1px, 4K 0.03px 정도다
  (16px 간격이면 1024 에서 0.4px 까지 벌어져 8px 로 잡았다. 4K 계산은 CPU 0.5초 안팎).
* 테두리: 각 변에 앵커 21개(1/24 간격, 격자점 자리는 건너뜀 — 겹치면 행렬이 특이해진다)를 둔다.
  '가장자리 고정'이면 테두리 12점과 앵커가 제자리라 테두리가 거의 휘지 않는다. 고정을 끄면 앵커는
  옮긴 테두리 점을 잇는 꺾은선을 따라간다(움직이지 않은 변은 그대로 곧다).
* ``INTER_LINEAR`` + ``BORDER_REPLICATE`` — 안쪽으로 당긴 테두리는 검은 띠 대신 가장자리 색으로 채운다.
  BGRA 는 통째로 옮겨 알파가 내용과 함께 움직인다(원근 보정과 같다).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

GRID = 4                        # 한 변의 점 수 (3×3 칸)
EDGE_ANCHORS_PER_SIDE = 24      # 테두리 앵커 간격 1/24 — 격자점(8/24 의 배수)은 건너뛴다
COARSE_MAX = 513                # 휘는 부분을 계산하는 성긴 격자의 한 변 최대 점 수
COARSE_STEP_PX = 8              # 성긴 격자 간격 목표(px)
MIN_POINT_GAP = 1e-4            # 정규화 좌표에서 이보다 가까운 두 점(앵커 포함)은 겹친 것으로 본다
MIN_GRID_GAP = 0.01             # 격자점 16개 사이 최소 거리(정규화) — 프론트는 0.02 로 막는다
IDENTITY_ATOL = 1e-6
EVAL_CHUNK = 8192

MSG_NEED_POINTS = '격자 변형: 점 16개가 필요합니다'
MSG_OVERLAP = '격자 점이 서로 겹쳤습니다 — 조금 떼어 놓으세요'
MSG_FOLDED = '격자가 접혔습니다 — 점을 조금 되돌려 주세요'


class MeshWarpError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 격자 변형 실패."""


# ── 격자 ───────────────────────────────────────────────────────────────────

def grid_uv(n: int = GRID) -> np.ndarray:
    """``(n·n, 2)`` 정규화 격자점, 왼쪽 위부터 행 우선."""
    t = np.linspace(0.0, 1.0, n)
    return np.array([[t[c], t[r]] for r in range(n) for c in range(n)], dtype=np.float64)


def edge_indices(n: int = GRID) -> np.ndarray:
    """테두리 점의 불리언 표시 ``(n·n,)``."""
    idx = np.arange(n * n)
    r, c = idx // n, idx % n
    return (r == 0) | (r == n - 1) | (c == 0) | (c == n - 1)


def _anchor_params(n: int, per_side: int) -> list[float]:
    if per_side % (n - 1):
        raise ValueError('앵커 수는 칸 수의 배수여야 합니다')
    step = per_side // (n - 1)
    return [k / per_side for k in range(1, per_side) if k % step]


def _side_points(n: int) -> dict[str, list[int]]:
    """변마다 그 변을 이루는 격자점 인덱스(진행 방향 순)."""
    return {
        'top': [c for c in range(n)],
        'bottom': [(n - 1) * n + c for c in range(n)],
        'left': [r * n for r in range(n)],
        'right': [r * n + n - 1 for r in range(n)],
    }


def anchor_points(grid_pts: np.ndarray, n: int = GRID, per_side: int = EDGE_ANCHORS_PER_SIDE) -> np.ndarray:
    """테두리 꺾은선(``grid_pts`` 의 테두리 점을 이은 선) 위의 앵커 ``(4·len, 2)``.

    ``grid_pts`` 가 원래 격자면 곧은 테두리 위의 ``k/24`` 자리가 된다. 순서는 늘 같다
    (변 top·bottom·left·right × t) — 원래 격자와 옮긴 격자의 앵커가 짝을 이룬다.
    """
    pts = np.asarray(grid_pts, dtype=np.float64).reshape(n * n, 2)
    sides = _side_points(n)
    out = []
    for t in _anchor_params(n, per_side):
        pos = t * (n - 1)
        j = min(int(math.floor(pos)), n - 2)
        frac = pos - j
        for name in ('top', 'bottom', 'left', 'right'):
            a, b = pts[sides[name][j]], pts[sides[name][j + 1]]
            out.append(a + (b - a) * frac)
    return np.array(out, dtype=np.float64)


def edge_anchor_uv(n: int = GRID, per_side: int = EDGE_ANCHORS_PER_SIDE) -> np.ndarray:
    """원래 격자의 테두리 앵커(곧은 변 위 ``k/24``, 격자점 자리 제외)."""
    return anchor_points(grid_uv(n), n, per_side)


def uv_to_px(uv: np.ndarray, w: int, h: int) -> np.ndarray:
    """정규화 좌표 → 픽셀 중심 좌표 (``x = u·w − 0.5``)."""
    uv = np.asarray(uv, dtype=np.float64)
    return np.stack([uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5], axis=1)


# ── 얇은 판 스플라인 ────────────────────────────────────────────────────────

def _kernel(r2: np.ndarray) -> np.ndarray:
    """``U(r²) = r²·log r²``, ``U(0) = 0``. 불리언 인덱싱 없이 — 0 은 아주 작은 수로 바꿔 0·log 가 0 이 되게."""
    return r2 * np.log(np.maximum(r2, 1e-300))


def tps_fit(ctrl: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``ctrl`` 에서 ``vals`` 로 가는 TPS. ``(가중치 (n, d), 아핀 (3, d))`` — 아핀 행은 [상수, x, y]."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    n = ctrl.shape[0]
    d2 = ((ctrl[:, None, :] - ctrl[None, :, :]) ** 2).sum(-1)
    p = np.hstack([np.ones((n, 1)), ctrl])
    system = np.zeros((n + 3, n + 3))
    system[:n, :n] = _kernel(d2)
    system[:n, n:] = p
    system[n:, :n] = p.T
    rhs = np.zeros((n + 3, vals.shape[1]))
    rhs[:n] = vals
    try:
        sol = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise MeshWarpError(MSG_OVERLAP) from exc
    if not np.all(np.isfinite(sol)):
        raise MeshWarpError(MSG_OVERLAP)
    return sol[:n], sol[n:]


def tps_bent(ctrl: np.ndarray, weights: np.ndarray, pts: np.ndarray, chunk: int = EVAL_CHUNK) -> np.ndarray:
    """휘는 부분 ``Σ wᵢ U(|p − cᵢ|²)`` 만. 덩어리로 나눠 임시 메모리를 작게 둔다."""
    pts = np.asarray(pts, dtype=np.float64)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    out = np.empty((pts.shape[0], weights.shape[1]), dtype=np.float64)
    c2 = (ctrl ** 2).sum(-1)[None, :]
    for start in range(0, pts.shape[0], chunk):
        part = pts[start:start + chunk]
        d2 = (part ** 2).sum(-1)[:, None] + c2 - 2.0 * (part @ ctrl.T)
        np.maximum(d2, 0.0, out=d2)   # 전개식의 반올림으로 생기는 아주 작은 음수
        out[start:start + chunk] = _kernel(d2) @ weights
    return out


def tps_eval(ctrl: np.ndarray, weights: np.ndarray, affine: np.ndarray, pts: np.ndarray,
             chunk: int = EVAL_CHUNK) -> np.ndarray:
    """TPS 값 = 아핀 + 휘는 부분."""
    pts = np.asarray(pts, dtype=np.float64)
    return affine[0] + pts @ affine[1:] + tps_bent(ctrl, weights, pts, chunk)


def _coarse_count(n: int) -> int:
    if n <= 1:
        return 1
    return int(min(COARSE_MAX, max(2, math.ceil((n - 1) / COARSE_STEP_PX) + 1)))


def _lerp_index(n_out: int, n_coarse: int) -> tuple[np.ndarray, np.ndarray]:
    """끝점을 포함한 성긴 격자(``n_coarse`` 점이 0 ~ n_out−1 을 덮음)에서 ``n_out`` 점을 선형 보간할
    왼쪽 이웃 인덱스와 비율."""
    if n_coarse <= 1 or n_out <= 1:
        return np.zeros(n_out, dtype=np.int64), np.zeros(n_out, dtype=np.float64)
    t = np.arange(n_out, dtype=np.float64) * ((n_coarse - 1) / (n_out - 1))
    i0 = np.clip(np.floor(t).astype(np.int64), 0, n_coarse - 2)
    return i0, t - i0


def _upsample(coarse: np.ndarray, w: int, h: int) -> np.ndarray:
    """``(ny, nx)`` 성긴 값을 ``(h, w)`` 로 — 가로 한 번, 세로 한 번 선형 보간."""
    ny, nx = coarse.shape
    ix, fx = _lerp_index(w, nx)
    iy, fy = _lerp_index(h, ny)
    if nx > 1:
        rows = coarse[:, ix] * (1.0 - fx) + coarse[:, ix + 1] * fx
    else:
        rows = np.repeat(coarse, w, axis=1)
    if ny > 1:
        out = rows[iy] * (1.0 - fy)[:, None]
        out += rows[iy + 1] * fy[:, None]
        return out
    return np.repeat(rows, h, axis=0)


def warp_maps(w: int, h: int, src_px: np.ndarray, dst_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.remap`` 용 ``(map_x, map_y)`` float32 — 출력 픽셀(옮긴 쪽)마다 원본 픽셀 위치."""
    scale = float(max(w, h))
    ctrl = np.asarray(dst_px, dtype=np.float64) / scale
    vals = np.asarray(src_px, dtype=np.float64) / scale
    weights, affine = tps_fit(ctrl, vals)

    nx, ny = _coarse_count(w), _coarse_count(h)
    gx = np.linspace(0.0, w - 1, nx) / scale
    gy = np.linspace(0.0, h - 1, ny) / scale
    pts = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    bent = tps_bent(ctrl, weights, pts).reshape(ny, nx, 2)
    xs = np.arange(w, dtype=np.float64) / scale
    ys = np.arange(h, dtype=np.float64) / scale

    maps = []
    for k in range(2):
        m = _upsample(bent[:, :, k], w, h)
        m += (affine[1, k] * xs)[None, :]
        m += (affine[0, k] + affine[2, k] * ys)[:, None]
        m *= scale
        maps.append(m.astype(np.float32))
        del m
    return maps[0], maps[1]


# ── 실행 ───────────────────────────────────────────────────────────────────

def control_points(dst_uv: np.ndarray, pin_edges: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``(원래 점들, 옮긴 점들)`` — 격자 16점 + 테두리 앵커. '가장자리 고정'이면 테두리 12점을 되돌린다."""
    src = grid_uv(GRID)
    dst = np.clip(np.asarray(dst_uv, dtype=np.float64).reshape(GRID * GRID, 2), 0.0, 1.0)
    if pin_edges:
        edge = edge_indices(GRID)
        dst[edge] = src[edge]
    return (np.vstack([src, anchor_points(src)]),
            np.vstack([dst, anchor_points(dst)]))


def _check_gaps(pts: np.ndarray, min_gap: float = MIN_POINT_GAP) -> None:
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    if float(d2.min()) < min_gap ** 2:
        raise MeshWarpError(MSG_OVERLAP)


def grid_cells(n: int = GRID) -> list[tuple[int, int, int, int]]:
    """칸마다 꼭짓점 인덱스(왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래)."""
    return [(r * n + c, r * n + c + 1, (r + 1) * n + c + 1, (r + 1) * n + c)
            for r in range(n - 1) for c in range(n - 1)]


def cell_is_convex(quad) -> bool:
    """칸이 볼록한가 — 네 꼭짓점 모두 처음 격자와 같은 방향으로 돈다(뒤집힘·꼬임·오목을 모두 거른다).

    프론트(utils/meshWarp.meshCellConvex)는 꼭짓점 각 sin > 0.05 로 조금 더 엄격하다(반올림 여유)."""
    q = [tuple(map(float, p)) for p in quad]
    for k in range(4):
        p, c, n = q[k - 1], q[k], q[(k + 1) % 4]
        if (n[0] - c[0]) * (p[1] - c[1]) - (n[1] - c[1]) * (p[0] - c[0]) <= 0:
            return False
    return True


def check_grid(grid_pts: np.ndarray) -> None:
    """옮긴 격자 16점 검사 — 너무 가까운 점(MSG_OVERLAP)·볼록하지 않은 칸(MSG_FOLDED)이면 MeshWarpError.

    접히거나 오목한 칸은 TPS 가 출력 일부를 뒤집어(이미지 밖 BORDER_REPLICATE 번짐까지) 프리뷰·적용이 망가진다.
    볼록한 칸이어도 아주 크게 끌면 TPS 가 조금 접힐 수 있다(실측: 점마다 ±0.3 무작위 끌기에서 13%, 볼록 검사
    전 36%) — 그건 프리뷰로 보인다."""
    pts = np.asarray(grid_pts, dtype=np.float64).reshape(GRID * GRID, 2)
    _check_gaps(pts, MIN_GRID_GAP)
    for cell in grid_cells(GRID):
        if not cell_is_convex(pts[list(cell)]):
            raise MeshWarpError(MSG_FOLDED)


def is_identity(dst_uv, pin_edges: bool = True) -> bool:
    _src, dst = control_points(dst_uv, pin_edges)
    return bool(np.allclose(dst[:GRID * GRID], grid_uv(GRID), atol=IDENTITY_ATOL, rtol=0.0))


def mesh_warp(img: np.ndarray, dst_uv, *, pin_edges: bool = True) -> np.ndarray:
    """격자점 16개를 ``dst_uv`` 로 옮긴 모양으로 이미지를 휜다. 같은 크기·dtype·채널.

    아무 점도 움직이지 않았으면 정확히 같은 사본을 돌려준다.
    """
    import cv2
    src, dst = control_points(dst_uv, pin_edges)
    if np.allclose(dst[:GRID * GRID], src[:GRID * GRID], atol=IDENTITY_ATOL, rtol=0.0):
        return img.copy()
    check_grid(dst[:GRID * GRID])
    _check_gaps(dst)
    h, w = img.shape[:2]
    map_x, map_y = warp_maps(w, h, uv_to_px(src, w, h), uv_to_px(dst, w, h))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_mesh_params(params: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    """``(dst_uv (16, 2), pin_edges)``. 모양이 틀리면 MeshWarpError(사용자용 문구)."""
    grid = params.get('grid', GRID)
    if not _finite_number(grid) or int(grid) != GRID:
        raise MeshWarpError(MSG_NEED_POINTS)
    raw = params.get('dst')
    if not isinstance(raw, (list, tuple)) or len(raw) != GRID * GRID:
        raise MeshWarpError(MSG_NEED_POINTS)
    pts = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not all(_finite_number(v) for v in item):
            raise MeshWarpError(MSG_NEED_POINTS)
        pts.append([float(item[0]), float(item[1])])
    pin = params.get('pin_edges', True)
    pin_edges = pin if isinstance(pin, bool) else str(pin).strip().lower() not in ('0', 'false', 'no', 'off', '')
    return np.clip(np.array(pts, dtype=np.float64), 0.0, 1.0), pin_edges


def run_mesh_warp(img: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    dst, pin_edges = parse_mesh_params(params)
    return mesh_warp(img, dst, pin_edges=pin_edges)
