"""X/Y/Z grid drawing for the Anima studio: cell boxes, legend text, margins, Z sub-grids and Row Count (PIL only).

Follows the layout of Forge Neo's X/Y/Z plot as the recon memo describes it; the pack carries no Forge code.

- A sub-grid holds the ``nx * ny`` cells of one Z value.  Every cell sits in a box of the largest cell size (a smaller
  image is centred), on a white background; a cell that is ``None`` (or missing) is a black picture of the first
  cell's size.
- Legend: X labels above the columns and Y labels left of the rows, black text on white.  Font size is
  ``(cell_w + cell_h) // 25``, line spacing half of it.  The left padding is ``cell_w * 3 // 4`` (0 when every Y label
  is empty), the top padding the tallest X header block plus two line spacings (0 when every X label is empty).
  Words wrap greedily to the column width (X) or the left padding (Y); a word that is wider than that gets its own
  line and its font shrinks 1 px at a time until it fits.
- ``margin`` pixels separate the X/Y cells only (never the outer edge); without the legend neither labels, padding
  nor margin exist.
- Z: the sub-grids sit side by side in one row, each with its Z label as a column title (margin 0, font size from the
  sub-grid size); a single empty Z title adds no padding, so with no Z axis the main grid equals the sub-grid.
- Row Count: when only X varies, the finished main grid is cut into strips of ``ceil(nx / rows)`` cells (the header
  strip travels along) which are stacked on a white canvas.  ``row_count`` 0 means ``int(sqrt(nx))``.
- Label font: Roboto (bundled in ``fonts/``) when it has every glyph of the line, otherwise a Korean-capable
  Windows font (Malgun Gothic, then Noto Sans KR), otherwise Pillow's default font.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#ffffff"
TEXT_COLOR = "#000000"
ROBOTO_PATH = Path(__file__).resolve().parent / "fonts" / "Roboto-Regular.ttf"
FALLBACK_FONT_NAMES = ("malgun.ttf", "NotoSansKR-VF.ttf")
_NOTDEF_PROBE = "￿"  # a noncharacter no font maps: its mask is the font's "missing glyph" box


def _font_dirs() -> list[Path]:
    dirs = []
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir:
        dirs.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    return dirs


@lru_cache(maxsize=1)
def _fallback_path() -> str | None:
    for name in FALLBACK_FONT_NAMES:
        for directory in _font_dirs():
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


@lru_cache(maxsize=64)
def _load_font(path: str | None, size: int):
    size = max(1, int(size))
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow before 10.1 has no sized default font
        return ImageFont.load_default()


@lru_cache(maxsize=4096)
def _roboto_has(char: str) -> bool:
    if char.isspace():
        return True
    font = _load_font(str(ROBOTO_PATH), 24)
    mask, probe = font.getmask(char), font.getmask(_NOTDEF_PROBE)
    return not (mask.size == probe.size and bytes(mask) == bytes(probe))


def font_path_for(text: str) -> str | None:
    """Path of the font that draws ``text``; ``None`` means Pillow's default font."""
    if ROBOTO_PATH.is_file() and all(_roboto_has(char) for char in text):
        return str(ROBOTO_PATH)
    return _fallback_path()


def _height(font) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def layout_label(text: str, font_size: int, allowed_width: int) -> list[tuple[str, object, int]]:
    """Wrap ``text`` into ``(line, font, line_height)`` triples that fit ``allowed_width`` where a line can."""

    def font_for(line: str, size: int):
        return _load_font(font_path_for(line), size)

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if not current or font_for(candidate, font_size).getlength(candidate) <= allowed_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    result = []
    for line in lines:
        size = font_size
        font = font_for(line, size)
        while size > 1 and font.getlength(line) > allowed_width:
            size -= 1
            font = font_for(line, size)
        result.append((line, font, _height(font)))
    return result


def _block_height(lines, spacing: int) -> int:
    return sum(h for _t, _f, h in lines) + max(0, len(lines) - 1) * spacing


def _draw_block(draw, lines, center_x: float, top: float, spacing: int) -> None:
    y = top
    for text, font, height in lines:
        draw.text((center_x, y + height / 2), text, fill=TEXT_COLOR, font=font, anchor="mm")
        y += height + spacing


def _compose(images, cols, rows, box_w, box_h, hor_texts, ver_texts, margin) -> Image.Image:
    """Lay ``images`` (x fastest) out in ``cols x rows`` boxes of ``box_w x box_h`` with the legend around them."""
    fs = (box_w + box_h) // 25
    spacing = fs // 2
    left = box_w * 3 // 4 if any(ver_texts) else 0
    hor = [layout_label(t, fs, box_w) for t in hor_texts]
    ver = [layout_label(t, fs, left) for t in ver_texts]
    top = max((_block_height(lines, spacing) for lines in hor), default=0)
    top = top + 2 * spacing if any(hor_texts) else 0
    canvas = Image.new("RGB", (left + cols * box_w + margin * (cols - 1), top + rows * box_h + margin * (rows - 1)),
                       BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        col, row = index % cols, index // cols
        x0 = left + col * (box_w + margin)
        y0 = top + row * (box_h + margin)
        canvas.paste(image, (x0 + (box_w - image.width) // 2, y0 + (box_h - image.height) // 2))
    for col, lines in enumerate(hor):
        x_center = left + col * (box_w + margin) + box_w / 2
        _draw_block(draw, lines, x_center, (top - _block_height(lines, spacing)) / 2, spacing)
    for row, lines in enumerate(ver):
        y_center = top + row * (box_h + margin) + box_h / 2
        # The block is centred on its height without the line spacing, so a many-line label sits slightly low.
        _draw_block(draw, lines, left / 2, y_center - sum(h for _t, _f, h in lines) / 2, spacing)
    return canvas


def _texts(labels, letter: str, count: int) -> list[str]:
    values = list(labels.get(letter) or [])
    return (values + [""] * count)[:count]


def _stack_rows(main: Image.Image, nx: int, header: int, cell_w: int, margin: int, rows: int) -> Image.Image:
    """Cut a one-row grid into horizontal strips of ``ceil(nx / rows)`` cells and stack them (header on each)."""
    per_row = math.ceil(nx / rows)
    pitch = cell_w + margin
    strips = math.ceil(nx / per_row)
    strip_w = per_row * pitch - margin
    out = Image.new("RGB", (strip_w, strips * main.height), BACKGROUND)
    for index in range(strips):
        left = index * per_row * pitch
        out.paste(main.crop((left, 0, min(main.width, left + strip_w), main.height)), (0, index * main.height))
    return out


def draw_xyz(cells, nx: int, ny: int, nz: int, labels, *, draw_legend: bool, margin: int, row_count: int):
    """Draw the main grid and the per-Z sub-grids; returns ``(main, [sub_grid for each z])``.

    ``cells`` maps ``(ix, iy, iz)`` to a PIL image or ``None``; ``labels`` maps ``"x"``/``"y"``/``"z"`` to the label
    strings of that axis.  ``margin`` applies between X/Y cells only and is ignored without the legend.
    """
    first = next((im for im in cells.values() if im is not None), None)
    if first is None:
        raise ValueError("no cell image to draw")
    black = Image.new("RGB", first.size, (0, 0, 0))

    def cell(ix: int, iy: int, iz: int) -> Image.Image:
        image = cells.get((ix, iy, iz))
        return black if image is None else image.convert("RGB")

    pictures = {(ix, iy, iz): cell(ix, iy, iz) for iz in range(nz) for iy in range(ny) for ix in range(nx)}
    box_w = max(im.width for im in pictures.values())
    box_h = max(im.height for im in pictures.values())
    legend = bool(draw_legend)
    margin = max(0, int(margin)) if legend else 0
    x_texts = _texts(labels, "x", nx) if legend else [""] * nx
    y_texts = _texts(labels, "y", ny) if legend else [""] * ny
    z_texts = _texts(labels, "z", nz) if legend else [""] * nz

    subs = [_compose([pictures[(ix, iy, iz)] for iy in range(ny) for ix in range(nx)], nx, ny, box_w, box_h,
                     x_texts, y_texts, margin)
            for iz in range(nz)]
    if nz == 1:
        main = subs[0].copy()
    else:
        sub_w = max(s.width for s in subs)
        sub_h = max(s.height for s in subs)
        main = _compose(subs, nz, 1, sub_w, sub_h, z_texts, [], 0)

    x_only = ny == 1 and nz == 1 and not any(_texts(labels, "y", 1)) and not any(_texts(labels, "z", 1))
    rows = int(row_count) if int(row_count) > 0 else int(math.sqrt(nx))
    rows = min(rows, nx)
    if x_only and rows > 1:
        main = _stack_rows(main, nx, main.height - box_h, box_w, margin, rows)
    return main, subs
