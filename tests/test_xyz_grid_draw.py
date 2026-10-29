"""X/Y/Z grid drawing: cell boxes, legend padding, margins, Z sub-grids, Row Count and the label font (PIL only)."""
import math
import unittest

from PIL import Image

from comfy_custom_nodes.ai_studio_forge_parity import xyz_grid_draw as gd

RED = (200, 0, 0)
WHITE = (255, 255, 255)


def _cells(nx, ny, nz=1, size=(64, 64), color=RED):
    return {(ix, iy, iz): Image.new("RGB", size, color)
            for iz in range(nz) for iy in range(ny) for ix in range(nx)}


def _labels(nx, ny, nz=1, x=True, y=True, z=False):
    return {"x": [f"CFG Scale: {i}" for i in range(nx)] if x else [""] * nx,
            "y": [f"Steps: {i}" for i in range(ny)] if y else [""] * ny,
            "z": [f"Seed: {i}" for i in range(nz)] if z else [""] * nz}


def _draw(cells, nx, ny, nz=1, labels=None, **options):
    opts = {"draw_legend": True, "margin": 0, "row_count": 0, **options}
    return gd.draw_xyz(cells, nx, ny, nz, labels or _labels(nx, ny, nz), **opts)


class SizeTests(unittest.TestCase):
    def test_size_with_legend(self):
        main, subs = _draw(_cells(4, 3), 4, 3)
        self.assertEqual(main.width, 64 * 4 + 64 * 3 // 4)
        self.assertGreater(main.height, 64 * 3)
        self.assertEqual(main.mode, "RGB")
        self.assertEqual(len(subs), 1)

    def test_no_legend_no_padding(self):
        main, subs = _draw(_cells(4, 3), 4, 3, draw_legend=False)
        self.assertEqual(main.size, (64 * 4, 64 * 3))
        self.assertEqual(subs[0].size, main.size)

    def test_empty_labels_need_no_padding(self):
        main, _ = _draw(_cells(4, 3), 4, 3, labels=_labels(4, 3, x=False, y=False))
        self.assertEqual(main.size, (64 * 4, 64 * 3))

    def test_only_x_labels_pad_the_top_only(self):
        main, _ = _draw(_cells(4, 3), 4, 3, labels=_labels(4, 3, y=False))
        self.assertEqual(main.width, 64 * 4)
        self.assertGreater(main.height, 64 * 3)

    def test_only_y_labels_pad_the_left_only(self):
        main, _ = _draw(_cells(4, 3), 4, 3, labels=_labels(4, 3, x=False))
        self.assertEqual(main.size, (64 * 4 + 64 * 3 // 4, 64 * 3))

    def test_top_padding_is_header_block_plus_two_line_spacings(self):
        fs = (64 + 64) // 25
        lines = gd.layout_label("CFG Scale: 0", fs, 64)
        block = sum(h for _t, _f, h in lines) + (len(lines) - 1) * (fs // 2)
        main, _ = _draw(_cells(1, 1), 1, 1, labels={"x": ["CFG Scale: 0"], "y": [""], "z": [""]})
        self.assertEqual(main.height, 64 + block + 2 * (fs // 2))


class MarginTests(unittest.TestCase):
    def test_margin_between_cells_only(self):
        labels = _labels(4, 3)
        flat, _ = _draw(_cells(4, 3), 4, 3, labels=labels)
        gapped, _ = _draw(_cells(4, 3), 4, 3, labels=labels, margin=10)
        self.assertEqual(gapped.width, flat.width + 30)
        self.assertEqual(gapped.height, flat.height + 20)

    def test_nothing_outside_the_outer_cells(self):
        main, _ = _draw(_cells(4, 3), 4, 3, labels=_labels(4, 3, x=False, y=False), margin=10)
        self.assertEqual(main.size, (64 * 4 + 30, 64 * 3 + 20))
        self.assertEqual(main.getpixel((0, 0)), RED)
        self.assertEqual(main.getpixel((main.width - 1, main.height - 1)), RED)
        self.assertEqual(main.getpixel((64 + 5, 5)), WHITE)  # the gap between column 0 and 1

    def test_margin_is_ignored_without_legend(self):
        main, _ = _draw(_cells(4, 3), 4, 3, draw_legend=False, margin=10)
        self.assertEqual(main.size, (64 * 4, 64 * 3))


class CellTests(unittest.TestCase):
    def test_failed_cell_black(self):
        cells = _cells(2, 2)
        cells[(1, 1, 0)] = None
        main, _ = _draw(cells, 2, 2, draw_legend=False)
        self.assertEqual(main.getpixel((64 + 32, 64 + 32)), (0, 0, 0))
        self.assertEqual(main.getpixel((32, 32)), RED)

    def test_missing_cell_is_black_with_the_first_cell_size(self):
        cells = _cells(2, 1, size=(40, 30))
        del cells[(1, 0, 0)]
        main, _ = _draw(cells, 2, 1, draw_legend=False)
        self.assertEqual(main.size, (80, 30))
        self.assertEqual(main.getpixel((60, 15)), (0, 0, 0))

    def test_smaller_image_is_centered_in_the_max_box(self):
        cells = _cells(2, 1)
        cells[(1, 0, 0)] = Image.new("RGB", (32, 32), RED)
        main, _ = _draw(cells, 2, 1, draw_legend=False)
        self.assertEqual(main.size, (128, 64))
        self.assertEqual(main.getpixel((64 + 32, 32)), RED)       # centre of the small image
        self.assertEqual(main.getpixel((64 + 2, 2)), WHITE)       # box corner stays background
        self.assertEqual(main.getpixel((64 + 16, 16)), RED)       # first pixel row/col of the small image
        self.assertEqual(main.getpixel((64 + 15, 16)), WHITE)

    def test_cells_land_in_x_fastest_row_major_order(self):
        cells = {(0, 0, 0): Image.new("RGB", (8, 8), (10, 0, 0)), (1, 0, 0): Image.new("RGB", (8, 8), (20, 0, 0)),
                 (0, 1, 0): Image.new("RGB", (8, 8), (30, 0, 0)), (1, 1, 0): Image.new("RGB", (8, 8), (40, 0, 0))}
        main, _ = _draw(cells, 2, 2, draw_legend=False)
        self.assertEqual([main.getpixel(p)[0] for p in ((4, 4), (12, 4), (4, 12), (12, 12))], [10, 20, 30, 40])

    def test_no_images_at_all_is_an_error(self):
        with self.assertRaises(ValueError):
            _draw({(0, 0, 0): None}, 1, 1)


class ZSubGridTests(unittest.TestCase):
    def test_z_subgrids_side_by_side(self):
        main, subs = _draw(_cells(2, 2, 2, size=(32, 32)), 2, 2, 2, draw_legend=False)
        self.assertEqual(len(subs), 2)
        self.assertEqual([s.size for s in subs], [(64, 64), (64, 64)])
        self.assertEqual(main.size, (128, 64))

    def test_z_subgrids_with_legend_are_at_least_as_wide_and_get_titles(self):
        labels = _labels(2, 2, 2, z=True)
        main, subs = _draw(_cells(2, 2, 2, size=(32, 32)), 2, 2, 2, labels=labels)
        self.assertGreaterEqual(main.width, sum(s.width for s in subs))
        self.assertEqual(main.width, sum(s.width for s in subs))
        self.assertGreater(main.height, subs[0].height)

    def test_empty_z_titles_add_no_padding(self):
        labels = _labels(2, 2, 2, z=False)
        main, subs = _draw(_cells(2, 2, 2, size=(32, 32)), 2, 2, 2, labels=labels)
        self.assertEqual(main.height, subs[0].height)

    def test_single_z_main_equals_the_sub_grid(self):
        main, subs = _draw(_cells(3, 2), 3, 2)
        self.assertEqual(len(subs), 1)
        self.assertEqual(main.size, subs[0].size)
        self.assertEqual(main.tobytes(), subs[0].tobytes())

    def test_subgrids_ignore_margin_between_themselves(self):
        flat, _ = _draw(_cells(2, 1, 2, size=(32, 32)), 2, 1, 2, labels=_labels(2, 1, 2, y=False))
        gapped, subs = _draw(_cells(2, 1, 2, size=(32, 32)), 2, 1, 2, labels=_labels(2, 1, 2, y=False), margin=6)
        self.assertEqual(gapped.width, 2 * subs[0].width)  # only the X/Y gaps inside each sub-grid
        self.assertEqual(subs[0].width, flat.width // 2 + 6)


class RowCountTests(unittest.TestCase):
    def test_row_count_x_only(self):
        main, subs = _draw(_cells(9, 1), 9, 1, labels=_labels(9, 1, y=False), draw_legend=False)
        self.assertEqual(main.size, (3 * 64, 3 * 64))
        self.assertEqual(subs[0].size, (9 * 64, 64))  # the sub-grid keeps its single row

    def test_row_count_auto_is_int_sqrt(self):
        for nx, rows in ((8, 2), (3, 1), (4, 2), (10, 3)):
            with self.subTest(nx=nx):
                main, _ = _draw(_cells(nx, 1, size=(16, 16)), nx, 1, labels=_labels(nx, 1, y=False), draw_legend=False)
                per = math.ceil(nx / rows)
                self.assertEqual(main.size, (per * 16, rows * 16))

    def test_explicit_row_count_and_short_last_strip(self):
        main, _ = _draw(_cells(5, 1, size=(16, 16)), 5, 1, labels=_labels(5, 1, y=False), draw_legend=False, row_count=2)
        self.assertEqual(main.size, (3 * 16, 2 * 16))
        self.assertEqual(main.getpixel((2 * 16 + 8, 16 + 8)), WHITE)  # the unused slot of the last strip
        self.assertEqual(main.getpixel((8, 16 + 8)), RED)

    def test_header_strip_travels_with_every_row(self):
        labels = _labels(9, 1, y=False)
        one, _ = _draw(_cells(9, 1), 9, 1, labels=labels, row_count=1)
        three, _ = _draw(_cells(9, 1), 9, 1, labels=labels)
        header = one.height - 64
        self.assertGreater(header, 0)
        self.assertEqual(three.size, (3 * 64, 3 * (64 + header)))

    def test_row_count_needs_a_single_y_and_z(self):
        main, _ = _draw(_cells(9, 2), 9, 2, draw_legend=False, row_count=3)
        self.assertEqual(main.size, (9 * 64, 2 * 64))

    def test_row_count_with_margin_has_no_stray_gap(self):
        main, _ = _draw(_cells(4, 1, size=(16, 16)), 4, 1, labels=_labels(4, 1, x=False, y=False), margin=4,
                        row_count=2)
        self.assertEqual(main.size, (2 * 16 + 4, 2 * 16))
        self.assertEqual(main.getpixel((main.width - 1, 5)), RED)


class FontTests(unittest.TestCase):
    def test_english_label_uses_roboto(self):
        path = gd.font_path_for("CFG Scale: 4.5")
        self.assertTrue(path.endswith("Roboto-Regular.ttf"), path)

    def test_korean_label_uses_korean_font(self):
        path = (gd.font_path_for("세기: 0.9") or "").casefold()
        self.assertTrue("malgun" in path or "notosanskr" in path, path)

    def test_the_bundled_roboto_exists_next_to_the_module(self):
        self.assertTrue(gd.ROBOTO_PATH.is_file())
        self.assertEqual(gd.ROBOTO_PATH.parent.name, "fonts")

    def test_empty_text_uses_roboto(self):
        self.assertTrue(gd.font_path_for("").endswith("Roboto-Regular.ttf"))

    def test_korean_legend_draws_real_glyphs_not_tofu(self):
        labels = {"x": ["세기"], "y": [""], "z": [""]}
        main, _ = _draw(_cells(1, 1, size=(256, 256)), 1, 1, labels=labels)
        header = main.crop((0, 0, 256, main.height - 256))
        self.assertNotEqual(header.tobytes(), Image.new("RGB", header.size, WHITE).tobytes())


class WrapTests(unittest.TestCase):
    def test_short_label_is_one_line(self):
        self.assertEqual(len(gd.layout_label("Steps: 20", 20, 400)), 1)

    def test_words_wrap_greedily(self):
        lines = gd.layout_label("alpha beta gamma delta epsilon zeta", 20, 120)
        self.assertGreater(len(lines), 1)
        self.assertEqual(" ".join(text for text, _f, _h in lines), "alpha beta gamma delta epsilon zeta")
        for text, font, _h in lines:
            if " " in text:
                self.assertLessEqual(font.getlength(text), 120)

    def test_overlong_word_gets_its_own_line_and_shrinks_to_fit(self):
        lines = gd.layout_label("hi extraordinarilylongunbreakableword", 30, 100)
        self.assertEqual([t for t, _f, _h in lines], ["hi", "extraordinarilylongunbreakableword"])
        text, font, _h = lines[1]
        self.assertLessEqual(font.getlength(text), 100)
        self.assertLess(font.size, 30)

    def test_empty_label_has_no_lines(self):
        self.assertEqual(gd.layout_label("  ", 20, 100), [])


if __name__ == "__main__":
    unittest.main()
