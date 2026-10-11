"""연관 태그 표 만들기 — 어휘 고르기 · 작은 parquet 에서 쌍 세기 · 파일 왕복(core/related_tags_build.py)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import related_tags_build as rb
from tests import _related_tags_fixture as fx

GENERAL_ORDER = [
    "1girl", "solo", "smile", "armor", "sword", "hat", "beach", "bikini", "cape", "holding_sword", "ocean",
    "shoulder_armor", "swimsuit", "weapon", "long_hair", "bikini_armor", "aqua_hair", "necktie", "twintails",
    "nurse_cap",
]


class VocabTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = fx.write_dataset(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def vocab(self, **kw):
        kw.setdefault("aliases", fx.ALIASES)
        kw.setdefault("implications", fx.IMPLICATIONS)
        return rb.select_vocab(rb.read_catalog_rows(self.data["catalog"]), **kw)

    def test_order_categories_and_candidates(self):
        vocab = self.vocab(n_candidates=5)
        self.assertEqual(list(vocab.names), GENERAL_ORDER + ["hatsune_miku", "vocaloid"])
        self.assertEqual(vocab.n_candidates, 5)
        self.assertEqual(vocab.kind[0], 0)
        self.assertEqual(vocab.kind[-2:], (4, 3))
        self.assertNotIn("some_artist", vocab.names)
        self.assertNotIn("highres", vocab.names)

    def test_group_limits_and_duplicate_names(self):
        rows = [("a", 0, 9), ("b", 0, 8), ("c", 0, 7), ("a", 4, 99), ("x", 4, 5), ("y", 3, 1), ("Z Z", 0, 1)]
        vocab = rb.select_vocab(rows, n_candidates=10, n_general=2, n_character=5, n_copyright=0)
        self.assertEqual(vocab.names, ("a", "b", "x"))
        self.assertEqual(vocab.n_candidates, 2)    # 일반 태그 수보다 많이 고를 수 없다
        self.assertEqual(rb.select_vocab([("Z Z", 0, 1)]).names, ("z_z",))

    def test_aliases_and_transitive_implications_inside_the_vocab(self):
        vocab = self.vocab()
        index = vocab.index()
        self.assertEqual(vocab.aliases(), {"longhair": index["long_hair"]})   # boobs → breasts 는 어휘 밖
        implications = vocab.implications()
        # holding_weapon 은 어휘 밖이지만 그 너머의 weapon 까지 따라간다
        self.assertEqual(sorted(vocab.names[i] for i in implications[index["holding_sword"]]), ["sword", "weapon"])
        self.assertEqual([vocab.names[i] for i in implications[index["nurse_cap"]]], ["hat"])
        self.assertEqual(vocab.fingerprint, self.vocab().fingerprint)
        self.assertNotEqual(vocab.fingerprint, self.vocab(n_candidates=3).fingerprint)

    def test_catalog_reader_skips_malformed_rows(self):
        path = self.root / "bad.csv"
        path.write_text('ok,0,5,""\nshort,0\nnan,x,3,""\n\nlast,3,1\n', encoding="utf-8")
        self.assertEqual(list(rb.read_catalog_rows(path)), [("ok", 0, 5), ("last", 3, 1)])

    def test_vocab_file_round_trip_and_rejects_tampering(self):
        vocab = self.vocab()
        path = self.root / "cache" / "vocab.npz"
        rb.save_vocab(path, vocab)
        loaded = rb.load_vocab(path)
        self.assertEqual(loaded.names, vocab.names)
        self.assertEqual(loaded.implications(), vocab.implications())
        self.assertEqual(loaded.aliases(), vocab.aliases())
        self.assertEqual(loaded.fingerprint, vocab.fingerprint)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        self.assertIsNone(rb.load_vocab(self.root / "nope.npz"))
        (self.root / "junk.npz").write_bytes(b"not a zip")
        self.assertIsNone(rb.load_vocab(self.root / "junk.npz"))
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays["names"] = np.array(list(vocab.names[:-1]) + ["other"])
        np.savez(self.root / "tampered.npz", **arrays)
        self.assertIsNone(rb.load_vocab(self.root / "tampered.npz"))


class BuildShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data = fx.write_dataset(cls.root)
        cls.vocab = rb.select_vocab(rb.read_catalog_rows(cls.data["catalog"]), aliases=fx.ALIASES,
                                    implications=fx.IMPLICATIONS)
        cls.index = cls.vocab.index()
        cls.g_path = cls.data["data_dir"] / "danbooru_2026_07_g.parquet"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def neighbors(self, table, name):
        tag_id = self.index[name]
        begin, end = int(table.indptr[tag_id]), int(table.indptr[tag_id + 1])
        return {self.vocab.names[g]: int(n) for g, n in zip(table.nbr[begin:end].tolist(),
                                                              table.cnt[begin:end].tolist())}

    def test_counts_occurrences_and_positive_association_only(self):
        seen = []
        table = rb.build_shard_table(self.g_path, self.vocab, batch_rows=50, progress=seen.append)
        self.assertEqual((table.posts, table.skipped), (670, 0))
        occ = {name: int(table.occ[i]) for name, i in self.index.items()}
        self.assertEqual(occ["1girl"], 670)
        self.assertEqual(occ["armor"], 100)
        self.assertEqual(occ["hatsune_miku"], 50)
        self.assertEqual(occ["cape"], 0)
        self.assertEqual(self.neighbors(table, "armor"),
                         {"sword": 100, "holding_sword": 100, "weapon": 100, "shoulder_armor": 100})
        # 모든 게시물에 있는 태그는 무엇을 더해 주지도, 더해지지도 않는다
        self.assertEqual(self.neighbors(table, "solo"), {})
        self.assertNotIn("solo", self.neighbors(table, "sword"))
        # 캐릭터·작품(후보 아님)도 기준으로서 후보 이웃을 갖는다
        self.assertEqual(self.neighbors(table, "hatsune_miku"),
                         {"twintails": 50, "aqua_hair": 50, "necktie": 50})
        self.assertEqual(self.neighbors(table, "hat"), {"smile": 120, "long_hair": 80, "nurse_cap": 40})
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 1.0)

    def test_candidates_limit_neighbors_and_min_pair_and_top_k(self):
        vocab = rb.select_vocab(rb.read_catalog_rows(self.data["catalog"]), n_candidates=5)
        table = rb.build_shard_table(self.g_path, vocab, min_pair=60, top_k=1)
        self.assertTrue(all(g < 5 for g in table.nbr.tolist()))
        hat = vocab.index()["hat"]          # 후보 아님 — 후보 이웃 smile(120) 하나만
        begin, end = int(table.indptr[hat]), int(table.indptr[hat + 1])
        self.assertEqual([vocab.names[g] for g in table.nbr[begin:end].tolist()], ["smile"])
        armor = vocab.index()["armor"]
        begin, end = int(table.indptr[armor]), int(table.indptr[armor + 1])
        self.assertEqual([vocab.names[g] for g in table.nbr[begin:end].tolist()], ["sword"])
        nurse = vocab.index()["nurse_cap"]   # smile 과 40번 — min_pair 60 미만
        self.assertEqual(int(table.indptr[nurse + 1] - table.indptr[nurse]), 0)

    def test_posts_with_too_many_tags_are_skipped_entirely(self):
        table = rb.build_shard_table(self.g_path, self.vocab, max_tags=5)
        # 어휘 안 태그가 7개인 armor·beach·miku 묶음(100+100+50)이 통째로 빠진다
        self.assertEqual(table.skipped, 250)
        self.assertEqual(table.posts, 420)
        self.assertEqual(int(table.occ[self.index["armor"]]), 0)
        self.assertEqual(int(table.occ[self.index["1girl"]]), 420)

    def test_a_tag_repeated_inside_one_post_counts_once(self):
        path = self.root / "dupes.parquet"
        fx.write_shard(path, [(30, "armor sword armor 1girl sword armor", "", ""), (10, "armor", "", "armor"),
                             (60, "1girl", "", "")])
        table = rb.build_shard_table(path, self.vocab, min_pair=1)
        self.assertEqual(int(table.occ[self.index["armor"]]), 40)
        self.assertEqual(int(table.occ[self.index["sword"]]), 30)
        self.assertEqual(self.neighbors(table, "armor"), {"sword": 30})
        # 1girl 은 sword 와는 늘 같이(1.0 > 0.9), armor 와는 덜(0.75 < 0.9) — 더한 값이 양수인 쪽만
        self.assertEqual(self.neighbors(table, "sword"), {"armor": 30, "1girl": 30})

    def test_cancel_between_batches_and_missing_columns(self):
        calls = []

        def stop():
            calls.append(1)
            return len(calls) > 2

        with self.assertRaises(rb.BuildCanceled):
            rb.build_shard_table(self.g_path, self.vocab, batch_rows=100, should_stop=stop)
        import pyarrow as pa
        import pyarrow.parquet as pq

        broken = self.root / "broken.parquet"
        pq.write_table(pa.table({"general": ["a"], "character": [""]}), broken)
        with self.assertRaisesRegex(ValueError, "copyright"):
            rb.build_shard_table(broken, self.vocab)
        with self.assertRaises(ValueError):
            rb.build_shard_table(self.g_path, rb.Vocab(names=(), kind=(), n_candidates=0))

    def test_shard_file_round_trip(self):
        table = rb.build_shard_table(self.g_path, self.vocab)
        table.meta.update({"format": 1, "shard": "g"})
        path = self.root / "out" / "shard_g.npz"
        rb.save_shard(path, table)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        loaded = rb.load_shard(path)
        self.assertEqual(loaded.posts, 670)
        self.assertEqual(loaded.indptr.tolist(), table.indptr.tolist())
        self.assertEqual(loaded.nbr.tolist(), table.nbr.tolist())
        self.assertEqual(rb.read_shard_meta(path)["shard"], "g")
        self.assertIsNone(rb.read_shard_meta(self.root / "missing.npz"))
        self.assertIsNone(rb.load_shard(self.root / "missing.npz"))


if __name__ == "__main__":
    unittest.main()
