"""Pack ``dpmpp_4m_sde``: the fourth-order step on unequal λ steps (sam-extra's fix of Clybius' ratios).

Clybius' fourth-order branch divided the history differences by ``h_3/h_2``, ``h_2/h`` and ``h/h_1`` —
three different bases — so on unequal λ steps (any flow schedule, Karras) even an x0 history linear in λ
gave a wrong slope and a curvature that is not there (review CP-02 / SAM-07). The pack and sam-extra now
compute the terms in ``_fourth_order_terms`` with divided differences:

* ``FourthOrderTermsTests`` — pure, rational arithmetic (no torch): on equal steps exactly upstream's
  ``d1``/``d2``; a history of degree ≤ 2 in λ gives ``h·P'(λ_s)``, ``h²·P''(λ_s)/2`` (the DPM++ 3M SDE terms)
  on any steps; the cubic part is upstream's Newton term damped to 3/8 (slope) and 1/4 (curvature);
  ``d1/h``, ``d2/h²`` depend on the history alone.
* ``ExtensionTermsTests`` — torch + the installed extension: the pack's terms == sam-extra's, bit for bit,
  and an x0 quadratic in λ samples like the host's ``sample_dpmpp_3m_sde`` at every step. The whole sampler
  is compared with sam-extra in ``tests/test_comfy_extra_samplers_parity.Dpmpp4mParityTests``.
"""
from __future__ import annotations

import itertools
import types
import unittest
from fractions import Fraction as Q

from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dpmpp_4m as m4
from tests._forge_k_diffusion import Predictor, forge_k_sampling, import_ext, installed_k_sampling
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import requires_extension

torch = None  # filled by bind_torch() in the @requires_torch classes


def upstream_terms(d, h, h_1, h_2, h_3):
    """Clybius' fourth-order ``d1``/``d2`` (``sample_clyb_4m_sde_momentumized``, BSD-3-Clause), as ported."""
    r0, r1, r2 = h_3 / h_2, h_2 / h, h / h_1
    d1_0 = (d[0] - d[1]) / r2
    d1_1 = (d[1] - d[2]) / r1
    d1_2 = (d[2] - d[3]) / r0
    d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
    d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
    return d1, d2


def three_m_terms(d, h, h_1, h_2):
    """The DPM++ 3M SDE step's ``d1``/``d2`` (k-diffusion / ComfyUI) from the three newest points."""
    r0, r1 = h_1 / h, h_2 / h
    d1_0 = (d[0] - d[1]) / r0
    d1_1 = (d[1] - d[2]) / r1
    return d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1), (d1_0 - d1_1) / (r0 + r1)


def history(poly, h_1, h_2, h_3):
    """x0 at λ_s = 0 and at λ_s − h_1, … (``poly`` lists the coefficients of λ^k)."""
    nodes = (Q(0), -h_1, -h_1 - h_2, -h_1 - h_2 - h_3)
    return [sum(c * lam ** k for k, c in enumerate(poly)) for lam in nodes]


STEPS = [tuple(Q(v) for v in s) for s in (
    ("2/5", "3/10", "1/5", "1/10"),     # the review's counter-example
    ("1/10", "1/5", "3/10", "2/5"),
    ("7/4", "1/3", "5/2", "3/8"),
    ("1/4", "1/4", "1/2", "1/8"),
    ("3/2", "1/8", "1/8", "1/8"),
)]
HISTORIES = ([Q(3, 5), Q(3, 10), Q(1, 10), Q(0)], [Q(1), Q(-2, 7), Q(5, 3), Q(-9, 11)],
             [Q(13, 17), Q(1, 19), Q(-23, 29), Q(31, 37)])


def terms(d, steps):
    return m4._fourth_order_terms(*d, *steps)


class FourthOrderTermsTests(unittest.TestCase):
    def test_equal_steps_are_upstreams_terms_exactly(self):
        for d, h in itertools.product(HISTORIES, (Q(1, 4), Q(7, 10), Q(3))):
            with self.subTest(d=d, h=h):
                self.assertEqual(terms(d, (h,) * 4), upstream_terms(d, h, h, h, h))

    def test_polynomials_up_to_degree_two_are_exact_on_any_steps(self):
        for poly, steps in itertools.product(([Q(3, 5)], [Q(3, 5), Q(1)], [Q(1), Q(2), Q(3)],
                                              [Q(-1, 3), Q(5, 7), Q(-11, 13)]), STEPS):
            with self.subTest(poly=poly, steps=steps):
                h = steps[0]
                d = history(poly, *steps[1:])
                slope = poly[1] if len(poly) > 1 else 0
                half_curvature = poly[2] if len(poly) > 2 else 0
                self.assertEqual(terms(d, steps), (h * slope, h * h * half_curvature))
                self.assertEqual(terms(d, steps), three_m_terms(d, *steps[:3]))

    def test_upstream_terms_are_wrong_on_unequal_steps(self):
        steps = STEPS[0]
        d = history([Q(3, 5), Q(1)], *steps[1:])     # x0 = 0.6 + λ: slope 1, no curvature
        self.assertEqual(terms(d, steps), (Q(2, 5), 0))
        d1, d2 = upstream_terms(d, *steps)
        self.assertAlmostEqual(float(d1), -0.06756198347107438, places=15)
        self.assertAlmostEqual(float(d2), -0.2194214876033058, places=15)

    def test_the_cubic_part_is_upstreams_damped_newton_term(self):
        for poly, steps in itertools.product(([0, 0, 0, Q(1)], [Q(1, 2), Q(-3), Q(2, 9), Q(7, 5)]), STEPS):
            with self.subTest(poly=poly, steps=steps):
                h, h_1, h_2, h_3 = steps
                d = history(poly, h_1, h_2, h_3)
                f01, f12, f23 = (d[0] - d[1]) / h_1, (d[1] - d[2]) / h_2, (d[2] - d[3]) / h_3
                f012, f123 = (f01 - f12) / (h_1 + h_2), (f12 - f23) / (h_2 + h_3)
                f0123 = (f012 - f123) / (h_1 + h_2 + h_3)
                self.assertEqual(f0123, poly[3])
                # Newton: P'(λ_s) = f01 + h_1·f012 + h_1(h_1 + h_2)·f0123, P''(λ_s)/2 = f012 + (2h_1 + h_2)·f0123
                self.assertEqual(terms(d, steps),
                                 (h * (f01 + h_1 * f012 + Q(3, 8) * h_1 * (h_1 + h_2) * f0123),
                                  h * h * (f012 + Q(1, 4) * (2 * h_1 + h_2) * f0123)))

    def test_estimates_depend_on_the_history_alone(self):
        d = HISTORIES[2]
        history_steps = (Q(1, 3), Q(5, 2), Q(3, 8))
        d1_a, d2_a = terms(d, (Q(7, 4),) + history_steps)
        d1_b, d2_b = terms(d, (Q(1, 9),) + history_steps)
        self.assertEqual(d1_a / Q(7, 4), d1_b / Q(1, 9))
        self.assertEqual(d2_a / Q(7, 4) ** 2, d2_b / Q(1, 9) ** 2)


class QuadraticX0Model:
    """``model(x, sigma)``: an x0 quadratic in the half-log-SNR λ(σ), the same for every pixel."""

    def __init__(self, predictor, kd, coefficients=(0.3, 0.2, -0.05)):
        self.inner_model = types.SimpleNamespace(predictor=predictor)
        self.kd = kd
        self.coefficients = coefficients

    def __call__(self, x, sigma, **kwargs):
        lam = self.kd.sigma_to_half_log_snr(sigma, self.inner_model.predictor)
        value = sum(c * lam ** k for k, c in enumerate(self.coefficients))
        return value.reshape(-1, *([1] * (x.ndim - 1))) + torch.zeros_like(x)


class _Noise:
    def __init__(self, like, seed):
        self.like = like
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, sigma, sigma_next):
        return torch.randn(self.like.shape, generator=self.generator, dtype=self.like.dtype)


@requires_torch
@requires_extension
class ExtensionTermsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.kd = forge_k_sampling()
        with installed_k_sampling(cls.kd):
            cls.ext = import_ext("sam3ext.extra_samplers.dpmpp_4m_sde")

    def test_the_terms_are_the_extensions_bit_for_bit(self):
        for d, steps in itertools.product(HISTORIES, STEPS + [(Q(1, 4),) * 4]):
            with self.subTest(d=d, steps=steps):
                self.assertEqual(terms(d, steps), self.ext._fourth_order_terms(*d, *steps))
        generator = torch.Generator().manual_seed(11)
        for dtype in (torch.float32, torch.float64):
            for _ in range(20):
                d = [torch.randn((2, 4, 3, 5), generator=generator, dtype=torch.float64).to(dtype) for _ in range(4)]
                steps = [(torch.rand((), generator=generator, dtype=torch.float64) + 0.05).to(dtype) for _ in range(4)]
                ours = terms(d, steps)
                theirs = self.ext._fourth_order_terms(*d, *steps)
                for a, b in zip(ours, theirs):
                    self.assertTrue(torch.equal(a, b))

    def test_a_quadratic_x0_history_samples_like_the_hosts_3m_sde_at_every_step(self):
        for flow, eta in itertools.product((False, True), (0.0, 1.0)):
            with self.subTest(flow=flow, eta=eta):
                if flow:
                    t = torch.linspace(1.0, 0.0, 13, dtype=torch.float64)
                    sigmas = 3.0 * t / (1 + 2.0 * t)
                else:
                    ramp = torch.linspace(0, 1, 12, dtype=torch.float64)
                    max_inv, min_inv = 14.6146 ** (1 / 7), 0.0292 ** (1 / 7)
                    sigmas = torch.cat([(max_inv + ramp * (min_inv - max_inv)) ** 7, torch.zeros(1, dtype=torch.float64)])
                x = torch.randn((1, 4, 4, 4), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
                x = x * float(sigmas[0])
                ours, host = [], []
                with installed_k_sampling(self.kd):
                    m4.sample_dpmpp_4m_sde(QuadraticX0Model(Predictor(flow), self.kd), x.clone(), sigmas, extra_args={},
                                           disable=True, eta=eta, noise_sampler=_Noise(x, 5), kd=self.kd,
                                           callback=lambda d: ours.append(d["x"]))
                    self.kd.sample_dpmpp_3m_sde(QuadraticX0Model(Predictor(flow), self.kd), x.clone(), sigmas,
                                                extra_args={}, disable=True, eta=eta, noise_sampler=_Noise(x, 5),
                                                callback=lambda d: host.append(d["x"]))
                self.assertEqual(len(ours), len(sigmas) - 1)
                for i, (a, b) in enumerate(zip(ours, host)):
                    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11, msg=f"step {i}")


if __name__ == "__main__":
    unittest.main()
