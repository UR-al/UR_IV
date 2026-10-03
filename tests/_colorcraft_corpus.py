"""Colorcraft 비교 말뭉치 — 고정 시드로 만든 설정(``core.colorcraft.Config``)과 손으로 고른 문자열·dict.

tests/test_colorcraft_spec_parity.py 가 이 설정들을 앱과 설치된 확장 양쪽에 넣어 같은 결과를 요구한다. 말뭉치가 덮는
것: 종류 7 × 패스 3, 마스크 없음·잎·조합(이어진 조합·미완성 조합·스스로/뒤 조합을 가리키는 값), 고급 스케줄 켬/끔, Advanced
스위치 끔, 아무것도 바꾸지 않는 항목(강도 0·모든 값 0), 슬라이더 범위 밖 값, ±0.0, 유효숫자 10자리를 넘는 float.
시드가 같으면 늘 같은 말뭉치다(``corpus()``).
"""
from __future__ import annotations

import random

from core import colorcraft as cc

SEED = 20261004
COUNT = 600

# 숫자 칸에 넣는 값 — 범위 밖·-0.0·10자리 넘는 값·아주 작은 값을 섞는다.
_SPECIAL_FLOATS = (0.0, -0.0, 1.0, -1.0, 0.5, 0.25, -0.25, 0.3, 0.1 + 0.2, 1e-7, -1e-7, 123.456789, 0.1234567891234,
                   1 / 3, 2.5, -4.0, 4.0, 10.0, -12.5, 64.0, 1e-12, 1234567.891, 0.75)


def _float(rng: random.Random, f: cc.Field) -> float:
    roll = rng.random()
    if roll < 0.35:
        return rng.choice(_SPECIAL_FLOATS)
    if roll < 0.75:
        return round(rng.uniform(f.minimum, f.maximum), 2)
    if roll < 0.9:
        return rng.uniform(f.minimum, f.maximum)                   # 17자리
    return rng.uniform(f.minimum * 3 - 1, f.maximum * 3 + 1)      # 범위 밖


def _random_value(rng: random.Random, f: cc.Field, choices=None):
    if f.kind == cc.FLOAT:
        return _float(rng, f)
    if f.kind == cc.BOOL:
        return rng.random() < 0.5
    return rng.choice(tuple(choices) if choices is not None else f.choices)


def _modifier(rng: random.Random, config: cc.Config, index: int) -> None:
    m = config.modifiers[index]
    m["active"] = rng.random() < 0.5
    m["kind"] = rng.choice(cc.KIND_CHOICES)
    m["pass"] = rng.choice(cc.PASS_CHOICES)
    roll = rng.random()
    m["mask"] = cc.MASK_NONE if roll < 0.35 else rng.choice(cc.MASK_TAGS if roll < 0.65 else cc.COMBO_TAGS)
    for name in ("strength", "start", "end"):
        if rng.random() < 0.5:
            m[name] = _float(rng, cc.MODIFIER_BY_NAME[name])
    if rng.random() < 0.12:
        m["strength"] = rng.choice((0.0, -0.0))                    # 스케줄 0
    m["advanced"] = rng.random() < 0.4
    for name in ("exponent", "bias", "start_off", "end_off"):
        if rng.random() < 0.4:
            m[name] = _float(rng, cc.MODIFIER_BY_NAME[name])
    m["smooth"] = rng.random() < 0.7
    params = [name for _, name in cc.KIND_PARAMS[cc.kind_of(m)]]
    others = [f.name for f in cc.MODIFIER_FIELDS[13:] if f.name not in params]
    if rng.random() < 0.15:
        return                                                      # 모든 값 0 = 아무것도 바꾸지 않는 항목
    for name in params:
        f = cc.MODIFIER_BY_NAME[name]
        if f.kind == cc.FLOAT and rng.random() < 0.35:
            m[name] = _float(rng, f)
        elif f.kind != cc.FLOAT and rng.random() < 0.3:
            m[name] = _random_value(rng, f)
    for name in rng.sample(others, k=min(len(others), rng.randint(0, 3))):   # 다른 종류의 칸(쓰지 않아야 한다)
        m[name] = _random_value(rng, cc.MODIFIER_BY_NAME[name])


def _leaf(rng: random.Random, config: cc.Config, index: int) -> None:
    leaf = config.leaves[index]
    for f in cc.LEAF_FIELDS:
        if rng.random() < 0.35:
            leaf[f.name] = _random_value(rng, f)


def _combo(rng: random.Random, config: cc.Config, index: int) -> None:
    combo = config.combos[index]
    refs = cc.combo_ref_choices(index)
    for name in ("mask_a", "mask_b"):
        roll = rng.random()
        if roll < 0.15:
            combo[name] = cc.MASK_NONE                              # 미완성
        elif roll < 0.2:
            combo[name] = rng.choice(cc.COMBO_TAGS[index:])        # 스스로·뒤 조합(확장 표가 막는 값 — 직접 넣은 설정)
        else:
            combo[name] = rng.choice(refs[1:])
    for f in cc.COMBO_FIELDS[2:]:
        if rng.random() < 0.3:
            combo[f.name] = _random_value(rng, f)


def _random_config(rng: random.Random) -> cc.Config:
    config = cc.default_config() if rng.random() < 0.5 else cc.blank_config()
    config.enabled = rng.random() < 0.9
    config.masking = rng.random() < 0.6
    for i in range(cc.MODIFIER_COUNT):
        if rng.random() < 0.6:
            _modifier(rng, config, i)
    for i in range(cc.MASK_COUNT):
        if rng.random() < 0.4:
            _leaf(rng, config, i)
    for i in range(cc.COMBO_COUNT):
        if rng.random() < 0.5:
            _combo(rng, config, i)
    return config


def _hand_picked() -> list:
    out = [cc.default_config(), cc.blank_config()]
    on = cc.default_config()
    on.enabled = True
    out.append(on)                                                   # 켰지만 바꾸는 값 없음
    for kind in cc.KIND_CHOICES:
        for pass_ in cc.PASS_CHOICES:
            config = cc.default_config()
            config.enabled = True
            m = config.modifiers[0]
            m.update({"kind": kind, "pass": pass_})
            for _, name in cc.KIND_PARAMS[cc.kind_of(m)]:
                f = cc.MODIFIER_BY_NAME[name]
                if f.kind == cc.FLOAT:
                    m[name] = 0.5
            out.append(config)
    gates = cc.default_config()
    gates.enabled = True
    gates.modifiers[0].update({"more_colors": False, "color_shift": False, "lab_a": 1.0, "color_shift_amount": 1.0})
    out.append(gates)                                                # 스위치가 끈 값만 = 아무것도 바꾸지 않음
    chain = cc.default_config()
    chain.enabled = chain.masking = True
    chain.modifiers[0].update({"mask": "C3", "exposure": 0.4})
    chain.modifiers[1].update({"active": True, "kind": "Basic", "mask": "M4", "contrast": 1.0})   # Basic 은 마스크 없음
    chain.modifiers[2].update({"active": True, "kind": "Shift", "mask": "C5", "color_shift_amount": 1.0})
    chain.combos[0].update({"mask_a": "M1", "mask_b": "M2", "operation": "or", "blur": 4.0})
    chain.combos[1].update({"mask_a": "C1", "mask_b": "M3"})
    chain.combos[2].update({"mask_a": "C2", "mask_b": "M5", "normalize": True})
    chain.combos[4].update({"mask_a": "C4", "mask_b": "M6"})          # C4 미완성 → C5 도 미완성
    chain.leaves[4].update({"spread": 1.0})
    out.append(chain)
    odd = cc.default_config()
    odd.enabled = odd.masking = True
    odd.modifiers[0].update({"kind": "Turbo", "mask": "M1", "exposure": 0.5, "chroma_plane": "lab"})   # 모르는 종류
    odd.modifiers[1].update({"active": True, "kind": "chroma_plus", "lab_a": 1.0})                     # 라벨 아닌 id
    out.append(odd)
    return out


def corpus(seed: int = SEED, count: int = COUNT) -> list:
    """손으로 고른 설정 + 고정 시드 무작위 설정(합계 ``count`` 이상)."""
    rng = random.Random(seed)
    configs = _hand_picked()
    while len(configs) < count:
        configs.append(_random_config(rng))
    return configs


def to_ext(spec, config: cc.Config):
    """앱 ``Config`` → 확장 ``spec.Config``(같은 dict 값, 디버그 칸은 확장 기본값)."""
    ext = spec.default_config()
    ext.enabled, ext.masking = config.enabled, config.masking
    ext.modifiers = [dict(m) for m in config.modifiers]
    ext.leaves = [dict(leaf) for leaf in config.leaves]
    ext.combos = [dict(c) for c in config.combos]
    return ext


# 손으로 고른 infotext 값(자기 형식) — 공백·중복·모르는 열쇠·읽을 수 없는 값·대소문자.
OWN_TEXTS = (
    "v1;mods=",
    "v1;mods=I",
    "v1;mods=I;I.exposure=0.3",
    " v1 ; mods = I , III ; I.exposure = 0.3 ; III.kind = punch ; III.clarity=0.5 ",
    "v1;mods=I,I,II,XI,i,;I.exposure=abc;II.kind=CHROMA_PLUS;II.temperature=1e400;II.tint=nan",
    "v1;mods=II;masking=1;masks=M9;combos=C5;II.mask=C2;C2.mask_a=C1;C2.mask_b=M1;C1.mask_a=M2;C1.mask_b=M3",
    "v1;mods=III;masking=yes;III.mask=c1;C1.mask_a=C1;C1.mask_b=M2",
    "v1;mods=I;masking=0;I.mask=M1;M1.mask_axis=HUE;M1.mask_mode=Protect Range;M1.blur=8",
    "v1;mods=I;I.active=0;I.pass=hires;I.advanced=on;I.smooth=off;I.bias=0.25",
    "v1;mods=I;I.exposure=1=2;=3;I.;.exposure=1;I.exposure",
    "v1;mods=X;X.color_shift_mode=LEGACY;X.chroma_plane=LAB;X.kind=shift;X.color_shift_amount=-0",
    "v1;mods=I;enabled=0;debug=1;debug_step=9;masks=M1",
    "v1;mods=I;masking=maybe;I.mask=M1",
    "v1;mods=I;masking=;I.mask=M1",
    "v1;mods=I;I.exposure=1_000;I.contrast=0x10;I.clarity= 2.5e-3 ",
    "v2;mods=I;I.exposure=0.3",
    "mods=I;I.exposure=0.3",
    "garbage",
    "",
)


# ── 남의 형식(붙여 넣기) ──────────────────────────────────────────────────────
def _messy_number(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.6:
        return repr(round(rng.uniform(-5, 5), rng.randint(0, 4)))
    if roll < 0.75:
        return rng.choice(("0", "1", "-0.0", "1e-7", "1e400", "2.5", "0.75"))
    return rng.choice(("abc", "", "nan", "inf", " 0.5 ", "1,5", "True"))


def upstream_text(rng: random.Random) -> str:
    """업스트림 Forge 스크립트 형식 ``MOD(태그:값,…;…)MASK(…)COMBO(…)`` — 값 개수가 틀리거나 못 읽는 값도 섞는다."""
    import core.colorcraft as module

    def values_for(names):
        out = []
        for name in names:
            if name in ("advanced", "smooth", "normalize", "hires"):
                out.append(rng.choice(("True", "False", "true", "1")))
            elif name in ("mask", "mask_a", "mask_b"):
                out.append(rng.choice(cc.MASK_CHOICES + ("M11", "c1", "")))
            elif name == "color_shift_mode":
                out.append(rng.choice(("default", "legacy", "Legacy", "x")))
            elif name == "mask_axis":
                out.append(rng.choice(cc.MASK_AXIS_OPTIONS + ("luma",)))
            elif name == "mask_mode":
                out.append(rng.choice(cc.MASK_MODE_OPTIONS + ("Highs",)))
            elif name == "operation":
                out.append(rng.choice(cc.COMBO_OP_OPTIONS + ("nand",)))
            else:
                out.append(_messy_number(rng))
        roll = rng.random()
        if roll < 0.1:
            out = out[:rng.randint(0, len(out))]
        elif roll < 0.15:
            out += ["9", "9"]
        return ",".join(out)

    def section(label, tags, names):
        picked = rng.sample(tags, k=rng.randint(0, min(4, len(tags))))
        body = ";".join(f"{tag}:{values_for(names)}" for tag in picked)
        return f"{label}({body})" if rng.random() < 0.9 else ""

    text = (section("MOD", list(cc.MODIFIER_TAGS) + ["XI", "i"], module.UPSTREAM_MODIFIER_VALUE_FIELDS)
            + section("MASK", list(cc.MASK_TAGS) + ["M0"], module.UPSTREAM_LEAF_FIELDS)
            + section("COMBO", list(cc.COMBO_TAGS) + ["C6"], module.UPSTREAM_COMBO_FIELDS))
    return text if "MOD(" in text else "MOD()" + text


def fork_text(rng: random.Random) -> str:
    """aoleg 포크 형식 ``v1;name=value;…`` — 칸 일부, 모르는 칸, 형이 틀린 값, 앞뒤 공백 붙은 선택지."""
    import core.colorcraft as module

    parts = ["v1"]
    names = list(module.FORK_PARAMS) + ["unknown", "mask_b"]
    for name in rng.sample(names, k=rng.randint(0, 25)):
        kind = module.FORK_PARAMS.get(name, ("float", 0.0))[0]
        if kind == "bool":
            value = rng.choice(("1", "0", "true", "True", "TRUE", "yes", "no", " 1 "))
        elif kind == "choice":
            choices = tuple(module.FORK_CHOICES.get(name, ("x",)))
            value = rng.choice(choices + (" " + choices[-1], choices[-1] + " ", choices[-1].upper(), "zzz"))
        else:
            value = _messy_number(rng)
        parts.append(f"{name}={value}")
    for forced in ("masking", "mask_combine", "mask_preview", "advanced"):
        if rng.random() < 0.4:
            parts.append(f"{forced}=1")
    if rng.random() < 0.3:
        parts.append(f"mask_blur_radius={rng.choice(('0', '3', '-1'))};mask_spread={rng.choice(('0', '0.5'))}")
    return ";".join(parts)


UPSTREAM_TEXTS = (
    "MOD(I:none,1.0,0.5,0.75,False,0.0,0.5,0.0,0.0,True,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
    "0.0,0.0,0.0,0.0,0.0,default,0.0,0.0,0.0,0.0,False)",
    "MOD(II:M1,0.8,0.2,0.9,True,1.5,0.25,0.1,-0.1,False,0.0,0.0,1.0,0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
    "0.0,0.0,0.0,0.0,1.0,legacy,0.5,-0.5,0.0,0.0,True)MASK(M1:hue,lows,1.0,0.5,0.0,2.0,8.0,0.0,0.0,False)",
    "MOD(III:C1,1.0)MASK(M1:exposure,highs;M2:tint,split,0.5)COMBO(C1:M1,M2,xor,0.0,0.0,0.0,True)",
    "MOD()MASK()COMBO()",
    "MOD(I:none)",
    "prefix MOD(V:M3,abc,,0.75) suffix",
    "MOD( II :M1,0.5; III:none,0.25)MASK( M1 :hue,lows)COMBO( C1 :M1,M2)",
)
FORK_TEXTS = (
    "v1",
    "v1;exposure=0.3",
    "v1;exposure=0.3;apply_to_hr=0;advanced=1;bias=0.3;exponent=2;start_off=0.1",
    "v1;bias=0.25;chroma_center=0.75",
    "v1;masking=1;mask_mode=lows;mask_axis=hue;mask_blur_radius=4;mask_spread=0.5",
    "v1;masking=1;mask_combine=1;mask_operation=or;mask_b_axis=tint;mask_b_mode=range;mask_spread=1",
    "v1;masking=1;mask_preview=1;mask_preview_color=blue;exposure=1",
    "v1;masking=1;mask_axis=clarity;mask_mode= lows;mode=LEGACY;chroma_plane=lab;dev=1;recenter_override=yes",
    "v1 ; exposure = 0.5 ; ; = ; contrast",
    "v10;exposure=0.3",
)
