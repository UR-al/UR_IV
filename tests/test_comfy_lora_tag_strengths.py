"""``<lora:…>`` 태그의 강도·문법 — ComfyUI 컴파일러도 Forge 와 똑같이 읽는다(사용자 결정 2026-09-27 "Forge와 똑같이").

Forge(읽기 전용 참조 C:\\sd-webui-forge-classic):
- modules/extra_networks.py:157 ``re_extra_net = <(\\w+):([^>]+)>`` 로 태그를 찾아 프롬프트에서 떼고(parse_prompt
  :160-173), 인자를 ``:`` 마다 나눈다. ``ExtraNetworkParams``(:26-37)는 ``=`` 가 하나인 항목을 이름 인자로, 나머지를
  위치 인자로 둔다.
- extensions-builtin/sd_forge_lora/extra_networks_lora.py:37-41 — 텍스트 인코더 = 둘째 위치 인자(``@`` 가 있으면 0.0,
  없으면 1.0), 이어서 ``te=`` 가 이긴다. UNet = 셋째 위치 인자(없으면 그 텍스트 인코더 값), 이어서 ``unet=`` 가 이긴다.
  ``dyn=`` 같은 다른 이름 인자는 강도와 상관없다.
- ComfyUI 로더는 strength_model = UNet, strength_clip = 텍스트 인코더다. 그래서 ``<lora:n:a:b>`` 는 clip a·model b
  (예전 컴파일러는 a 를 model, b 를 clip 으로 읽었다), ``<lora:n:a>`` 는 둘 다 a, ``<lora:n>`` 은 둘 다 1.0 이다.
- 거는 것은 이름이 정확히 ``lora`` 인 태그뿐이다(extra_network_registry 키 — ``<LoRA:x>``·``<lyco:x>``·``<hypernet:x>``
  는 모르는 네트워크로 떼기만 한다, :73-104 lookup_extra_networks — :100 'Skipping unknown extra network'). 콜론 앞에
  공백이 있는 ``<lora :x>`` 는 태그가 아니라 글자 그대로다.
- 태그는 ``<이름:`` 에서 첫 ``>`` 까지다. 그래서 닫히지 않은 ``<3:``·``<hypernet:x`` 뒤의 ``<lora:…>`` 는 그 모르는
  네트워크 태그에 삼켜져 떼어지기만 하고 걸리지 않는다. positive 글(메인·ADetailer·SAM3 인페인트)은 Forge 처럼 모든
  태그를 뗀다. 네거티브는 Forge 가 파싱하지 않는다 — 메인 Comfy 경로의 네거티브 LoRA(앱 규약)는 lora 태그만 떼고 걸며,
  보조 패스 네거티브는 글자 그대로다.

한 가중치·가중치 없는 태그는 이 변경 전과 바이트 단위로 같은 그래프여야 한다 — 옛 파서를 그대로 옮긴
``_old_parse_lora_tags`` 로 컴파일한 그래프와 비교한다.
"""
import copy
import json
import re
import unittest
from unittest import mock

from core import sam3_args
from core import comfy_workflow_compiler as compiler_module
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, LoraSpec, parse_lora_tags
from core.lenient_numbers import finite_float
from core.lora_stack import append_lora_stack_to_prompt, build_lora_text
from tests.test_comfy_anima38_compiler import _anima_capabilities, _modules, V2_MODEL
from tests.test_comfy_sam3_lora_branch import _upstream_loras
from tests.test_comfy_workflow_compiler import _capabilities, _custom_workflow

_LORA_CLASSES = {"LoraLoader", "ForgeNeoAnimaLoraLoader"}

# ---- 이 변경 전 컴파일러의 파서(바이트 동일 비교용 — core/comfy_workflow_compiler.py 옛 _LORA_RE·parse_lora_tags) ----
_OLD_LORA_RE = re.compile(
    r"<lora\s*:\s*([^:>]+?)\s*(?::\s*([^:>]+?))?\s*(?::\s*([^>]+?))?\s*>",
    re.IGNORECASE,
)


def _old_strip(text):
    """옛 컴파일러가 보조 패스 글(ADetailer Impact 글·SAM3 인페인트 글)에서 태그를 떼던 식."""
    return _OLD_LORA_RE.sub("", text)


def _old_parse_lora_tags(*prompts):
    loras, cleaned = [], []
    for prompt in prompts:
        text = str(prompt or "")
        for match in _OLD_LORA_RE.finditer(text):
            name = match.group(1).strip()
            if not name:
                continue
            model_strength = finite_float(match.group(2), 1.0)
            clip_strength = finite_float(match.group(3), model_strength)
            loras.append(LoraSpec(name, model_strength, clip_strength))
        text = re.sub(r"(?:\s*,\s*){2,}", ", ", _old_strip(text))     # 옛 _clean_prompt_after_loras
        cleaned.append(text.strip(" \t\r\n,"))
    return loras, cleaned


# ---- Forge 원본 규칙을 그대로 옮긴 기준(위 모듈 독스트링의 file:line) -------------------------------------------
_FORGE_EXTRA_NET_RE = re.compile(r"<(\w+):([^>]+)>")


def _forge_params(args):
    """ExtraNetworkParams(modules/extra_networks.py:26-37) — (positional, named)."""
    positional, named = [], {}
    for item in args.split(":"):
        parts = item.split("=", 2)
        if len(parts) == 2:
            named[parts[0]] = parts[1]
        else:
            positional.append(item)
    return positional, named


def _forge_strengths(args):
    """ExtraNetworkLora.activate(extra_networks_lora.py:35-44) — (이름, te, unet). 숫자가 아니면 Forge 처럼 예외."""
    positional, named = _forge_params(args)
    te = (0.0 if "@" in positional[1] else float(positional[1])) if len(positional) > 1 else 1.0
    te = float(named.get("te", te))
    unet = float(positional[2]) if len(positional) > 2 else te
    unet = float(named.get("unet", unet))
    return positional[0], te, unet


def _forge_lora_parse(text):
    """parse_prompt(:160-173) + lookup_extra_networks(:73-104) — (모든 추가 네트워크 태그를 뗀 글, 걸리는 (이름, te,
    unet) 목록). 이름과 상관없이 모든 ``<이름:…>`` 를 떼고, 걸리는 것은 등록된 정확히 'lora' 뿐이다."""
    loaded = []

    def found(match):
        if match.group(1) == "lora":
            strengths = _forge_strengths(match.group(2))
            if strengths[0]:        # 이름 '' 는 networks.load_networks 가 찾지 못해 건너뛴다(networks.py:154-176)
                loaded.append(strengths)
        return ""

    return _FORGE_EXTRA_NET_RE.sub(found, text), loaded


def _specs(text):
    return [(spec.name, spec.strength_model, spec.strength_clip) for spec in parse_lora_tags(text)[0]]


class ForgeStrengthOrderTests(unittest.TestCase):
    def test_positional_weights_are_text_encoder_then_unet(self):
        """(이름, strength_model=UNet, strength_clip=텍스트 인코더)."""
        cases = {
            "<lora:ink>": [("ink", 1.0, 1.0)],
            "<lora:ink:0.7>": [("ink", 0.7, 0.7)],
            "<lora:ink:0.7:0.3>": [("ink", 0.3, 0.7)],
            "<lora:ink:-0.5:1.25>": [("ink", 1.25, -0.5)],
            "<lora:ink:1:0.5:0.25>": [("ink", 0.5, 1.0)],          # 넷째 이후 위치 인자는 강도가 아니다
            "<lora:ink:0.5@10>": [("ink", 0.0, 0.0)],              # '@' → 텍스트 인코더 0, UNet 도 그 값
            "<lora:ink:0.5@10:0.8>": [("ink", 0.8, 0.0)],
            "<lora: ink : 0.7 : 0.3 >": [("ink", 0.3, 0.7)],       # float(' 0.7 ') — Forge 도 숫자 둘레 공백을 받는다
            "a, <lora:ink:0.9:0.1>, b <lora:alice:0.2>": [("ink", 0.1, 0.9), ("alice", 0.2, 0.2)],
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(_specs(text), expected)
                forge = [(name.strip(), unet, te) for name, te, unet in _forge_lora_parse(text)[1]]
                self.assertEqual(_specs(text), forge)

    def test_named_te_and_unet_arguments_like_forge(self):
        cases = {
            "<lora:ink:te=0.3>": [("ink", 0.3, 0.3)],               # UNet 기본값 = te 가 덮은 텍스트 인코더 값
            "<lora:ink:unet=0.2>": [("ink", 0.2, 1.0)],
            "<lora:ink:te=0.3:unet=0.6>": [("ink", 0.6, 0.3)],
            "<lora:ink:unet=0.6:te=0.3>": [("ink", 0.6, 0.3)],
            "<lora:ink:0.5:te=0.3>": [("ink", 0.3, 0.3)],
            "<lora:ink:0.5:0.8:te=0.3>": [("ink", 0.8, 0.3)],
            "<lora:ink:0.5:0.8:unet=0.1:te=0.2>": [("ink", 0.1, 0.2)],
            "<lora:ink:te=0.3:0.8>": [("ink", 0.3, 0.3)],          # 위치 인자 0.8 은 둘째 → te 0.8 을 te= 가 덮는다
            "<lora:ink:te=0.3:0.8:0.6>": [("ink", 0.6, 0.3)],
            "<lora:ink:0.5:dyn=64>": [("ink", 0.5, 0.5)],           # dyn= 은 강도가 아니다
            "<lora:ink:0.5:0.8:dyn=64>": [("ink", 0.8, 0.5)],
            "<lora:ink:te=0.1:te=0.4>": [("ink", 0.4, 0.4)],        # 같은 이름은 뒤가 이긴다(dict)
            "<lora:ink: te=0.3>": [("ink", 1.0, 1.0)],             # 이름 인자 키는 ' te' — 'te' 가 아니다
            "<lora:ink=v2:0.80>": [("0.80", 1.0, 1.0)],            # '=' 가 하나 든 항목은 이름 인자 — 이름은 '0.80'
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(_specs(text), expected)
                forge = [(name.strip(), unet, te) for name, te, unet in _forge_lora_parse(text)[1]]
                self.assertEqual(_specs(text), forge)

    def test_the_compiled_loader_gets_clip_first_weight_and_model_second(self):
        """메인 스택·사용자 워크플로·단독 후처리·외부 워크플로 매핑의 LoRA 로더가 모두 같은 파서를 지난다."""
        payload = {"prompt": "portrait, <lora:ink:0.9:0.2>, <lora:alice:te=0.4:unet=0.6>", "seed": 1}
        expected = [("styles/ink.safetensors", 0.2, 0.9), ("characters/alice.safetensors", 0.6, 0.4)]
        compiler = ComfyWorkflowCompiler(_capabilities())
        graphs = {
            "default": compiler.compile("txt2img", "checkpoint.safetensors", copy.deepcopy(payload)),
            "custom workflow": compiler.compile("txt2img", "checkpoint.safetensors", copy.deepcopy(payload),
                                                workflow=_custom_workflow()),
            "external workflow": compiler.map_external_workflow(
                _custom_workflow(), "txt2img", "checkpoint.safetensors", copy.deepcopy(payload)),
            "postprocess": compiler.compile_postprocess(
                "checkpoint.safetensors",
                {**copy.deepcopy(payload), "alwayson_scripts": sam3_args.build_alwayson(
                    {"sam3_mode": "Inpaint", "sam3_prompt": "face", "sam3_inpaint_prompt": payload["prompt"]})},
                uploaded_image="in.png"),
        }
        anima = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, {**copy.deepcopy(payload), "forge_additional_modules": _modules()})
        graphs["anima"] = anima
        for label, graph in graphs.items():
            with self.subTest(label):
                loaders = [node["inputs"] for node in graph.values() if node["class_type"] in _LORA_CLASSES]
                self.assertEqual([(inputs["lora_name"], inputs["strength_model"], inputs["strength_clip"])
                                  for inputs in loaders], expected)


class ForgeTagGrammarTests(unittest.TestCase):
    def test_stripping_removes_every_extra_network_tag_forge_consumes_and_nothing_else(self):
        """떼는 범위 = Forge parse_prompt 가 positive 프롬프트에서 떼는 모든 ``<이름:…>`` 태그(이름 인자 포함, lora 가
        아닌 이름도). 거는 것은 이름이 정확히 lora 인 것. 닫히지 않은 ``<3:``·``<hypernet:x`` 가 뒤의 ``<lora:…>`` 까지
        한 태그로 삼키면 Forge 는 그 LoRA 를 걸지 않는다 — 여기서도 걸지 않는다. 뗀 글에는 Impact wildcard 의 LoRA
        태그(``<lora:([^>]+)>`` — ComfyUI-Impact-Pack wildcards.py:845)가 남지 않는다."""
        cases = (
            "portrait, <lora:ink:te=0.5:unet=0.3>, 1girl",
            "<lora:ink:0.5:dyn=4>face",
            "a <LoRA:ink:0.5> b",                # Forge: 모르는 네트워크 'LoRA' — 떼지만 걸지 않는다
            "a <LORA:ink> b <lora:alice:0.2>",
            "a <lora::1> b",                     # Forge: 이름 '' — 떼지만 걸 것이 없다
            "a <lora:ink:> b",
            "a <lora :ink:0.5> b",               # Forge: 태그가 아니다 — 글자 그대로
            "a < lora:ink> b",
            "a <lyco:ink:0.5> b",                # Forge: 모르는 네트워크 'lyco'(별칭 없음) — 떼지만 걸지 않는다
            "a <hypernet:h:1> b <lora:alice:0.2>",
            "<lora_x:1> <lora:ink:0.5>",         # 이름 'lora_x' — lora 가 아니다
            "a <한글:x> b",                        # \w 는 유니코드 글자도 받는다
            "a <lora:ink:0.5:0.25> <lora:alice>",
            "(<lora:ink:0.5>:1.2), BREAK <lora:alice:0.1:0.9>",
            "1girl, (<3:1.2), smile, <lora:ink:0.8>",      # '<3:' 가 뒤 LoRA 태그의 '>' 까지 삼킨다 — 걸지 않는다
            "<hypernet:foo portrait, <lora:ink:0.8>",
            "(<3:1.2), <lora:ink:0.8>, <lora:alice:0.4>",  # 첫 LoRA 만 삼켜지고 둘째는 걸린다
            "<3 <lora:ink:0.5>",                 # '<3 ' 는 태그 시작이 아니다
        )
        for text in cases:
            with self.subTest(text):
                forge_text, forge_loaded = _forge_lora_parse(text.replace("<lora:ink:>", "<lora:ink:1>"))
                if "<lora:ink:>" in text:
                    forge_text = text.replace("<lora:ink:>", "")   # Forge 는 float('') 예외 — 떼는 범위만 비교
                stripped = compiler_module._strip_extra_networks(text)
                self.assertEqual(stripped, forge_text)
                if "<lora:ink:>" not in text:
                    self.assertEqual(_specs(text), [(name.strip(), unet, te) for name, te, unet in forge_loaded])
                self.assertEqual(re.findall(r"<lora:([^>]+)>", stripped), [])
                self.assertEqual(_FORGE_EXTRA_NET_RE.findall(parse_lora_tags(text)[1][0]), [])

    def test_the_negative_keeps_the_app_lora_convention_and_other_tags_as_text(self):
        """Forge 는 네거티브를 파싱하지 않는다. 메인 Comfy 경로의 네거티브 LoRA(앱 규약 — 걸고 뗀다)는 그대로이고, lora
        가 아닌 태그는 떼지 않는다(글자 그대로 인코딩)."""
        cases = {
            "bad, <lyco:ink:0.5>": ([], "bad, <lyco:ink:0.5>"),
            "bad, <lora:ink:0.5:0.2>": ([("ink", 0.2, 0.5)], "bad"),
            "bad, <LoRA:ink:0.5>": ([], "bad"),
            "bad, (<3:1.2), <lora:ink:0.8>": ([("ink", 0.8, 0.8)], "bad, (<3:1.2)"),
        }
        for negative, (expected, text) in cases.items():
            with self.subTest(negative):
                specs, cleaned = parse_lora_tags("portrait", negative)
                self.assertEqual([(s.name, s.strength_model, s.strength_clip) for s in specs], expected)
                self.assertEqual(cleaned, ["portrait", text])

    def test_every_positive_pass_text_drops_extra_network_tags_and_negatives_keep_them(self):
        """컴파일된 그래프: 메인 프롬프트·ADetailer 가 Impact 에 넘기는 글·SAM3 인페인트 글에서 Forge 가 떼는 태그가 모두
        빠지고(삼켜진 LoRA 는 걸리지 않는다), 네거티브(메인·ad_negative_prompt·sam3_negative_prompt)의 lora 가 아닌
        태그는 글자 그대로다. 로더 = Forge 가 거는 것."""
        scripts = sam3_args.build_alwayson({
            "sam3_mode": "Inpaint", "sam3_prompt": "face", "sam3_inpaint_prompt": "eyes <hypernet:h:1>, <lora:alice:0.3>",
            "sam3_negative_prompt": "worst <lyco:neg:1>"})
        scripts["ADetailer"] = {"args": [True, False, {
            "ad_tab_enable": True, "ad_model": "face_yolov8n.pt", "ad_prompt": "face (<3:1.2) <lora:ink:0.8>",
            "ad_negative_prompt": "lowres <lyco:neg:1>"}]}
        payload = {"prompt": "portrait, <lyco:old:0.5>, <lora:alice:0.3>", "negative_prompt": "bad <lyco:neg:1>",
                   "seed": 3, "alwayson_scripts": scripts}
        alice = ("characters/alice.safetensors", 0.3, 0.3)
        compiler = ComfyWorkflowCompiler(_capabilities())
        graphs = {
            "default": compiler.compile("txt2img", "checkpoint.safetensors", copy.deepcopy(payload)),
            "custom workflow": compiler.compile("txt2img", "checkpoint.safetensors", copy.deepcopy(payload),
                                                workflow=_custom_workflow()),
            "postprocess": compiler.compile_postprocess("checkpoint.safetensors", copy.deepcopy(payload),
                                                        uploaded_image="in.png"),
            "anima": ComfyWorkflowCompiler(_anima_capabilities()).compile(
                "txt2img", V2_MODEL, {**copy.deepcopy(payload), "forge_additional_modules": _modules()}),
            "external workflow": compiler.map_external_workflow(
                _custom_workflow(), "txt2img", "checkpoint.safetensors",
                {key: payload[key] for key in ("prompt", "negative_prompt", "seed")}),
        }
        for label, graph in graphs.items():
            with self.subTest(label):
                strings = []
                for node in graph.values():
                    for key, value in node["inputs"].items():
                        if key == "settings_json":
                            strings.append(json.loads(value)["prompt"])      # Impact 에 넘기는 글
                        elif isinstance(value, str):
                            strings.append(value)
                self.assertFalse([s for s in strings if re.search(r"<(lyco:old|hypernet|3:)", s)], strings)
                self.assertFalse([s for s in strings if "<lora:" in s], strings)
                self.assertIn("bad <lyco:neg:1>", strings)
                self.assertIn("portrait", strings)
                if label == "external workflow":
                    continue
                self.assertIn("worst <lyco:neg:1>", strings)
                self.assertIn("lowres <lyco:neg:1>", strings)
                [ad] = [n["inputs"] for n in graph.values() if n["class_type"] == "ForgeNeoADetailer"]
                self.assertEqual(json.loads(ad["settings_json"])["prompt"], "face (")
                [sam] = [n["inputs"] for n in graph.values() if n["class_type"] == "ForgeNeoSAM3Detailer"]
                self.assertEqual(sam["inpaint_prompt"], "eyes , ")
                self.assertEqual(_upstream_loras(graph, ad["model"], ad["clip"]), [])       # 삼켜진 ink 는 없다
                self.assertEqual(_upstream_loras(graph, sam["model"], sam["clip"]), [alice])
                self.assertNotIn("styles/ink.safetensors", json.dumps(graph))

    def test_uppercase_and_spaced_tags_are_not_loaded(self):
        """예전 컴파일러는 ``<LoRA:x>``·``<lora :x>`` 도 걸었다. Forge 는 둘 다 걸지 않는다."""
        specs, cleaned = parse_lora_tags("portrait, <LoRA:ink:0.5>, <lora :alice:0.4>")
        self.assertEqual(specs, [])
        self.assertEqual(cleaned, ["portrait, <lora :alice:0.4>"])

    def test_app_emitted_tags_keep_their_meaning(self):
        """앱이 쓰는 태그: LoRA 스택(core/lora_stack — ``<lora:이름:0.80>``)은 가중치 하나라 그대로(model = clip).
        sam-extra LoRA Manager 가 보내는 ``<lora:이름:강도:clip>``(sam3ext/lora_manager_core.py _BRIDGE_JS)은 Forge 가
        둘째 값을 텍스트 인코더, 셋째 값을 UNet 으로 읽는다 — ComfyUI 도 이제 Forge 와 같다."""
        stack = [{"name": "styles/ink", "weight": 0.8, "enabled": True},
                 {"name": "alice", "weight": -0.35, "enabled": True}]
        for text in (build_lora_text(stack, unit="multiplier"),
                     append_lora_stack_to_prompt("1girl, solo", stack)):
            with self.subTest(text):
                self.assertEqual(_specs(text), [("styles/ink", 0.8, 0.8), ("alice", -0.35, -0.35)])
                self.assertEqual(_specs(text), [(spec.name, spec.strength_model, spec.strength_clip)
                                                for spec in _old_parse_lora_tags(text)[0]])
        bridge = "<lora:" + "chars/alice" + ":" + "0.8" + ":" + "0.5" + ">"
        self.assertEqual(_specs(bridge), [("chars/alice", 0.5, 0.8)])
        self.assertEqual(_forge_lora_parse(bridge)[1], [("chars/alice", 0.8, 0.5)])


class SingleWeightGraphIdentityTests(unittest.TestCase):
    """한 가중치·가중치 없는 태그: 옛 파서로 컴파일한 그래프와 바이트 단위로 같다(메인·네거티브·SAM3·ADetailer·
    사용자 워크플로·외부 워크플로·단독 후처리·Anima)."""

    TAGS = ("<lora:ink>", "<lora:ink:0.8>", "<lora: ink : 0.8 >", "<lora:styles/ink.safetensors:-0.5>",
            "<lora:alice:1e-1>")

    def _graphs(self, tag):
        adetailer = {"args": [True, False,
                              {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt", "ad_prompt": f"face, {tag}"},
                              {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt", "ad_prompt": tag},
                              {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt"}]}

        def payload(**extra):
            scripts = sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face",
                                                "sam3_inpaint_prompt": f"eyes, {tag}"})
            scripts["ADetailer"] = copy.deepcopy(adetailer)
            scripts["NegPiP"] = {"args": [True]}
            return {"prompt": f"portrait, {tag}, <lora:alice:0.3>", "negative_prompt": f"bad, {tag}",
                    "seed": 3, "alwayson_scripts": scripts, **extra}

        compiler = ComfyWorkflowCompiler(_capabilities())
        anima = ComfyWorkflowCompiler(_anima_capabilities())
        return {
            "default": compiler.compile("txt2img", "checkpoint.safetensors", payload()),
            "custom workflow": compiler.compile("txt2img", "checkpoint.safetensors", payload(),
                                                workflow=_custom_workflow()),
            "external workflow": compiler.map_external_workflow(
                _custom_workflow(), "txt2img", "checkpoint.safetensors",
                {"prompt": f"portrait, {tag}", "negative_prompt": f"bad, {tag}", "seed": 3}),
            "postprocess": compiler.compile_postprocess("checkpoint.safetensors", payload(),
                                                        uploaded_image="in.png"),
            "anima": anima.compile("txt2img", V2_MODEL, payload(forge_additional_modules=_modules())),
        }

    def test_single_and_no_weight_tags_compile_exactly_as_before(self):
        for tag in self.TAGS:
            graphs = self._graphs(tag)
            with mock.patch.object(compiler_module, "_strip_extra_networks", _old_strip), \
                    mock.patch.object(compiler_module, "parse_lora_tags", _old_parse_lora_tags):
                before = self._graphs(tag)
            for label, graph in graphs.items():
                with self.subTest(tag=tag, graph=label):
                    self.assertEqual(json.dumps(graph, sort_keys=True), json.dumps(before[label], sort_keys=True))
                    self.assertTrue(any(node["class_type"] in _LORA_CLASSES for node in graph.values()))


if __name__ == "__main__":
    unittest.main()
