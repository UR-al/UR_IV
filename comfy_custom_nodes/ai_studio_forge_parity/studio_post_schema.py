"""The Hires, upscale and SAM3 settings of the Anima studio panel: one table for the state and the panel.

Pure data. ``studio_state.py`` adds these keys to its ``hires`` / ``upscale`` / ``sam3`` sections and clamps them;
``tools/build_studio_post_schema.py`` writes the table as ``web/ai_studio/post_schema.js`` for the panel.

A field has the shape of ``ext_schema.json`` (``key``, ``label``, ``kind``, ``default``, ``min``/``max``, ``choices``, ``input``,
``section``) plus ``step`` (the number box), ``list`` (the ``/choices`` kind that fills a file or sampler pick), ``empty`` (what
"" means there), ``title`` (the mouse-over text), ``multiline``, ``redraw`` (a change redraws the panel) and, in ``sam3``,
``node``: which SAM3 node takes ``input``.

A key without ``input`` is written by ``studio_logic.js`` itself, because its value depends on the main settings.
"""
from __future__ import annotations

SAME = "메인과 같음"
SAM3_MODES = ("Inpaint", "Mask only")
MASK_MODES = ("Individual", "Combined")
FILL_MODES = ("fill", "original", "latent noise", "latent nothing")
SAM3_DEVICES = ("auto", "cuda", "cpu")
# Easy SAM3 keeps fp32 weights and uses this only for autocast; Forge runs it under bf16 on CUDA.
SAM3_PRECISIONS = ("bf16", "fp16", "fp32")
CN_MODULES = (
    "inpaint_only", "none", "tile_resample", "depth_midas", "depth_zoe", "depth_anything", "openpose", "openpose_full",
    "openpose_hand", "lineart_realistic", "lineart_anime", "lineart_coarse", "canny", "softedge_hed", "scribble_pidinet",
)
CN_CONTROL_MODES = ("Balanced", "My prompt is more important", "ControlNet is more important")
UPSCALE_MODES = ("Scale by", "Scale to")


def _flag(key, label, default, **extra):
    return {"key": key, "label": label, "kind": "bool", "default": default, **extra}


def _int(key, label, default, low, high, **extra):
    return {"key": key, "label": label, "kind": "int", "default": default, "min": low, "max": high, **extra}


def _float(key, label, default, low, high, **extra):
    return {"key": key, "label": label, "kind": "float", "default": default, "min": low, "max": high, **extra}


def _text(key, label, **extra):
    return {"key": key, "label": label, "kind": "text", "default": "", **extra}


def _choice(key, label, choices, **extra):
    return {"key": key, "label": label, "kind": "choice", "default": choices[0], "choices": list(choices), **extra}


def _pick(key, label, kind, empty, **extra):
    """A pick from a ``/choices`` list; "" = ``empty``."""
    return {"key": key, "label": label, "kind": "choice", "default": "", "choices": [], "list": kind, "empty": empty, **extra}


def _in(section, *fields):
    return [{**field, "section": section} for field in fields]


def _mask(key, *rest, make=_int, **extra):
    return make(key, *rest, input=key, node="mask", **extra)


def _detailer(key, *rest, make=_int, **extra):
    return make(key, *rest, input=key, node="detailer", **extra)


POST_SCHEMA: dict[str, list[dict]] = {
    "hires": [
        _pick("scheduler", "스케줄러", "schedulers", SAME),
        _float("cfg", "CFG", 0.0, 0.0, 30.0, step=0.5, title="0 = 메인과 같음"),
        _float("shift", "Shift", 0.0, 0.0, 100.0, step=0.5,
               title="0 = 메인과 같음. flow_cosmos_dynamic·rho7 스케줄은 Shift 를 읽지 않음"),
        _int("resize_width", "폭", 0, 0, 16384, step=8, input="resize_width", title="폭·높이 0 = 배율 사용. 한쪽만 넣으면 비율대로"),
        _int("resize_height", "높이", 0, 0, 16384, step=8, input="resize_height", title="폭·높이 0 = 배율 사용. 한쪽만 넣으면 비율대로"),
        _int("size_step", "크기 단위", 64, 8, 256, step=8, input="size_step", title="Forge Resolution Step: 결과 크기를 이 배수로 맞춤"),
        _int("tile_size", "타일", 512, 0, 4096, step=16, input="tile_size", title="업스케일 모델 타일 크기. 0 = 나누지 않음"),
        _int("tile_overlap", "겹침", 32, 0, 512, step=4, input="tile_overlap", title="작으면 이음매가 보임"),
        _pick("vae", "VAE", "vaes", SAME),
        _text("prompt", "프롬프트", multiline=True, input="positive_text",
              title="비우면 메인 프롬프트. LoRA 태그와 [a:b:0.5] 는 읽지 않음"),
        _text("negative", "네거티브", multiline=True, input="negative_text",
              title="비우면 메인 네거티브. LoRA 태그와 [a:b:0.5] 는 읽지 않음"),
    ],
    "upscale": [
        _choice("mode", "방식", UPSCALE_MODES, input="resize_mode", redraw=True),
        _int("max_side", "최대 변", 0, 0, 8192, step=8, input="max_side_length", title="0 = 제한 없음. 긴 변이 넘으면 줄임"),
        _int("width", "폭", 1024, 64, 8192, step=8, input="width"),
        _int("height", "높이", 1024, 64, 8192, step=8, input="height"),
        _flag("crop", "잘라 맞추기", True, input="crop", title="가운데를 잘라 정확히 이 크기로. 끄면 비율 유지"),
        _pick("model_2", "두 번째 모델", "upscalers", "없음"),
        _float("visibility", "두 번째 비중", 0.0, 0.0, 1.0, step=0.05, input="upscaler_2_visibility"),
        _flag("color_correction", "색 보정", False, input="color_correction", title="결과의 색을 원본에 맞춤"),
        _int("tile_size", "타일", 512, 0, 4096, step=16, input="tile_size", title="0 = 나누지 않음"),
        _int("tile_overlap", "겹침", 32, 0, 512, step=4, input="tile_overlap", title="작으면 이음매가 보임"),
    ],
    "sam3": [
        *_in(
            "찾기",
            _choice("mode", "동작", SAM3_MODES, title="Mask only: 마스크만 찾아 저장하고 그림은 그대로"),
            _mask("exclude_prompt", "뺄 것", make=_text, title="찾은 영역에서 뺄 대상. 쉼표(,)로 여러 개"),
            _mask("threshold", "임계값", 0.4, 0.0, 1.0, make=_float, step=0.01),
            _mask("detection_limit", "최대 개수", -1, -1, 1000, title="-1 = 제한 없음"),
            _choice("mask_mode", "마스크 처리", MASK_MODES, input="mask_mode", node="both",
                    title="Individual: '/' 로 나눈 묶음마다 따로 그림. Combined: 한 번에"),
        ),
        *_in(
            "마스크",
            _mask("mask_dilation", "확장", 0, 0, 512),
            _mask("convex_hull", "볼록 껍질", False, make=_flag, title="머리카락 사이까지 감싸기"),
            _mask("mask_outline_px", "윤곽 확장", 0, 0, 512, title="가장자리를 따라 넓히기"),
            _mask("mask_blur", "흐림", 4, 0, 256),
            _mask("invert", "반전", False, make=_flag, title="마스크 바깥을 그리기"),
        ),
        *_in(
            "얼굴",
            _mask("face_crop", "얼굴 안에서만", False, make=_flag, title="얼굴을 먼저 찾고 그 안에서만 SAM3 를 돌림"),
            _pick("face_crop_model", "얼굴 모델", "face_models", "[얼굴 모델]", input="face_crop_model", node="mask"),
            _mask("face_crop_confidence", "얼굴 신뢰도", 0.3, 0.0, 1.0, make=_float, step=0.01),
        ),
        *_in(
            "인페인트",
            _detailer("inpaint_prompt", "프롬프트", make=_text, multiline=True,
                      title="비우면 메인 프롬프트. LoRA 태그와 [a:b:0.5] 는 읽지 않음"),
            _detailer("negative_prompt", "네거티브", make=_text, multiline=True,
                      title="비우면 메인 네거티브. LoRA 태그와 [a:b:0.5] 는 읽지 않음"),
            _choice("fill_mode", "마스크 내용", ("original", *[mode for mode in FILL_MODES if mode != "original"]),
                    input="fill_mode", node="detailer"),
            _detailer("only_masked", "마스크 영역만", True, make=_flag, title="마스크 둘레만 잘라 크게 그려서 붙임"),
            _detailer("mask_padding", "패딩", 32, 0, 2048),
            _detailer("grow_mask_by", "샘플링 마스크 확장", 6, 0, 64, title="Forge 에는 없는 값. 0 이 가장 가까움"),
            _detailer("use_custom_size", "크기 지정", False, make=_flag, title="끄면 마스크 영역을 생성 크기로 그림"),
            _detailer("custom_width", "폭", 512, 64, 8192, step=8),
            _detailer("custom_height", "높이", 512, 64, 8192, step=8),
        ),
        *_in(
            "샘플링",
            _int("steps", "스텝", 0, 0, 150, title="0 = 메인과 같음. 적은 만큼 그대로 돈다 (Forge 는 스텝 × 디노이즈)"),
            _float("cfg", "CFG", 0.0, 0.0, 30.0, step=0.5, title="0 = 메인과 같음"),
            _pick("sampler", "샘플러", "samplers", SAME),
            _pick("scheduler", "스케줄러", "schedulers", SAME),
            _int("seed", "시드", -1, -1, 2 ** 53 - 1, title="-1 = 메인과 같음"),
            _detailer("noise_multiplier", "노이즈 배율", 1.0, 0.0, 2.0, make=_float, step=0.01),
        ),
        *_in(
            "ControlNet",
            _flag("cn_enable", "켜기", False, input="controlnet_enable", node="detailer"),
            _pick("cn_model", "모델", "controlnets", "[ControlNet 모델]", input="controlnet_model_name", node="detailer"),
            _choice("cn_module", "전처리기", CN_MODULES, input="controlnet_module", node="detailer",
                    title="LLLite 타일·인페인트 파일은 전처리기를 쓰지 않음"),
            _float("cn_strength", "강도", 1.0, 0.0, 10.0, step=0.05, input="controlnet_strength", node="detailer"),
            _float("cn_start", "시작", 0.0, 0.0, 1.0, step=0.01),
            _float("cn_end", "끝", 1.0, 0.0, 1.0, step=0.01),
            _flag("cn_pixel_perfect", "Pixel Perfect", True),
            _choice("cn_control_mode", "제어 방식", CN_CONTROL_MODES),
            _int("cn_processor_res", "전처리 해상도", 512, 0, 4096, step=8, input="controlnet_processor_resolution", node="detailer"),
            _float("cn_threshold_a", "임계값 A", -1.0, -1.0, 256.0, title="-1 = 쓰지 않음"),
            _float("cn_threshold_b", "임계값 B", -1.0, -1.0, 256.0, title="-1 = 쓰지 않음"),
        ),
        *_in(
            "실행",
            _choice("device", "장치", SAM3_DEVICES, input="device", node="mask"),
            _choice("precision", "정밀도", SAM3_PRECISIONS, title="Forge 는 bf16. CPU 는 fp32 만 됨"),
            _mask("unload_after", "끝나면 VRAM 비우기", True, make=_flag),
            _flag("keep_in_ram", "RAM 에 남기기", True, title="다음 실행이 빨라짐. 끄면 메모리에서 내림"),
            _mask("save_artifacts", "마스크 저장", True, make=_flag, title="output/sam3 에 마스크와 오버레이를 저장"),
            _flag("overlay", "오버레이로 보기", False, title="Mask only 일 때 결과 대신 오버레이를 보임"),
        ),
    ],
}


def post_defaults(section: str) -> dict:
    return {field["key"]: field["default"] for field in POST_SCHEMA[section]}
