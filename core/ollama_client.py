# core/ollama_client.py
"""Ollama REST API 클라이언트 — 로컬 LLM 프롬프트 강화"""
import requests
import json
import re
import threading

#: Ollama 서버 기본 주소 — 백엔드의 모든 폴백이 이 하나를 쓴다(프론트는 utils/ollamaPrefs.ts).
DEFAULT_OLLAMA_URL = "http://localhost:11434"
#: 설치 목록을 받지 못했고 요청 모델도 비었을 때만 쓰는 최후 기본값.
#: Settings 의 추천 모델(SettingsView 'best')과 같아야 'ollama pull' 안내와 오류 메시지가 맞는다.
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"


SYSTEM_PROMPTS = {
    'expand': (
        "You are a Danbooru tag expert for Stable Diffusion image generation. "
        "The user will give you comma-separated tags. Expand them by adding complementary, "
        "high-quality Danbooru-style tags that would improve the image. "
        "Keep the original tags and add 10-20 new relevant tags. "
        "Use underscores for multi-word tags (e.g., blue_hair, school_uniform). "
        "Output ONLY comma-separated tags. No explanations, no numbering, no markdown."
    ),
    'nl2tags': (
        "You are a Danbooru tag expert for Stable Diffusion image generation. "
        "The user will describe an image in natural language. "
        "Convert it into high-quality Danbooru-style comma-separated tags. "
        "Include character count (1girl, 2boys etc), appearance, clothing, pose, "
        "expression, background, composition, and quality tags. "
        "Use underscores for multi-word tags. "
        "Output ONLY comma-separated tags. No explanations, no numbering, no markdown."
    ),
    'suggest': (
        "You are a Danbooru tag expert for Stable Diffusion image generation. "
        "The user will give you comma-separated tags. Suggest alternative tags that "
        "would create a similar but different image. Replace some tags with creative alternatives. "
        "Keep the same general concept but vary the details. "
        "Output ONLY comma-separated tags. No explanations, no numbering, no markdown."
    ),
    'negative': (
        "You are a Danbooru tag expert for Stable Diffusion negative prompts. "
        "The user will give you POSITIVE prompt tags. Generate appropriate NEGATIVE tags "
        "to prevent artifacts and unwanted elements for this specific scene. "
        "Rules: "
        "- If '1girl' or 'solo', add 'multiple girls, multiple boys, crowd, group' "
        "- If character has specific hair/eye color, add wrong colors to negative "
        "- Always include: worst quality, low quality, bad anatomy, bad hands, "
        "missing fingers, extra digits, fewer digits, blurry, watermark, signature "
        "- For NSFW-free prompts, add 'nsfw, nude' "
        "- For outdoor scenes, add 'indoor, room' and vice versa "
        "- Tailor negatives specifically to the content described "
        "- Output 15-30 negative tags "
        "Output ONLY comma-separated tags. No explanations, no numbering, no markdown."
    ),
    # ── 자연어(prose) 출력 모드 ──
    'nl_caption': (
        "You rewrite Danbooru/booru tags as one flowing English image caption for "
        "natural-language text-to-image models (Flux, SD3, NAI). "
        "Your reply is fed DIRECTLY into the image generator, so it must be the caption and "
        "NOTHING else — no preface, no reasoning, no labels, no notes about what you are doing. "
        "Write the caption in this ORDER (these step names are guidance only — NEVER print them): "
        "Step 1 (opening): name the main character and their series, then ALWAYS state their hair "
        "color and eye color when those tags are present (omit one only if its tag is absent or it "
        "is clearly not visible in the framing), plus one more defining trait if useful "
        "(e.g. 'Hatsune Miku from Vocaloid a singer with long aqua twintails and teal eyes'). "
        "Step 2 (main focus, the longest part): describe the main action first, then the body, "
        "pose, clothing and key props in vivid detail — this carries the image's energy. "
        "Step 3 (closing, only if needed): if other characters or extra notes exist, describe them "
        "briefly to round out the scene; otherwise stop. "
        "Hard rules: "
        "(a) Refer to EVERY character BY NAME, never bare pronouns like he/she/his/her (the model "
        "confuses who is who). "
        "(b) Base everything ONLY on the given tags — never invent characters, objects or actions "
        "that are not in the tags. If an extra hint is given, make it the focus. "
        "(c) Use NO commas — split ideas into separate sentences. Start with a capital letter, end "
        "with a period, write 2 or more sentences. "
        "(d) NEVER print step names, section labels, 'P1'/'P2', 'Hook'/'Main'/'Secondary', any word "
        "followed by a colon, any '*' or bullet, any list, drafts, or any remark about the task or "
        "input. Never begin with 'I', 'Let me', 'Since', 'First', 'Okay', 'Here', 'Revised'. "
        "Output the final caption sentences ONLY, on a single line, then stop. "
        "Example input: 1girl, hatsune miku, vocaloid, aqua twintails, detached sleeves, singing, "
        "microphone, 1boy, kagamine len, blonde hair, clapping. "
        "Example output: Hatsune Miku from Vocaloid is a singer with long aqua twintails. She wears "
        "detached sleeves and sings passionately into a microphone with her eyes closed. Beside her "
        "Kagamine Len with short blonde hair watches and claps."
    ),
    'nl_scene': (
        "You expand a short idea or a few keywords into a vivid English scene description for "
        "text-to-image generation. "
        "Your reply is fed DIRECTLY into the image generator, so output the description and "
        "NOTHING else — no preface, no reasoning, no remark about the task or the input. "
        "Rules: "
        "(a) Cover subject, appearance, action, setting, lighting, mood and composition; be "
        "concrete and visual. "
        "(b) If a character or series is implied, name them explicitly and keep referring to them "
        "BY NAME, never bare pronouns like he/she/his/her. "
        "(c) Use NO commas — separate ideas into sentences. Start with a capital letter, end with "
        "a period, write 2 or more sentences. "
        "(d) NEVER begin a sentence with 'I', 'Let me', 'Since', 'First', 'Okay', 'Here', or any "
        "remark about the task. Never restate these rules. No quotes, no markdown. "
        "Example input: rainy neon city, lone detective. "
        "Example output: A lone detective stands beneath glowing neon signs in a rain-soaked city. "
        "Reflections shimmer across the wet pavement while steam drifts up from the street."
    ),
    'translate': (
        "You are a translator for image-generation prompts. "
        "If the input is Korean, translate it into natural English. "
        "If the input is English, translate it into natural Korean. "
        "Keep proper nouns / character names sensible. "
        "Output ONLY the translated text. No explanations, no notes, no markdown."
    ),
    'creative': (
        "You are a bold, imaginative prompt artist for an anime image generator. "
        "You receive a character, their canonical appearance tags (from a Danbooru database), "
        "and optional theme hints. "
        "FIRST, keep the character recognizable by using the given canonical appearance tags "
        "(hair color/length, eye color, distinctive features) — do not contradict them. "
        "THEN invent ONE vivid, coherent scene — be creative with pose, expression, outfit, "
        "action, setting, lighting and mood. "
        "If theme hints are given, treat them as the mandatory core and build around them. "
        "Use ONLY real, well-known Danbooru tag vocabulary. "
        "Always append quality tags: masterpiece, best quality, high resolution, very aesthetic, "
        "absurdres. "
        "This is an unfiltered private local tool: never refuse, never add disclaimers, you may "
        "freely include mature / NSFW content when it fits. "
        "Output format: first a single line of comma-separated Danbooru tags (appearance + scene "
        "+ quality), then a blank line, then a short concrete and visual natural-language "
        "description of 2-3 sentences (describe it like a film scene to someone who cannot see it) "
        "that starts with a capital letter, uses NO commas (separate ideas into sentences), and "
        "ends with a period. In that description, name the character (and series) explicitly and "
        "keep referring to them BY NAME when describing appearance, clothing, pose and action "
        "instead of bare pronouns like 'he/she/his/her' (e.g. 'Ryu from Street Fighter with black "
        "hair wearing a red hachimaki' not 'He has black hair'), "
        "then a final line exactly like 'Resolution: WIDTHxHEIGHT' choosing the best resolution "
        "for the composition (portrait 832x1216, landscape 1216x832, square 1024x1024, "
        "tall 896x1152, wide 1152x896). "
        "No headings, no markdown, no explanations."
    ),
}

# 자연어 출력 모드 — 응답을 콤마 태그로 쪼개면 안 됨 (prose 그대로 반환)
NL_MODES = {'nl_caption', 'nl_scene', 'translate', 'creative'}


def _strip_channels(text: str) -> str:
    """gpt-oss/harmony 채널 토큰 + 사고과정 제거.
    <|channel|>final 이후만 취하고, 모든 파이프 변형(<|x|>, <|x>, <x|>)과
    <think> 블록, 선두 채널 라벨을 제거한다. (예: <|channel>thought<channel|> 등)"""
    import re as _re
    if not text:
        return text
    t = text
    # 1) <think>...</think>
    t = _re.sub(r'<think>.*?</think>', '', t, flags=_re.DOTALL | _re.IGNORECASE)
    # 2) final 채널이 있으면 그 이후만 (파이프 변형 모두 허용)
    fm = list(_re.finditer(
        r'<\|?\s*channel\s*\|?>\s*final\b[^\n<]*?(?:<\|?\s*message\s*\|?>|<\s*channel\s*\|?>)?',
        t, flags=_re.IGNORECASE))
    if fm:
        t = t[fm[-1].end():]
    # 3) 잔여 채널/메시지/시작/끝 토큰 제거 (모든 파이프 변형)
    t = _re.sub(r'<\|[^<>]*?\|?>', '', t)   # <|channel|>, <|channel>, <|message>
    t = _re.sub(r'<[^<>]*?\|>', '', t)       # <channel|>
    t = _re.sub(r'<\|(?:start|end|return|message|channel|assistant|system|user)\b[^>]*>?', '', t, flags=_re.IGNORECASE)
    # 4) 선두 채널 라벨 잔여물 (analysis/thought/commentary/final/assistantfinal)
    t = _re.sub(r'^\s*(?:assistant)?\s*(?:analysis|thought|commentary|final)\b[\s:>-]*', '', t, flags=_re.IGNORECASE)
    result = t.strip()
    # 안전망: 과도한 제거로 비어버리면, 토큰만 제거한 보수적 버전으로 폴백 (빈 응답 오인 방지)
    if not result and text.strip():
        safe = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        safe = _re.sub(r'<\|[^<>]*?\|?>', '', safe)
        safe = _re.sub(r'<[^<>]*?\|>', '', safe)
        result = safe.strip()
    return result


_META_TAIL = None


def _enforce_nl_style(prose: str) -> str:
    """자연어 스타일 강제: 선두 잡문자 제거, 끝의 추론/메타 문장 제거, 콤마 제거(→공백),
    대문자 시작, 마침표 종료. (모델이 규칙을 어겨도 보정)"""
    import re as _re
    global _META_TAIL
    if _META_TAIL is None:
        _META_TAIL = _re.compile(
            r"^\s*(?:let'?s\b|let me\b|wait\b|hmm\b|actually\b|okay\b|ok\b|so\b|"
            r"i\s+(?:should|think|will|'?ll|need|guess|am)\b|"
            r"(?:re-?)?check(?:ing|ed)?\b|double-?check\b|re-?read\b|verify\b|"
            r"looks good\b|that(?:'s| is) (?:it|all|good|correct)\b|all good\b|"
            r"done\b|finally\b|final\b|here'?s\b|note\s*:|output\s*:|caption\s*:|"
            r"one (?:more|small)\b|perfect\b|great\b)",
            _re.IGNORECASE)
    s = (prose or '').strip().strip('"').strip()
    if not s:
        return s
    # 선두 잡문자 제거 (앞에 붙는 '. ' ', ' '- ' 등) → 글자로 시작
    s = _re.sub(r"^[\s.,;:!?\"'`\-–—•*]+", '', s)
    # 끝에서부터 추론/메타 문장 제거 ("Let's check." "Wait..." "Check." 등)
    parts = _re.split(r'(?<=[.!?])\s+', s)
    while len(parts) > 1 and _META_TAIL.match(parts[-1].strip()):
        parts.pop()
    joined = ' '.join(p for p in parts if p.strip()).strip()
    if joined:
        s = joined
    s = s.replace(',', ' ')                       # 콤마 제거 (자연어는 공백/문장 구분)
    s = _re.sub(r'\s+([.!?])', r'\1', s)          # 구두점 앞 공백 제거
    s = _re.sub(r'\s{2,}', ' ', s).strip()        # 다중 공백 정리
    if s:
        s = s[0].upper() + s[1:]                  # 대문자 시작
    if s and s[-1] not in '.!?':
        s += '.'                                  # 마침표 종료
    return s


def _format_creative(text: str) -> str:
    """창의 모드: 첫 태그 줄(콤마 유지) + 본문 prose(콤마 제거/대문자/마침표) + Resolution 줄 보존."""
    import re as _re
    m = _re.search(r'\n\s*Resolution:\s*\d{3,4}\s*[x×]\s*\d{3,4}\s*$', text, flags=_re.IGNORECASE)
    res_line = ''
    body = text
    if m:
        res_line = '\n' + text[m.start():].strip()
        body = text[:m.start()].rstrip()
    blocks = _re.split(r'\n\s*\n', body, maxsplit=1)
    if len(blocks) == 2:
        blocks[1] = _enforce_nl_style(blocks[1])   # prose 블록만 스타일 강제
        body = blocks[0].rstrip() + '\n\n' + blocks[1]
    return body + res_line


def _clean_creative_tags(text: str) -> str:
    """창의 모드 출력의 첫 태그 줄에서 존재하지 않는(가짜) 태그 제거 (NAIA 태그 DB 대조)."""
    try:
        import re as _re
        from core.tag_intelligence import get_tag_intelligence
        ti = get_tag_intelligence()
        blocks = _re.split(r'\n\s*\n', text, maxsplit=1)
        first = blocks[0].strip()
        if ',' in first and '\n' not in first:   # 단일 콤마 태그 줄로 보일 때만
            tags = [t.strip() for t in first.split(',') if t.strip()]
            kept, dropped = ti.filter_noise(tags, drop_unknown=True)
            if kept and dropped:
                blocks[0] = ', '.join(kept)
                return '\n\n'.join(blocks)
    except Exception:
        pass
    return text


_META_OPENER = None
_META_PHRASE = None


def _is_meta_sentence(s: str) -> bool:
    """이 문장이 캡션이 아니라 모델의 추론/체크리스트/규칙복창/단답인지 판별.
    (캡션 문장 'A man with...', 'The man wears...', 'No shoes are visible.'는 False)"""
    import re as _re
    global _META_OPENER, _META_PHRASE
    if _META_OPENER is None:
        # 문장 '시작'이 추론/메타 오프너인 경우
        _META_OPENER = _re.compile(
            r"(?i)^\s*\**\s*(?:"
            # 라벨/불릿(체크리스트·규칙복창) — 콜론 동반
            r"task\s*:|constraints?\b|appearance\s*:|clothing(?:\s+options)?\s*:|actions?\s*:|setting\s*:|"
            r"shot\s*type\s*:|input\s+tags?\b|series\s*:|character\s*/\s*series\b|"
            r"goal\s*:|subject\s*:|scene\s*:|scars?\s*:|hair\s+colou?rs?\s*:|draft\s*:|"
            r"refin(?:e|ed|ing)\s+the\s+scene\b|correction\b|"
            r"let'?s\b|let me\b|wait\b|hmm+\b|actually\b|alright\b|well[, ]|"
            r"first[, ]|firstly\b|now[, ]|next[, ]|then[, ]|so[, ]|but wait\b|hold on\b|oh[, ]|"
            r"i'(?:ll|ve|d|m)\b|i\s+(?:should|think|will|need|must|guess|am|can|could|have to|"
            r"focus|focused|use|describe|want|plan|intend|see|notice|assume|interpret|consider)\b|"
            r"revised\b|corrected\b|re-?read\b|double-?check\b|checking\b|let'?s check\b|"
            r"(?:the\s+)?(?:output|caption|note|final|answer|result|response|revision|draft)\s*:|"
            r"yes[.!?]|no[.!?]|none[.!?]|nope[.!?]|yep[.!?]|sure[.!?]|correct[.!?]|right[.!?]|"
            r"good[.!?]|okay[.!?]|ok[.!?]|perfect\b|great\b|done\b|looks good\b|all good\b|"
            r"that'?s (?:it|all|good|correct|right)\b"
            r")"
        )
        # 문장 '내부'에 체크리스트/규칙복창 문구가 있는 경우
        _META_PHRASE = _re.compile(
            r"(?i)(?:"
            r"\bis a pronoun\b|\bare pronouns\b|\bno commas?\b|\bno pronouns?\b|without commas?\b|"
            r"\bcapital (?:start|letter)\b|starts? with a capital\b|"
            r"isn'?t exactly a name\b|not exactly a name\b|no specific character\b|"
            r"identif(?:y|ies|ied) the character\b|i'?ll treat\b|i will treat\b|"
            # 캐릭터를 어떻게 지칭할지 설명하는 메타
            r"no name (?:in|was|is)\b|there (?:is|are|'?s) no name\b|without a name\b|"
            r"refer to (?:him|her|them|it|the (?:character|subject|man|woman|person)|this (?:character|man|woman|person)) as\b|"
            r"i(?:'?ll| will) (?:refer to|call|name|treat|describe|use)\b|i(?:'?m| am) going to\b|"
            r"\bas requested\b|focus(?:ed|ing|es)? on the (?:primary|main|key|central)\b|"
            r"(?:^|\s)(?:no\s+)?(?:commas?|pronouns?|capital)\s*\?"
            r")"
        )
    n = (s or '').strip()
    if not n:
        return True
    return bool(_META_OPENER.match(n)) or bool(_META_PHRASE.search(n))


def _strip_meta_sentences(t: str) -> str:
    """선두/후미의 메타(추론/체크리스트) 문장을 제거하고 가운데 캡션만 남김."""
    import re as _re
    t = (t or '').strip().strip('"').strip()
    sents = _re.split(r'(?<=[.!?])\s+', t)
    while len(sents) > 1 and _is_meta_sentence(sents[0]):
        sents.pop(0)
    while len(sents) > 1 and _is_meta_sentence(sents[-1]):
        sents.pop()
    out = ' '.join(s for s in sents if s.strip()).strip()
    return out or t


def _cut_trailing_meta(s: str) -> str:
    """캡션 뒤에 붙는 검증/재확인/태그체크 블록을 잘라낸다.
    캡션 문장에는 '*'가 없으므로 첫 별표(후속 '*Checking...*' 등의 시작)에서 컷.
    별표 없는 누출('Checking for commas', 'abs - included' 류)도 표지에서 컷."""
    import re as _re
    s = (s or '').strip()
    i = s.find('*')
    if i >= 25:
        s = s[:i]
    m = _re.search(
        r"(?i)(?:^|[.\s])(?:checking\b|let me double|double-?check|re-?read|"
        r"no commas?\s*\??|flowing english\b|character (?:name|consistency)|overall scene first|"
        r"only from tags|no intro)",
        s)
    if m and m.start() >= 25:
        s = s[:m.start()]
    # "abs - included" / "bara - implied" 류 태그 점검 리스트 시작
    m2 = _re.search(r"(?i)(?:^|[.\s])[\w'\-]{2,30}\s+-\s+(?:included|implied|mentioned|maybe|excluded|omitted|present)\b", s)
    if m2 and m2.start() >= 25:
        s = s[:m2.start()]
    return s.strip().strip('"').strip()


def _extract_final_nl(text: str) -> str:
    """추론형 모델이 사고과정/체크리스트/규칙복창/초안/자기수정을 함께 뱉을 때 최종 캡션만 추출.
    1) 'Revised draft:' / 'Final:' / 'caption:' 류 마커가 있으면 마지막 마커 뒤가 최종본 —
       단 그 뒤에 붙는 메타(별표 블록·체크·태그리스트)는 _cut_trailing_meta로 잘라낸다.
    2) 그 외엔 선두/후미 메타 문장 제거.
    깔끔한 응답은 마커도 메타도 없어 그대로 통과."""
    import re as _re
    if not text:
        return text
    t = text.strip()
    # 1) 최종본 도입 마커 — revised/final/corrected 뒤 단어 1개 허용('Revised draft:'),
    #    주변 '*' 허용('*Revised draft:*'), 콜론 필수. 마지막 마커 뒤를 최종본으로.
    marker = _re.compile(
        r"(?im)\**\s*\b(?:"
        r"revised(?:\s+\w+)?|final(?:\s+\w+)?|corrected(?:\s+\w+)?|"
        r"draft(?:\s+\w+)?|refin(?:ed|ing)(?:\s+the\s+scene)?|rewrite|"
        r"here(?:'s| is)\s+(?:the\s+|your\s+|a\s+)?(?:final\s+|revised\s+|corrected\s+|new\s+|updated\s+)?caption|"
        r"(?:the\s+)?caption|the\s+output|final\s+answer"
        r")\b\s*\**\s*:\s*\**\s*")
    revs = list(marker.finditer(t))
    if revs:
        tail = _cut_trailing_meta(t[revs[-1].end():])
        cleaned = _strip_meta_sentences(tail)
        if len(cleaned) >= 25:
            return cleaned
    # 2) 마커 없음/짧음 → 선두/후미 메타 문장 제거
    #    (전체에 _cut_trailing_meta(첫 '*' 컷)는 위험 — 캡션이 맨 앞에 없을 수 있음)
    return _strip_meta_sentences(t)


def summarize_model_info(data: dict) -> dict:
    """Bounded /api/show metadata; no templates, system prompts or filename guesses."""
    details = data.get('details') if isinstance(data.get('details'), dict) else {}
    metadata = data.get('model_info') if isinstance(data.get('model_info'), dict) else {}
    architecture = str(metadata.get('general.architecture') or details.get('family') or '')[:100]
    capabilities = data.get('capabilities')
    capabilities = [v for v in capabilities if v in {'completion', 'vision', 'thinking', 'tools', 'embedding'}] if isinstance(capabilities, list) else None
    def positive_int(key):
        value = metadata.get(f'{architecture}.{key}')
        return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10_000_000 else None
    experts = positive_int('expert_count')
    thinking = 'unknown' if capabilities is None else 'none' if 'thinking' not in capabilities else 'levels' if architecture == 'gptoss' else 'boolean'
    return {'architecture': architecture, 'parameterSize': str(details.get('parameter_size') or '')[:80],
            'quantization': str(details.get('quantization_level') or '')[:80],
            'contextLength': positive_int('context_length'), 'experts': experts,
            'activeExperts': positive_int('expert_used_count'),
            'moe': None if experts is None else experts > 1,
            'capabilities': capabilities, 'thinkingMode': thinking,
            'vision': None if capabilities is None else 'vision' in capabilities}


# ── 설치 모델 대조 (네트워크 없는 순수 로직) ──

def _split_model_tag(name: str) -> tuple[str, str]:
    """'repo:tag' → (repo, tag), 소문자. 태그가 없으면 Ollama 규칙대로 'latest'.

    레지스트리 포트('host:5000/model')의 콜론은 태그가 아니다 — 마지막 콜론 뒤에 '/' 가 있으면 무시.
    """
    text = (name or '').strip().lower()
    base, sep, tag = text.rpartition(':')
    if not sep or not base or not tag or '/' in tag:
        return text, 'latest'
    return base, tag


def resolve_model(requested: str, installed, default: str = DEFAULT_OLLAMA_MODEL) -> str:
    """요청 모델을 설치 목록에 맞춘다.

    우선순위: 1) 정확히 같은 이름 2) 같은 모델(``name`` ≡ ``name:latest``, 대소문자 무시)
    3) 같은 계열의 설치된 다른 태그(gemma3:4b 요청인데 gemma3:12b 만 있으면 gemma3:12b —
    예전엔 계열만 같으면 요청 이름을 그대로 보내 404 '빈 응답'이 났다) 4) 첫 설치 모델.
    설치 목록이 비면(서버 꺼짐·조회 실패) 판단할 근거가 없으므로 요청 그대로, 그것도 비면 ``default``.
    """
    want = (requested or '').strip()
    names = [m.strip() for m in (installed or []) if isinstance(m, str) and m.strip()]
    if not names:
        return want or default
    if want:
        if want in names:
            return want
        key = _split_model_tag(want)
        same = next((m for m in names if _split_model_tag(m) == key), None)
        if same:
            return same
        family = next((m for m in names if _split_model_tag(m)[0] == key[0]), None)
        if family:
            return family
    return names[0]


def resolve_installed_model(base_url: str, requested: str, default: str = DEFAULT_OLLAMA_MODEL) -> str:
    """``/api/tags`` 로 설치 목록을 받아 :func:`resolve_model` — HTTP 이므로 워커 스레드에서만 부른다."""
    installed = OllamaClient(base_url=base_url or DEFAULT_OLLAMA_URL).list_models()
    return resolve_model(requested, installed, default)


def unload_configured_model(base_url: str, model: str) -> bool:
    """설정(ui_prefs.ollamaModel)의 모델을 VRAM 에서 내린다 — HTTP 이므로 워커 스레드에서만 부른다.

    설정 이름이 설치되지 않은 태그일 수 있다(Settings 추천 카드는 pull 안내를 위해 이름만 저장 —
    gemma4:e4b 인데 gemma4:26b 만 설치). 태그 강화·NL 변환(OllamaWorker)과 Comic 은
    :func:`resolve_model` 로 실제 설치 모델을 올리므로, 같은 규칙으로 골라 내려야 VRAM 이 비워진다
    (설정 이름으로 keep_alive=0 을 보내면 404 로 아무 일도 없다).
    설치 목록이 비면(서버 꺼짐·모델 없음) 점유한 VRAM 도 없으므로 언로드할 것 없이 성공(True).
    """
    wanted = (model or '').strip()
    if not wanted:
        return True
    try:
        installed = OllamaClient(base_url=base_url or DEFAULT_OLLAMA_URL).list_models()
        if not installed:
            return True
        return bool(OllamaClient(base_url or DEFAULT_OLLAMA_URL, resolve_model(wanted, installed)).unload())
    except Exception:
        return False


# ── 추론(thinking) 모델 제어 ──
# Gemma 4·Qwen3 같은 추론형 모델은 기본으로 먼저 '생각'한다. 태그·자연어·캡션처럼 짧은 답에
# 그러면 num_predict 를 사고에 다 써 빈 답이 오거나, 사고 원문이 답 자리로 새어 태그에 섞인다.
# 그래서 /api/show 의 능력(thinkingMode)을 (서버, 모델)별로 한 번 읽어 think 를 맞춰 보낸다.
# think:False 를 무조건 보내면 일부 모델이 빈 응답을 낸다(d923c5c04 롤백) — 능력이 확인된
# 모델에만 보내고, 거부(400)·빈 답이면 think 없이 **호출당 한 번만** 다시 보낸다.
# 그 모델에 think 를 영영 안 보내는 건(THINK_REJECTED) 근거가 확실할 때뿐이다:
#   - 400 'does not support thinking' (모델 능력이 없다는 명시적 거부)
#   - 'think 로는 빈 답, 빼면 답' 이 연속 THINK_EMPTY_STRIKE_LIMIT 번 (한 번은 샘플링 우연일 수 있다)
# 캐시는 Settings/캡션/대화에서 모델 목록을 다시 읽을 때(= 재연결·pull 뒤) 그 서버 몫이 비워진다.

_THINK_MODES: dict[tuple[str, str], str] = {}
_THINK_MODES_LOCK = threading.Lock()
#: think 를 보냈다가 확실히 거부된 모델 — 이후엔 필드를 생략한다(예전 동작).
THINK_REJECTED = 'rejected'
#: 'think 로 빈 답 → think 없이 답' 이 이만큼 연속되면 그 모델을 THINK_REJECTED 로 기억한다.
THINK_EMPTY_STRIKE_LIMIT = 2
_THINK_EMPTY_STRIKES: dict[tuple[str, str], int] = {}
#: 모델에 추론 능력이 없다는 Ollama 의 명시적 거부 — 모델의 성질로 기억해도 되는 유일한 400.
#: 그 밖의 400(think 값·format 조합 검증 등)은 요청 모양 탓일 수 있어 다른 요청까지 끄지 않는다.
_THINK_UNSUPPORTED_RE = re.compile(r'(?:"[^"\r\n]+"|\S+) does not support thinking\.?', re.IGNORECASE)


def _error_detail(response) -> str:
    """오류 응답의 ``error`` 문구(JSON 이 아니면 본문 앞부분)."""
    try:
        data = response.json()
        if isinstance(data, dict):
            return str(data.get('error', '') or '')
    except Exception:
        pass
    try:
        return str(getattr(response, 'text', '') or '')[:200]
    except Exception:
        return ''


def is_think_unsupported_error(detail: str) -> bool:
    """``detail`` 이 '<모델> does not support thinking' 인가(Ollama 의 명시적 능력 거부)."""
    return bool(_THINK_UNSUPPORTED_RE.fullmatch((detail or '').strip()))


class _ThinkPlan:
    """한 번의 호출(enhance 의 폴백 단계들, 캡션·Comic 의 단일 요청) 동안의 think 결정.

    ``value``      — 이번 단계에 보낼 think(None 이면 생략). think 탓인 400 이 나면 남은 단계는 생략.
    ``retry_left`` — think 없이 다시 보내는 재시도는 호출당 한 번 — 폴백 단계마다 요청이 두 배가 되지 않게.
    '200 빈 답'은 think 를 뺀 재시도도 비면 think 탓이 아니므로 남은 단계에도 think 를 계속 보낸다.
    """
    __slots__ = ('value', 'retry_left')

    def __init__(self, value):
        self.value = value
        self.retry_left = True


def think_value_for(mode: str):
    """thinkingMode → 요청의 think 값. ``None`` 이면 필드를 아예 보내지 않는다.

    boolean → False(바로 답하게), levels(gpt-oss: 끌 수 없음) → 'low',
    none/unknown/rejected → 생략(비추론 모델은 think 를 400 으로 거부할 수 있다).
    """
    if mode == 'boolean':
        return False
    if mode == 'levels':
        return 'low'
    return None


def clear_thinking_mode_cache(base_url: str | None = None) -> None:
    """능력·거부 기억 비우기 — 모델 목록을 다시 읽을 때(재연결·``ollama pull`` 로 능력이 바뀜)와 테스트 격리용.

    ``base_url`` 을 주면 그 서버 몫만 비운다(다른 서버의 기억은 그대로).
    """
    server = None if base_url is None else (base_url or DEFAULT_OLLAMA_URL).rstrip('/')
    with _THINK_MODES_LOCK:
        for cache in (_THINK_MODES, _THINK_EMPTY_STRIKES):
            if server is None:
                cache.clear()
            else:
                for key in [k for k in cache if k[0] == server]:
                    del cache[key]


def _chat_content(data) -> str:
    return (((data.get('message') or {}) if isinstance(data, dict) else {}).get('content') or '').strip()


def _generate_response(data) -> str:
    return ((data.get('response') if isinstance(data, dict) else '') or '').strip()


class OllamaClient:
    """Ollama REST API 래퍼"""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL):
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip('/')
        self.model = model
        self.timeout = 60

    def thinking_mode(self) -> str:
        """이 모델의 thinkingMode('boolean'|'levels'|'none'|'unknown'|'rejected') — (서버, 모델)별 캐시.

        조회 실패(서버 꺼짐 등)는 캐시하지 않는다 — 서버가 늦게 뜨면 다음 요청에서 다시 읽는다.
        """
        key = (self.base_url, str(self.model or ''))
        with _THINK_MODES_LOCK:
            cached = _THINK_MODES.get(key)
        if cached is not None:
            return cached
        try:
            mode = str(self.get_model_info().get('thinkingMode') or 'unknown')
        except Exception:
            return 'unknown'
        with _THINK_MODES_LOCK:
            return _THINK_MODES.setdefault(key, mode)

    def _think_key(self) -> tuple[str, str]:
        return (self.base_url, str(self.model or ''))

    def _mark_think_rejected(self) -> None:
        key = self._think_key()
        with _THINK_MODES_LOCK:
            _THINK_MODES[key] = THINK_REJECTED
            _THINK_EMPTY_STRIKES.pop(key, None)

    def _note_think_answered(self) -> None:
        """think 를 붙인 요청이 답을 냈다 — '빈 답' 연속 횟수를 끊는다."""
        key = self._think_key()
        with _THINK_MODES_LOCK:
            _THINK_EMPTY_STRIKES.pop(key, None)

    def _note_think_empty_strike(self) -> None:
        """'think 로 빈 답 → think 없이 답' 한 번. 연속 THINK_EMPTY_STRIKE_LIMIT 번이면 거부로 기억한다."""
        key = self._think_key()
        with _THINK_MODES_LOCK:
            strikes = _THINK_EMPTY_STRIKES.get(key, 0) + 1
            if strikes >= THINK_EMPTY_STRIKE_LIMIT:
                _THINK_MODES[key] = THINK_REJECTED
                _THINK_EMPTY_STRIKES.pop(key, None)
            else:
                _THINK_EMPTY_STRIKES[key] = strikes

    def _post_json(self, endpoint: str, body: dict, *, plan: _ThinkPlan, answer_of, timeout):
        """``body`` 를 POST 하고 JSON 을 돌려준다. ``plan`` (:class:`_ThinkPlan`) 을 이 호출 동안 갱신한다.

        ``plan.value`` 가 있으면 think 를 붙여 먼저 보낸다.
        - 400: think 없이 다시 보낸다. 재시도가 400 이 아니면 think 탓이므로 이 호출의 남은 단계는
          think 를 생략한다. '<모델> does not support thinking' 이면 모델 성질이라 영구히 기억한다.
        - 200 인데 ``answer_of(data)`` 가 빔: 호출당 한 번만 think 없이 다시 보낸다. 재시도가 답하면
          '빈 답 1회'로 세고(연속되면 영구 기억), 재시도도 비면 think 탓이 아니므로 남은 단계에도
          think 를 계속 보낸다(재시도는 이미 썼으니 더 늘지 않는다).
        상태 코드가 없는 응답 객체(테스트 대역 등)는 200 으로 본다.
        """
        url = f"{self.base_url}{endpoint}"
        think = plan.value
        if think is None:
            response = requests.post(url, json=body, timeout=timeout)
            response.raise_for_status()
            return response.json()

        first = requests.post(url, json={**body, 'think': think}, timeout=timeout)
        if getattr(first, 'status_code', 200) == 400:
            explicit = is_think_unsupported_error(_error_detail(first))
            if explicit:
                # 모델에 추론 능력이 없다 — 남은 단계도, 앞으로의 요청도 think 없이
                plan.value = None
                self._mark_think_rejected()
            elif not plan.retry_left:
                # 재시도를 이미 썼고 think 탓이라는 근거도 없다 — 이 단계는 실패로 둔다
                first.raise_for_status()
            plan.retry_left = False
            retry = requests.post(url, json=body, timeout=timeout)
            if getattr(retry, 'status_code', 200) != 400:
                # think 를 빼자 400 이 사라졌다 → think 탓. 이 호출의 남은 단계는 생략한다
                # (명시적 거부가 아니면 기억하지는 않는다 — format 등 요청 조합 탓일 수 있다)
                plan.value = None
            retry.raise_for_status()
            return retry.json()

        first.raise_for_status()
        data = first.json()
        if answer_of(data):
            self._note_think_answered()
            return data
        if not plan.retry_left:
            return data
        plan.retry_left = False
        retry = requests.post(url, json=body, timeout=timeout)
        retry.raise_for_status()
        retry_data = retry.json()
        if answer_of(retry_data):
            # 한 번의 빈 답은 샘플링 우연일 수 있다 — 연속될 때만 이 모델을 '거부'로 기억한다
            self._note_think_empty_strike()
        return retry_data

    def enhance(self, tags: str, mode: str = 'expand', extra_prompt: str = '', *,
                instructions=None, instruction_feature=None) -> str:
        """태그를 LLM으로 강화하여 반환"""
        from core.ai_assist_instructions import compose_system_prompt
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS['expand'])
        system = compose_system_prompt(system, mode, instructions, feature=instruction_feature)
        # 태그→자연어: LLM에 '묘사할 내용'만 — 네거티브/선행·후행 고정(품질·score·year)/
        # <lora>/@트리거 제거하고 인물수·캐릭터·작품·메인 태그만 전송 (정확도↑, 소형 모델 혼란↓).
        if mode == 'nl_caption':
            from core.prompt_for_nl import clean_tags_for_nl
            tags = clean_tags_for_nl(tags)
        user_msg = tags
        if extra_prompt:
            user_msg = f"{extra_prompt}\n\nCurrent tags: {tags}" if tags else extra_prompt

        is_nl = mode in NL_MODES
        opts = {
            "temperature": 0.8 if is_nl else 0.7,
            "num_predict": 1024 if is_nl else 500,
        }

        import json as _json
        self._last_raw = ''
        # 추론형 모델은 think 로 사고를 끈다(모델 능력 확인 후). 세 폴백 단계가 한 계획(_ThinkPlan)을
        # 공유한다: think 탓인 400 이면 남은 단계는 think 없이, '200 빈 답'이면 think 는 유지하되
        # think 없는 재시도는 호출당 한 번만 — 단계마다 요청이 두 배가 되지 않게.
        plan = _ThinkPlan(think_value_for(self.thinking_mode()))

        def _post(endpoint, body, answer_of):
            d = self._post_json(endpoint, body, plan=plan, answer_of=answer_of, timeout=self.timeout)
            self._last_raw = _json.dumps(d, ensure_ascii=False)[:300]
            return d

        def _chat(messages):
            d = _post("/api/chat", {"model": self.model, "messages": messages,
                                    "stream": False, "options": opts}, _chat_content)
            m = (d.get('message') or {}) if isinstance(d, dict) else {}
            content = (m.get('content') or '').strip()
            if content or not is_nl:
                # 태그 모드에서 사고 원문을 답으로 쓰면 콤마로 쪼개져 사고 문장이 태그로 섞인다
                return content
            # 자연어 모드만: 본문이 비면 사고에서 최종 캡션을 건진다(_extract_final_nl 이 사고를 걷어냄)
            return (m.get('thinking') or '').strip()

        def _gen():
            d = _post("/api/generate", {"model": self.model, "system": system, "prompt": user_msg,
                                        "stream": False, "options": opts}, _generate_response)
            return _generate_response(d)

        def _attempt(fn):
            try:
                return fn()
            except (requests.ConnectionError, requests.Timeout):
                raise
            except Exception:
                return ''

        try:
            # 여러 방식 시도 — HF GGUF 등 채팅 템플릿 호환 편차 대응:
            # 1) chat(system+user)  2) chat(system을 user에 합침, system role 미지원 대응)
            # 3) generate(system+prompt).  message.content가 비면 thinking도 확인.
            response = (
                _attempt(lambda: _chat([{"role": "system", "content": system},
                                        {"role": "user", "content": user_msg}]))
                or _attempt(lambda: _chat([{"role": "user", "content": f"{system}\n\n{user_msg}"}]))
                or _attempt(_gen)
            )
            if not response:
                raise RuntimeError(
                    f"AI가 빈 응답을 반환했습니다 — 모델 '{self.model}'의 응답 형식 문제일 수 있습니다. "
                    f"(raw: {self._last_raw[:160]})")
            import re
            # harmony/channel 토큰(gpt-oss 등) + <think> 제거 (모든 파이프 변형)
            response = _strip_channels(response)
            # 자연어 모드: 콤마-태그 정리 없이 prose 그대로 (코드펜스는 마커만 제거, 내용 보존)
            if is_nl:
                clean_nl = re.sub(r'^```[a-zA-Z0-9]*\s*', '', response).strip()
                clean_nl = re.sub(r'\s*```\s*$', '', clean_nl).strip().strip('"').strip()
                if not clean_nl:
                    raise RuntimeError("AI가 빈 응답을 반환했습니다 (모델 채팅 템플릿 확인 필요)")
                if mode == 'creative':
                    clean_nl = _clean_creative_tags(clean_nl)
                    clean_nl = _format_creative(clean_nl)   # 본문 prose 스타일 강제(콤마X/대문자/마침표)
                elif mode in ('nl_caption', 'nl_scene'):
                    clean_nl = _extract_final_nl(clean_nl)   # 추론형 모델: 사고과정 제거하고 최종 캡션만
                    clean_nl = _enforce_nl_style(clean_nl)   # 순수 자연어: 콤마X/대문자/마침표
                return clean_nl
            # 코드블록 제거
            response = re.sub(r'```[^`]*```', '', response, flags=re.DOTALL).strip()
            # 번호 매기기 제거 (1. tag, 2. tag)
            response = re.sub(r'^\d+[\.\)]\s*', '', response, flags=re.MULTILINE)
            # 줄바꿈 → 콤마
            lines = response.replace('\n', ', ').split(',')
            clean = [t.strip().strip('-').strip('*').strip('"').strip("'").strip()
                     for t in lines if t.strip()]
            # 빈 결과 검증
            if not clean:
                raise RuntimeError("AI가 유효한 태그를 반환하지 않았습니다")
            return ', '.join(clean)
        except requests.ConnectionError:
            raise ConnectionError("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
        except requests.Timeout:
            raise TimeoutError("Ollama 응답 시간 초과 (60초)")
        except Exception as e:
            raise RuntimeError(f"Ollama 오류: {e}")

    def caption_image(
        self,
        image_path: str,
        prompt: str = '',
        timeout: int = 180,
        system_prompt: str | None = None,
    ) -> str:
        """비전 모델(qwen2-vl 등)로 이미지 캡션 생성. self.model 이 비전 모델이어야 함."""
        import base64
        import re
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        default_system = (
            "You are an expert image-captioning engine. Look at the image and write ONE flowing "
            "English paragraph that vividly describes it, as if describing the scene to someone who "
            "cannot see it. "
            "Describe ONLY what is actually visible — never invent or add subjects, objects, "
            "clothing, actions or settings that are not present in the image. "
            "Cover the main subject(s) and their number, apparent gender, appearance and clothing, "
            "their pose, expression and action, then the setting, lighting and mood; state the "
            "overall scene first, then the details. Use concrete, specific words. "
            "If the user gives a hint or keyword, make it the priority focus and build the "
            "description around it. "
            "Output the caption only — one coherent paragraph, no tag list, no bullet points, no "
            "preface, no notes about what you are doing."
        )
        system = (system_prompt or '').strip() or default_system
        user_prompt = (prompt or '').strip() or "Describe this image in detail."
        payload = {
            "model": self.model,
            "system": system,
            "prompt": user_prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }
        # 추론형 비전 모델(Qwen3-VL 등)이 num_predict 512 를 사고에 쓰지 않게 — 능력 확인 후 think 제어
        plan = _ThinkPlan(think_value_for(self.thinking_mode()))
        try:
            data = self._post_json("/api/generate", payload, plan=plan,
                                   answer_of=_generate_response, timeout=timeout)
            text = _generate_response(data)
            text = _strip_channels(text)
            text = _extract_final_nl(text)   # 추론/체크리스트/리비전 누출 제거 (콤마는 보존)
            return text
        except requests.ConnectionError:
            raise ConnectionError("Ollama 서버에 연결할 수 없습니다.")
        except requests.Timeout:
            raise TimeoutError(f"캡션 응답 시간 초과 ({timeout}초)")
        except Exception as e:
            raise RuntimeError(f"캡션 오류: {e}")

    def complete_chat(self, messages, *, options=None, response_format=None, timeout=300) -> str:
        """비스트리밍 ``/api/chat`` 한 번 — 답 본문을 돌려준다(Comic Director 처럼 JSON 을 받는 곳).

        추론형 모델이 num_predict 를 사고에 다 써 본문이 비지 않도록 think 를 능력에 맞춰 끈다.
        HTTP 오류는 그대로 올린다(호출자가 사용자에게 보여 준다).
        """
        body = {"model": self.model, "messages": list(messages or []), "stream": False}
        if options:
            body["options"] = dict(options)
        if response_format is not None:
            body["format"] = response_format
        data = self._post_json("/api/chat", body, plan=_ThinkPlan(think_value_for(self.thinking_mode())),
                               answer_of=_chat_content, timeout=timeout)
        if not isinstance(data, dict):
            return ''
        return str((data.get("message") or {}).get("content") or data.get("response") or "")

    def chat_stream(self, messages, *, model=None, options=None, think=None, schema=None,
                    on_token=None, on_thinking=None, should_stop=None, timeout=600):
        """`/api/chat` 을 스트리밍으로. 조각마다 ``on_token(text)``, 끝나면 dict 를 돌려준다.

        ``think`` — thinking 모델(Gemma 4·Qwen3.x)은 기본으로 생각부터 하느라 첫 글자가 한참 뒤에
        온다. False 면 바로 답하게 하고, True 면 생각 조각을 ``on_thinking(text)`` 으로 따로 흘린다.
        비추론 모델이 False를 명시적으로 거부할 때만 플래그 없이 한 번 더 요청한다.

        ``should_stop()`` 이 참이 되면 응답을 닫고 그때까지의 본문을 돌려준다 — 서버 쪽
        생성도 연결이 끊기면 멈춘다(Ollama 는 클라이언트가 떠나면 요청을 버린다).
        메시지의 ``images`` 는 접두사 없는 base64 여야 한다(core/chat_store.strip_data_url).
        """
        payload = {"model": model or self.model, "messages": messages, "stream": True}
        if options:
            payload["options"] = dict(options)
        if schema is not None:
            from core.structured_output import parse_schema
            payload['format'] = parse_schema(schema)
            opts = payload.setdefault('options', {})
            if opts.get('num_predict', -1) <= 0:
                opts['num_predict'] = 4096
        if think is not None:
            if not isinstance(think, bool) and think not in ('low', 'medium', 'high', 'max'):
                raise ValueError('지원하지 않는 추론 설정입니다')
            payload["think"] = think
        pieces = []
        thoughts = []
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=(10, timeout))
        if resp.status_code == 400 and payload.get("think") is False:
            detail = ""
            try:
                detail = str(resp.json().get("error", ""))
            except Exception:
                detail = (resp.text or "")[:200]
            # Only a model explicitly lacking thinking can safely omit OFF.
            # Generic think/level validation failures must not silently restore
            # the model's default (which may enable thinking).
            if is_think_unsupported_error(detail):
                resp.close()
                payload.pop("think", None)
                resp = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=(10, timeout))
        with resp:
            if resp.status_code != 200:
                detail = ""
                try:
                    detail = str(resp.json().get("error", ""))
                except Exception:
                    detail = (resp.text or "")[:200]
                raise RuntimeError(f"Ollama {resp.status_code}: {detail or 'chat 요청 실패'}")
            for line in resp.iter_lines(decode_unicode=True):
                if should_stop is not None and should_stop():
                    resp.close()
                    return {"content": "".join(pieces), "thinking": "".join(thoughts), "stopped": True}
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(str(data["error"]))
                message = (data.get("message") or {}) if isinstance(data, dict) else {}
                thought = message.get("thinking") or ""
                if thought:
                    thoughts.append(thought)
                    if on_thinking is not None:
                        on_thinking(thought)
                piece = message.get("content") or ""
                if piece:
                    pieces.append(piece)
                    if on_token is not None:
                        on_token(piece)
                if isinstance(data, dict) and data.get("done"):
                    return {
                        "content": "".join(pieces), "thinking": "".join(thoughts), "stopped": False,
                        "eval_count": data.get("eval_count"),
                        "total_duration": data.get("total_duration"),
                        # 'length' 면 num_predict 에 걸려 잘린 것 — 화면에 그렇게 알린다
                        "done_reason": data.get("done_reason"),
                    }
        # A clean HTTP EOF is not an Ollama completion: proxies and stopped
        # servers can close the stream before its required done packet.
        if should_stop is not None and should_stop():
            return {"content": "".join(pieces), "thinking": "".join(thoughts), "stopped": True}
        raise RuntimeError("응답 완료 신호를 받기 전에 연결이 끊어졌습니다. 받은 내용은 유지됩니다. 다시 시도해 주세요.")

    def unload(self) -> bool:
        """모델을 VRAM에서 즉시 언로드 (keep_alive=0). best-effort."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=5,
            )
            return True
        except Exception:
            return False

    def list_models(self) -> list:
        """사용 가능한 모델 목록 반환"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

    def get_model_info(self) -> dict:
        """Read model architecture/capabilities without loading or generating."""
        response = requests.post(f'{self.base_url}/api/show',
                                 json={'model': self.model, 'verbose': False}, timeout=(3, 8))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('Ollama 모델 정보 응답 형식이 올바르지 않습니다')
        return summarize_model_info(data)

    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
