# core/ollama_client.py
"""Ollama REST API 클라이언트 — 로컬 LLM 프롬프트 강화"""
import requests
import json
import re


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


class OllamaClient:
    """Ollama REST API 래퍼"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = 60

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

        def _chat(messages):
            r = requests.post(f"{self.base_url}/api/chat",
                              json={"model": self.model, "messages": messages,
                                    "stream": False, "options": opts},
                              timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
            self._last_raw = _json.dumps(d, ensure_ascii=False)[:300]
            m = d.get('message') or {}
            return ((m.get('content') or '') or (m.get('thinking') or '')).strip()

        def _gen():
            r = requests.post(f"{self.base_url}/api/generate",
                              json={"model": self.model, "system": system, "prompt": user_msg,
                                    "stream": False, "options": opts},
                              timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
            self._last_raw = _json.dumps(d, ensure_ascii=False)[:300]
            return (d.get('response') or '').strip()

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
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            text = (r.json().get('response', '') or '').strip()
            text = _strip_channels(text)
            text = _extract_final_nl(text)   # 추론/체크리스트/리비전 누출 제거 (콤마는 보존)
            return text
        except requests.ConnectionError:
            raise ConnectionError("Ollama 서버에 연결할 수 없습니다.")
        except requests.Timeout:
            raise TimeoutError(f"캡션 응답 시간 초과 ({timeout}초)")
        except Exception as e:
            raise RuntimeError(f"캡션 오류: {e}")

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
            if re.fullmatch(r'(?:"[^"\r\n]+"|\S+) does not support thinking\.?', detail.strip(), re.IGNORECASE):
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
