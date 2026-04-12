# core/ollama_client.py
"""Ollama REST API 클라이언트 — 로컬 LLM 프롬프트 강화"""
import requests
import json


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
}


class OllamaClient:
    """Ollama REST API 래퍼"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = 60

    def enhance(self, tags: str, mode: str = 'expand', extra_prompt: str = '') -> str:
        """태그를 LLM으로 강화하여 반환"""
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS['expand'])
        user_msg = tags
        if extra_prompt:
            user_msg = f"{extra_prompt}\n\nCurrent tags: {tags}" if tags else extra_prompt

        payload = {
            "model": self.model,
            "system": system,
            "prompt": user_msg,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 500},
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            response = data.get('response', '').strip()
            # 마크다운, 번호, 코드블록, 사고과정(<think>) 정리
            import re
            # <think>...</think> 블록 제거 (qwen3 등 thinking 모드)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
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

    def list_models(self) -> list:
        """사용 가능한 모델 목록 반환"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
