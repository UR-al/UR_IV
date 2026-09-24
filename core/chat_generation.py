"""Explicit, cancellable media generation for chat; no Qt or language model needed."""
from __future__ import annotations

from dataclasses import dataclass
import re
import math
import base64
import copy
import os
from pathlib import Path
import threading
import uuid
import tempfile
from io import BytesIO


@dataclass(frozen=True)
class GenerationPlan:
    kind: str
    family: str
    prompt: str
    image: str = ''
    duration: float = 5.0
    denoise: float = 0.65


def plan_chat_generation(payload: dict) -> GenerationPlan | None:
    """Route only the latest user's actual request, never assistant claims."""
    options = payload.get('generation') or {}
    if not isinstance(options, dict):
        raise ValueError('생성 옵션 형식이 올바르지 않습니다')
    mode = options.get('mode', 'auto')
    latest = next((m for m in reversed(payload.get('messages') or [])
                   if isinstance(m, dict) and m.get('role') == 'user'), {})
    prompt = str(latest.get('content') or '').strip()
    if mode not in {'auto', 'chat', 'image', 'video'}:
        raise ValueError('지원하지 않는 채팅 모드입니다')
    if mode == 'chat' or not prompt:
        return None
    intent = re.sub(r'```[\s\S]*?```|`[^`]*`', '', prompt).strip()
    if mode == 'auto':
        # Explicit media selection is authoritative: its text describes the
        # scene (possibly a sign saying "do not enter"), not a routing command.
        if re.search(r'프롬프트|태그|방법|어떻게|설명|알려|번역|요약|하지\s*마|하지말|말고|말아|않[아는]|말해|'
                     r'안\s*(?:그려|만들|생성)|\b(?:prompt|explain|how|why|translate|summari[sz]e|don[’\x27]t|do\s+not|never)\b', intent, re.I):
            return None
        video = bool(re.search(r'영상|동영상|비디오|\b(?:video|clip|movie)\b', intent, re.I))
        image = bool(re.search(r'이미지|그림|사진|일러스트|\b(?:image|picture|photo|illustration)\b', intent, re.I))
        draw = bool(re.search(r'그려\s*(?:줘|주세요|주라)|^(?:please\s+)?draw\b', intent, re.I))
        command = bool(re.search(r'(?:생성해|만들어|뽑아)\s*(?:줘|주세요|주라)|'
                                 r'^(?:please\s+|(?:can|could|would)\s+you\s+)?(?:generate|create|render|make)\b', intent, re.I))
        if not draw and not (command and (image or video)):
            return None
        mode = 'video' if video else 'image'
    family = 'h3' if mode == 'video' else str(options.get('family') or 'current')
    if family not in ({'h3'} if mode == 'video' else {'current', 'krea2'}):
        raise ValueError('지원하지 않는 생성 모델입니다')
    images = [x for x in latest.get('images', []) if isinstance(x, str) and x]
    if options.get('hadImage') and not images:
        raise ValueError('이전 참조 이미지를 다시 첨부한 후 생성해 주세요')
    if len(images) > 1:
        raise ValueError('생성 참조 이미지는 한 장만 첨부해 주세요')
    def bounded(key, default, low, high):
        value = float(options.get(key, default))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} 값은 {low}~{high} 범위여야 합니다')
        return value
    return GenerationPlan(mode, family, prompt, images[0] if images else '',
                          bounded('duration', 5, 1, 15), bounded('denoise', .65, .01, 1))


class GenerationCancelled(RuntimeError):
    pass


def prepare_prompt_payload(payload: dict) -> dict:
    """Run the snapshot's deferred, potentially file-reading hooks off the UI."""
    params = copy.deepcopy(payload)
    recipe = params.pop('_chat_deferred_prompt', None)
    if recipe is not None:
        from core.standard_hooks import run_pipeline_on_text
        from utils.file_wildcard import resolve_file_wildcards
        from utils.wildcard import process_wildcards
        for key in ('prompt', 'negative_prompt'):
            text = str(params.get(key) or '')
            if recipe.get('wildcards'):
                text = process_wildcards(resolve_file_wildcards(text))
            params[key] = run_pipeline_on_text(text)
    return params


def read_reference_image(value: str) -> bytes:
    """Accept one local path or bounded raster data URL; never fetch a URL."""
    from PIL import Image
    if value.startswith('data:'):
        if not value.startswith('data:image/') or ';base64,' not in value or len(value) > 28_000_000:
            raise ValueError('지원하지 않거나 너무 큰 참조 이미지입니다')
        data = base64.b64decode(value.split(',', 1)[1], validate=True)
    else:
        if re.match(r'^(https?|file|blob):', value, re.I):
            raise ValueError('참조 이미지는 로컬 파일을 첨부해 주세요')
        path = Path(value).expanduser()
        if not path.is_file() or path.stat().st_size > 20_000_000:
            raise ValueError('참조 이미지가 없거나 20MB를 넘습니다')
        data = path.read_bytes()
    if len(data) > 20_000_000:
        raise ValueError('참조 이미지가 20MB를 넘습니다')
    with Image.open(BytesIO(data)) as image:
        if image.format not in {'PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF'} or image.width * image.height > 64_000_000:
            raise ValueError('지원하지 않는 이미지 형식 또는 해상도입니다')
        image.verify()
    return data


class MediaGenerationJob:
    """One request owns its cancel flag, backend, temporary media and results.

    ``run_current`` and ``prepare_creator`` are blocking worker entrypoints.
    ``cancel`` never performs backend I/O on the caller/UI thread. The backend
    lease remains held until any in-flight interrupt is finished.
    """

    def __init__(self, request_id: str, plan: GenerationPlan, on_event=None):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError('생성 요청 ID는 1~100자여야 합니다')
        self.id, self.plan = request_id, plan
        self.cancelled = threading.Event()
        self._lock = threading.RLock()
        # Do not make the GUI wait on the backend interrupt's network lock.
        self._state_lock = threading.RLock()
        self._backend = None
        self._on_event = on_event or (lambda event: None)
        self._temporary = None
        self._terminal = False

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise GenerationCancelled('생성을 중지했습니다')

    def event(self, phase: str, **values) -> dict:
        packet = {'id': self.id, 'kind': self.plan.kind, 'phase': phase, **values}
        self._on_event(packet)
        return packet

    def cancel(self):
        with self._state_lock:
            if self._terminal or self.cancelled.is_set():
                return False
            self.cancelled.set()

        def interrupt_owned():
            with self._lock:
                if self._backend is not None:
                    try:
                        self._backend.interrupt()
                    except Exception:
                        pass
        threading.Thread(target=interrupt_owned, daemon=True, name='chat-media-stop').start()
        return True

    def terminal(self, *, error='', artifacts=None):
        with self._state_lock:
            stopped = self.cancelled.is_set()
            ok = not error and not stopped
            self._terminal = True
        return self.event('complete' if ok else 'stopped' if stopped else 'error',
                          done=True, ok=ok, stopped=stopped,
                          message=('영상' if self.plan.kind == 'video' else '이미지') + ' 생성 완료' if ok else '생성을 중지했습니다' if stopped else error,
                          error='' if stopped else error, artifacts=artifacts or [] if ok else [])

    def prepare_creator(self, snapshot: dict) -> dict:
        self.check_cancelled()
        self.event('preparing', message='Creator 생성과 참조 이미지를 준비하는 중')
        from core.krea2_generation import normalise_sampler, strip_standard_lora_tags

        values = prepare_prompt_payload(snapshot)
        video = self.plan.kind == 'video'
        mode = ('h3_i2v' if self.plan.image else 'h3_t2v') if video else ('krea2_edit' if self.plan.image else 'krea2_t2i')
        # Creator 그래프(Krea2/H3)는 Forge 식 <lora:...> 를 해석하지 않는다 — T2I Krea2 러너와 같이 뗀다.
        prompt = strip_standard_lora_tags(str(values.get('prompt', self.plan.prompt) or ''))
        params = {'requestId': self.id, 'mode': mode, 'prompt': prompt,
                  'seed': int(values.get('seed', -1))}
        if video:
            # The base H3 pack works without optional Turbo LoRA/grid assets.
            params.update(duration=self.plan.duration, fps=24, includeAudio=False,
                          quality='quality', steps=20)
        else:
            params.update(width=int(values.get('width', 1024)), height=int(values.get('height', 1024)))
            # T2I 스냅샷에서 온 요청이면 사용자가 정한 샘플링 설정을 따른다(Krea2 T2I 워커와 같은 해석).
            # steps 는 T2I(Turbo) 기준이라 T2I 모드에만 — 편집(krea2_edit)은 자기 기본 step 을 쓴다.
            if 'cfg_scale' in values or 'cfg' in values:
                params['cfg'] = float(values.get('cfg_scale', values.get('cfg')))
            if 'sampler_name' in values or 'sampler' in values:
                params['sampler'] = normalise_sampler(values.get('sampler_name', values.get('sampler')))
            if mode == 'krea2_t2i' and 'steps' in values:
                params['steps'] = int(values['steps'])
        if self.plan.image:
            data = read_reference_image(self.plan.image)
            self.check_cancelled()
            self._temporary = tempfile.TemporaryDirectory(prefix='aistudio-chat-input-')
            from PIL import Image
            with Image.open(BytesIO(data)) as image:
                suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'BMP': '.bmp', 'TIFF': '.tiff'}[image.format]
            source = Path(self._temporary.name) / ('reference' + suffix)
            source.write_bytes(data)
            params['sourcePath'] = str(source)
        self.check_cancelled()
        return params

    def close(self):
        """Call after the Creator result, off the GUI thread."""
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def run_current(self, backend, model: str, payload: dict, output_dir,
                    *, coordinator=None, unload_llm=False) -> dict:
        from core.resource_coordinator import get_generation_coordinator

        created = []
        try:
            self.check_cancelled()
            self.event('preparing', message='선택한 모델과 생성 설정을 준비하는 중')
            params = prepare_prompt_payload(payload)
            params.pop('_generation_family', None)
            params['save_images'] = False  # publish owned outputs only after success
            if self.plan.image:
                params['init_images'] = [base64.b64encode(read_reference_image(self.plan.image)).decode('ascii')]
                params['denoising_strength'] = self.plan.denoise
                # Hires.fix is a txt2img pass, not an img2img API parameter.
                for key in list(params):
                    if key.startswith('hr_') or key == 'enable_hr':
                        params.pop(key)
            # 직전 생성 뒤의 '생성 후 언로드' 요청이 아직 날아가는 중이면 끝난 뒤에 보낸다
            # (이 작업 스레드에서 기다린다 — 생성 워커와 같은 규칙, core.post_generation)
            from core.post_generation import reserve_generation_lease, wait_for_pending_unload
            wait_for_pending_unload(cancelled=self.cancelled.is_set)
            self.check_cancelled()
            # 기다림 뒤 막 시작된 언로드가 리스를 쥐었으면 그걸 기다렸다 다시 잡는다(언로드와 배타)
            with reserve_generation_lease(
                f'chat:{self.id}', coordinator=coordinator or get_generation_coordinator(),
                unload_llm=unload_llm, cancelled=self.cancelled.is_set,
            ):
                with self._lock:
                    self.check_cancelled()
                    self._backend = backend
                try:
                    self.event('generating', message='이미지를 생성하는 중')
                    def progress(value, maximum, _preview=None):
                        if not self.cancelled.is_set():
                            self.event('generating', progress=max(0, min(100, round(value * 100 / maximum))) if maximum else 0)
                    generate = backend.img2img if self.plan.image else backend.txt2img
                    # cancel_check 로 모델 전환 중·발송 직후의 중지도 백엔드가 직접 확인한다
                    # (전역 interrupt 한 번은 그 구간에서 사라진다 — core/cancellable_call.py).
                    from core.cancellable_call import call_with_optional_cancel
                    result = call_with_optional_cancel(
                        generate, model, params,
                        progress_callback=progress, cancel_check=self.cancelled.is_set,
                    )
                    self.check_cancelled()
                    if not result.success:
                        raise RuntimeError(result.error or '생성 결과를 받지 못했습니다')
                finally:
                    with self._lock:
                        self._backend = None
            self.event('saving', message='생성 결과를 저장하는 중')
            entries = list(getattr(result, 'artifacts', []) or [])
            if not entries and result.image_data:
                from backends.base import MediaArtifact
                entries = [MediaArtifact('image', data=result.image_data, filename='image.png', mime='image/png')]
            if not entries:
                raise RuntimeError('백엔드가 저장할 이미지 결과를 반환하지 않았습니다')
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for index, artifact in enumerate(entries):
                self.check_cancelled()
                data = getattr(artifact, 'data', None)
                if not data and getattr(artifact, 'path', None):
                    data = Path(artifact.path).read_bytes()
                if not data:
                    continue
                suffix = Path(getattr(artifact, 'filename', '') or 'image.png').suffix.lower()
                if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm'}:
                    suffix = '.png'
                target = directory / f'chat_{uuid.uuid4().hex}{suffix}'
                temporary = target.with_suffix(suffix + '.part')
                try:
                    temporary.write_bytes(data)
                    self.check_cancelled()
                    os.replace(temporary, target)
                    created.append({'kind': artifact.kind, 'path': str(target.resolve()).replace('\\', '/'),
                                    'filename': target.name, 'mime': getattr(artifact, 'mime', '') or ''})
                finally:
                    temporary.unlink(missing_ok=True)
            with self._state_lock:
                self.check_cancelled()
                if not created:
                    raise RuntimeError('생성 결과에 저장 가능한 이미지가 없습니다')
                return self.terminal(artifacts=created)
        except Exception as exc:
            # Only files created by this exact request can be removed here.
            for artifact in created:
                try:
                    Path(artifact['path']).unlink(missing_ok=True)
                except OSError:
                    pass
            return self.terminal(error=str(exc)[:2000])
