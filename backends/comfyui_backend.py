# backends/comfyui_backend.py
"""ComfyUI 백엔드 구현"""
import json
import uuid
import random
import requests
import websocket
from typing import Dict, List, Optional, Tuple

from backends.base import (
    AbstractBackend, BackendInfo, GenerationResult, ProgressCallback
)

import config
from utils.app_logger import get_logger

_logger = get_logger('comfyui')


def analyze_workflow(file_path: str) -> dict:
    """워크플로우 JSON을 분석하여 요약 정보 반환

    Returns:
        {
            'valid': bool,
            'error': str | None,
            'format': 'api' | 'web',
            'node_count': int,
            'checkpoint': str | None,
            'ksampler_type': str | None,
            'has_positive_clip': bool,
            'has_negative_clip': bool,
            'has_save_node': bool,
            'width': int | None,
            'height': int | None,
            'nodes_summary': list[str],
        }
    """
    import os
    result = {
        'valid': False, 'error': None, 'format': None,
        'node_count': 0, 'checkpoint': None, 'ksampler_type': None,
        'has_positive_clip': False, 'has_negative_clip': False,
        'has_save_node': False, 'width': None, 'height': None,
        'nodes_summary': [],
    }

    if not file_path or not os.path.exists(file_path):
        result['error'] = "파일을 찾을 수 없습니다."
        return result

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result['error'] = f"JSON 파싱 오류: {e}"
        return result
    except Exception as e:
        result['error'] = f"파일 읽기 오류: {e}"
        return result

    # 포맷 감지
    if 'nodes' in data and isinstance(data['nodes'], list):
        result['format'] = 'web'
        nodes_by_type = {}
        for node in data.get('nodes', []):
            nt = node.get('type', 'Unknown')
            nodes_by_type[nt] = nodes_by_type.get(nt, 0) + 1

            wv = node.get('widgets_values', [])
            if nt == 'CheckpointLoaderSimple' and wv:
                result['checkpoint'] = str(wv[0])
            elif nt in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom'):
                result['ksampler_type'] = nt
            elif nt == 'EmptyLatentImage' and len(wv) >= 2:
                try:
                    result['width'] = int(wv[0])
                    result['height'] = int(wv[1])
                except (ValueError, TypeError):
                    pass
            elif nt in ('SaveImage', 'PreviewImage'):
                result['has_save_node'] = True
            elif nt in ('CLIPTextEncode', 'CLIPTextEncodeSDXL'):
                result['has_positive_clip'] = True

        result['node_count'] = len(data.get('nodes', []))
        result['nodes_summary'] = [f"{t} x{c}" for t, c in sorted(nodes_by_type.items())]
    else:
        result['format'] = 'api'
        nodes_by_type = {}
        for node_id, node in data.items():
            if not isinstance(node, dict):
                continue
            cls = node.get('class_type', 'Unknown')
            nodes_by_type[cls] = nodes_by_type.get(cls, 0) + 1
            inputs = node.get('inputs', {})

            if cls == 'CheckpointLoaderSimple':
                result['checkpoint'] = inputs.get('ckpt_name')
            elif cls in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom'):
                result['ksampler_type'] = cls
            elif cls == 'EmptyLatentImage':
                try:
                    result['width'] = int(inputs.get('width', 0))
                    result['height'] = int(inputs.get('height', 0))
                except (ValueError, TypeError):
                    pass
            elif cls in ('SaveImage', 'PreviewImage'):
                result['has_save_node'] = True
            elif cls in ('CLIPTextEncode', 'CLIPTextEncodeSDXL'):
                result['has_positive_clip'] = True

        result['node_count'] = len([k for k, v in data.items() if isinstance(v, dict)])
        result['nodes_summary'] = [f"{t} x{c}" for t, c in sorted(nodes_by_type.items())]

    # 유효성 검사
    errors = []
    if not result['ksampler_type']:
        errors.append("KSampler 노드 없음")
    if not result['has_positive_clip']:
        errors.append("CLIPTextEncode 노드 없음")
    if not result['has_save_node']:
        errors.append("SaveImage/PreviewImage 노드 없음")

    if errors:
        result['error'] = ", ".join(errors)
    else:
        result['valid'] = True

    return result


class ComfyUIBackend(AbstractBackend):
    """ComfyUI API 백엔드"""

    def __init__(self, api_url: str):
        super().__init__(api_url)

    def get_backend_type(self) -> str:
        return "comfyui"

    def test_connection(self) -> bool:
        """ComfyUI 연결 상태 확인"""
        try:
            r = requests.get(f'{self.api_url}/system_stats', timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def get_info(self) -> BackendInfo:
        """ComfyUI /object_info에서 모델/샘플러/스케줄러 추출"""
        info = BackendInfo()

        obj_info = requests.get(
            f'{self.api_url}/object_info', timeout=15
        ).json()

        # 체크포인트 모델
        ckpt_node = obj_info.get('CheckpointLoaderSimple', {})
        ckpt_input = ckpt_node.get('input', {}).get('required', {})
        models = ckpt_input.get('ckpt_name', [[]])[0]
        if isinstance(models, list):
            info.models = models
            info.checkpoints = ["Use same checkpoint"] + models

        # 샘플러
        ksampler_node = obj_info.get('KSampler', {})
        ksampler_input = ksampler_node.get('input', {}).get('required', {})
        samplers = ksampler_input.get('sampler_name', [[]])[0]
        if isinstance(samplers, list):
            info.samplers = samplers

        # 스케줄러
        schedulers = ksampler_input.get('scheduler', [[]])[0]
        if isinstance(schedulers, list):
            info.schedulers = schedulers

        # 업스케일러
        try:
            upscale_node = obj_info.get('UpscaleModelLoader', {})
            upscale_input = upscale_node.get('input', {}).get('required', {})
            upscalers = upscale_input.get('model_name', [[]])[0]
            if isinstance(upscalers, list):
                info.upscalers = upscalers
        except Exception:
            pass

        # VAE
        try:
            vae_node = obj_info.get('VAELoader', {})
            vae_input = vae_node.get('input', {}).get('required', {})
            vae_list = vae_input.get('vae_name', [[]])[0]
            if isinstance(vae_list, list):
                info.vae = ["Use same VAE"] + vae_list
        except Exception:
            pass

        return info

    def get_system_stats(self) -> dict:
        """GPU/VRAM 상태 조회"""
        try:
            r = requests.get(f'{self.api_url}/system_stats', timeout=3)
            if r.status_code == 200:
                data = r.json()
                devices = data.get('devices', [])
                if devices:
                    dev = devices[0]
                    return {
                        'vram_used': dev.get('vram_total', 0) - dev.get('vram_free', 0),
                        'vram_total': dev.get('vram_total', 0),
                        'vram_free': dev.get('vram_free', 0),
                    }
        except Exception:
            pass
        return {}

    def get_loras(self) -> list:
        """ComfyUI LoRA 목록 반환"""
        try:
            resp = requests.get(
                f'{self.api_url}/object_info/LoraLoader', timeout=10
            )
            resp.raise_for_status()
            obj_info = resp.json()
            lora_node = obj_info.get('LoraLoader', {})
            lora_input = lora_node.get('input', {}).get('required', {})
            names = lora_input.get('lora_name', [[]])[0]
            if isinstance(names, list):
                return [
                    {'name': n, 'alias': n, 'path': '', 'trigger_words': []}
                    for n in names
                ]
        except Exception as e:
            _logger.warning(f"ComfyUI LoRA 목록 로드 실패: {e}")
        return []

    # ── 워크플로우 포맷 감지 및 변환 ──

    def _load_workflow(self) -> dict:
        """사용자 워크플로우 JSON 로드 (API/웹 포맷 자동 감지)"""
        workflow_path = getattr(config, 'COMFYUI_WORKFLOW_PATH', '')
        if not workflow_path:
            raise RuntimeError(
                "ComfyUI 워크플로우 파일이 설정되지 않았습니다.\n"
                "API 관리에서 워크플로우 JSON 파일을 선택해주세요."
            )

        import os
        if not os.path.exists(workflow_path):
            raise RuntimeError(
                f"워크플로우 파일을 찾을 수 없습니다:\n{workflow_path}"
            )

        with open(workflow_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 포맷 감지: 웹 포맷은 'nodes' 키가 있음
        if 'nodes' in data and isinstance(data['nodes'], list):
            _logger.info("웹 포맷 워크플로우 감지 → API 포맷으로 변환")
            return self._convert_web_to_api(data)

        # API 포맷: 최상위 키가 노드 ID
        _logger.info(f"API 포맷 워크플로우 로드 (노드 {len(data)}개)")
        return data

    def _convert_web_to_api(self, web_data: dict) -> dict:
        """ComfyUI 웹 포맷 → API 포맷 변환"""
        nodes = web_data.get('nodes', [])
        links = web_data.get('links', [])

        # 링크 맵 구성: link_id → (source_node_id, source_slot)
        link_map = {}
        for link in links:
            # link = [link_id, source_node_id, source_slot, dest_node_id, dest_slot, type]
            if len(link) >= 6:
                link_id = link[0]
                source_node_id = link[1]
                source_slot = link[2]
                link_map[link_id] = (source_node_id, source_slot)

        api_workflow = {}

        for node in nodes:
            node_id = str(node.get('id', ''))
            node_type = node.get('type', '')
            if not node_id or not node_type:
                continue

            api_node = {
                'class_type': node_type,
                'inputs': {}
            }

            # 위젯 값을 inputs에 매핑
            widget_values = node.get('widgets_values', [])
            node_inputs = node.get('inputs', [])
            node_outputs = node.get('outputs', [])

            # 입력 슬롯 처리 (링크 연결)
            for inp in node_inputs:
                inp_name = inp.get('name', '')
                inp_link = inp.get('link')
                if inp_link is not None and inp_link in link_map:
                    src_id, src_slot = link_map[inp_link]
                    api_node['inputs'][inp_name] = [str(src_id), src_slot]

            # 위젯 값 매핑 (노드 타입별)
            self._map_widget_values(api_node, node_type, widget_values)

            api_workflow[node_id] = api_node

        _logger.info(f"웹→API 변환 완료 (노드 {len(api_workflow)}개)")
        return api_workflow

    def _map_widget_values(self, api_node: dict, node_type: str, values: list):
        """노드 타입별 위젯 값을 inputs에 매핑"""
        inputs = api_node['inputs']

        if not values:
            return

        try:
            if node_type in ('KSampler',):
                # KSampler: seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise
                if len(values) >= 7:
                    inputs.setdefault('seed', values[0])
                    # values[1] = control_after_generate (skip)
                    inputs.setdefault('steps', values[2])
                    inputs.setdefault('cfg', values[3])
                    inputs.setdefault('sampler_name', values[4])
                    inputs.setdefault('scheduler', values[5])
                    inputs.setdefault('denoise', values[6])

            elif node_type == 'KSamplerAdvanced':
                # add_noise, noise_seed, control_after_generate, steps, cfg, sampler_name, scheduler,
                # start_at_step, end_at_step, return_with_leftover_noise
                if len(values) >= 10:
                    inputs.setdefault('add_noise', values[0])
                    inputs.setdefault('noise_seed', values[1])
                    inputs.setdefault('steps', values[3])
                    inputs.setdefault('cfg', values[4])
                    inputs.setdefault('sampler_name', values[5])
                    inputs.setdefault('scheduler', values[6])
                    inputs.setdefault('start_at_step', values[7])
                    inputs.setdefault('end_at_step', values[8])
                    inputs.setdefault('return_with_leftover_noise', values[9])

            elif node_type == 'CLIPTextEncode':
                if len(values) >= 1:
                    inputs.setdefault('text', values[0])

            elif node_type == 'CheckpointLoaderSimple':
                if len(values) >= 1:
                    inputs.setdefault('ckpt_name', values[0])

            elif node_type == 'EmptyLatentImage':
                if len(values) >= 3:
                    inputs.setdefault('width', values[0])
                    inputs.setdefault('height', values[1])
                    inputs.setdefault('batch_size', values[2])

            elif node_type == 'SaveImage':
                if len(values) >= 1:
                    inputs.setdefault('filename_prefix', values[0])

            elif node_type == 'VAELoader':
                if len(values) >= 1:
                    inputs.setdefault('vae_name', values[0])

        except (IndexError, TypeError):
            pass

    # ── 노드 탐색 ──

    def _find_ksampler_node(self, workflow: dict) -> Tuple[str, dict]:
        """KSampler 계열 노드 찾기"""
        sampler_types = (
            'KSampler', 'KSamplerAdvanced',
            'SamplerCustom', 'SamplerCustomAdvanced',
        )
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            cls = node.get('class_type', '')
            if cls in sampler_types:
                return node_id, node

        raise RuntimeError(
            "워크플로우에서 KSampler 노드를 찾을 수 없습니다.\n"
            f"지원되는 노드: {', '.join(sampler_types)}"
        )

    def _find_clip_encode_node(self, workflow: dict, start_node_id: str,
                                max_depth: int = 5) -> Optional[str]:
        """링크를 따라가며 CLIPTextEncode 노드를 찾기 (다단계 추적)"""
        clip_types = ('CLIPTextEncode', 'CLIPTextEncodeSDXL')
        visited = set()

        def trace(node_id: str, depth: int) -> Optional[str]:
            if depth > max_depth or node_id in visited:
                return None
            visited.add(node_id)

            node = workflow.get(node_id)
            if not node or not isinstance(node, dict):
                return None

            if node.get('class_type', '') in clip_types:
                return node_id

            # 이 노드의 입력을 추적하여 CLIPTextEncode 찾기
            inputs = node.get('inputs', {})
            for key, val in inputs.items():
                if isinstance(val, list) and len(val) >= 1:
                    linked_id = str(val[0])
                    result = trace(linked_id, depth + 1)
                    if result:
                        return result
            return None

        return trace(start_node_id, 0)

    def _trace_clip_nodes(self, workflow: dict, ksampler_node: dict) -> Tuple[Optional[str], Optional[str]]:
        """KSampler의 positive/negative 입력에서 CLIPTextEncode 노드 ID 찾기"""
        inputs = ksampler_node.get('inputs', {})

        positive_id = None
        negative_id = None

        # positive 입력 추적 (다단계)
        pos_input = inputs.get('positive')
        if isinstance(pos_input, list) and len(pos_input) >= 1:
            start_id = str(pos_input[0])
            positive_id = self._find_clip_encode_node(workflow, start_id)

        # negative 입력 추적 (다단계)
        neg_input = inputs.get('negative')
        if isinstance(neg_input, list) and len(neg_input) >= 1:
            start_id = str(neg_input[0])
            negative_id = self._find_clip_encode_node(workflow, start_id)

        return positive_id, negative_id

    def _apply_params(self, workflow: dict, model_name: str, payload: dict):
        """워크플로우 노드에 UI 파라미터 매핑"""
        ksampler_id, ksampler_node = self._find_ksampler_node(workflow)
        inputs = ksampler_node.get('inputs', {})
        cls = ksampler_node.get('class_type', '')

        _logger.info(f"KSampler 노드: ID={ksampler_id}, type={cls}")

        # KSampler 파라미터
        seed = payload.get('seed', -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        if cls == 'KSamplerAdvanced':
            inputs['noise_seed'] = seed
        else:
            inputs['seed'] = seed

        inputs['steps'] = payload.get('steps', 20)
        inputs['cfg'] = payload.get('cfg_scale', 7.0)
        inputs['sampler_name'] = payload.get('sampler_name', 'euler')
        inputs['scheduler'] = payload.get('scheduler', 'normal')
        inputs['denoise'] = payload.get('denoising_strength', 1.0)

        # CLIP Text Encode (positive/negative) — 다단계 추적
        pos_id, neg_id = self._trace_clip_nodes(workflow, ksampler_node)
        _logger.info(f"CLIPTextEncode: positive={pos_id}, negative={neg_id}")

        if pos_id and pos_id in workflow:
            pos_node = workflow[pos_id]
            pos_cls = pos_node.get('class_type', '')
            if pos_cls == 'CLIPTextEncode':
                pos_node['inputs']['text'] = payload.get('prompt', '')
            elif pos_cls == 'CLIPTextEncodeSDXL':
                # SDXL: text_g와 text_l 모두 설정
                prompt_text = payload.get('prompt', '')
                pos_node['inputs']['text_g'] = prompt_text
                pos_node['inputs']['text_l'] = prompt_text

        if neg_id and neg_id in workflow:
            neg_node = workflow[neg_id]
            neg_cls = neg_node.get('class_type', '')
            if neg_cls == 'CLIPTextEncode':
                neg_node['inputs']['text'] = payload.get('negative_prompt', '')
            elif neg_cls == 'CLIPTextEncodeSDXL':
                neg_text = payload.get('negative_prompt', '')
                neg_node['inputs']['text_g'] = neg_text
                neg_node['inputs']['text_l'] = neg_text

        # CheckpointLoaderSimple
        if model_name:
            for node_id, node in workflow.items():
                if not isinstance(node, dict):
                    continue
                if node.get('class_type') == 'CheckpointLoaderSimple':
                    node['inputs']['ckpt_name'] = model_name
                    break

        # EmptyLatentImage
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            if node.get('class_type') == 'EmptyLatentImage':
                node['inputs']['width'] = payload.get('width', 512)
                node['inputs']['height'] = payload.get('height', 512)
                node['inputs']['batch_size'] = 1
                break

        _logger.info("워크플로우 파라미터 매핑 완료")

    # ── 생성 및 결과 수신 ──

    def _queue_and_wait(self, workflow: dict,
                        progress_callback: Optional[ProgressCallback] = None) -> GenerationResult:
        """워크플로우를 큐에 넣고 WebSocket으로 결과 대기"""
        ws = None
        try:
            # ★ 요청마다 고유 client_id 생성
            client_id = str(uuid.uuid4())

            # ★ WebSocket 먼저 연결 (프롬프트 제출 전에 연결해야 메시지를 놓치지 않음)
            ws_url = self.api_url.replace('http://', 'ws://').replace('https://', 'wss://')
            _logger.info(f"WebSocket 연결: {ws_url}/ws?clientId={client_id}")
            ws = websocket.create_connection(
                f'{ws_url}/ws?clientId={client_id}',
                timeout=600
            )

            # 프롬프트 제출
            _logger.info("프롬프트 제출 중...")
            prompt_response = requests.post(
                f'{self.api_url}/prompt',
                json={'prompt': workflow, 'client_id': client_id},
                timeout=30
            )

            if prompt_response.status_code != 200:
                error_text = prompt_response.text
                try:
                    error_data = prompt_response.json()
                    # ComfyUI 에러 응답 파싱
                    error_info = error_data.get('error', {})
                    error_msg = error_info.get('message', error_text)

                    # 노드별 에러 상세
                    node_errors = error_data.get('node_errors', {})
                    if node_errors:
                        details = []
                        for nid, nerr in node_errors.items():
                            for e in nerr.get('errors', []):
                                details.append(f"  노드 {nid}: {e.get('message', str(e))}")
                        if details:
                            error_msg += "\n\n노드 에러:\n" + "\n".join(details)
                except Exception:
                    error_msg = error_text

                _logger.error(f"프롬프트 제출 실패: {error_msg}")
                return GenerationResult(
                    success=False,
                    error=f"ComfyUI 큐 등록 실패 (HTTP {prompt_response.status_code}):\n{error_msg}"
                )

            resp_data = prompt_response.json()
            prompt_id = resp_data.get('prompt_id')
            if not prompt_id:
                return GenerationResult(success=False, error="prompt_id를 받지 못했습니다.")

            # 노드 에러 확인 (200이어도 에러가 있을 수 있음)
            node_errors = resp_data.get('node_errors', {})
            if node_errors:
                details = []
                for nid, nerr in node_errors.items():
                    for e in nerr.get('errors', []):
                        details.append(f"  노드 {nid}: {e.get('message', str(e))}")
                if details:
                    _logger.warning(f"노드 경고: {details}")

            _logger.info(f"프롬프트 등록 완료: {prompt_id}")

            # 결과 대기
            return self._wait_for_result(ws, prompt_id, progress_callback)

        except requests.exceptions.RequestException as e:
            _logger.error(f"API 요청 실패: {e}")
            return GenerationResult(success=False, error=f"ComfyUI API 요청 실패: {e}")
        except websocket.WebSocketException as e:
            _logger.error(f"WebSocket 오류: {e}")
            return GenerationResult(success=False, error=f"ComfyUI WebSocket 오류: {e}")
        except Exception as e:
            _logger.error(f"생성 오류: {e}", exc_info=True)
            return GenerationResult(success=False, error=f"ComfyUI 생성 오류: {e}")
        finally:
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass

    def _wait_for_result(self, ws, prompt_id: str,
                         progress_callback: Optional[ProgressCallback]) -> GenerationResult:
        """WebSocket 메시지를 수신하며 결과 대기"""
        _logger.info(f"결과 대기 중... (prompt_id={prompt_id})")

        while True:
            msg = ws.recv()
            if isinstance(msg, bytes):
                continue  # 바이너리 메시지 (프리뷰 이미지) 스킵

            data = json.loads(msg)
            msg_type = data.get('type', '')

            if msg_type == 'status':
                # 큐 상태 업데이트
                continue

            elif msg_type == 'progress':
                d = data.get('data', {})
                if progress_callback:
                    step = d.get('value', 0)
                    total = d.get('max', 0)
                    progress_callback(step, total, None)

            elif msg_type == 'executing':
                d = data.get('data', {})
                if d.get('node') is None:
                    # 실행 완료 (node=None은 전체 완료를 의미)
                    _logger.info("실행 완료")
                    break

            elif msg_type == 'execution_error':
                d = data.get('data', {})
                error_msg = d.get('exception_message', '알 수 없는 오류')
                traceback_lines = d.get('traceback', [])
                if traceback_lines:
                    error_msg += "\n" + "".join(traceback_lines[-3:])
                _logger.error(f"실행 오류: {error_msg}")
                return GenerationResult(success=False, error=f"ComfyUI 실행 오류:\n{error_msg}")

            elif msg_type == 'execution_cached':
                # 캐시된 노드 알림
                continue

        # 결과 이미지 가져오기
        return self._fetch_result_image(prompt_id)

    def _fetch_result_image(self, prompt_id: str) -> GenerationResult:
        """히스토리에서 결과 이미지 다운로드"""
        _logger.info(f"결과 이미지 다운로드 중... (prompt_id={prompt_id})")

        history = requests.get(
            f'{self.api_url}/history/{prompt_id}', timeout=10
        ).json()

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get('outputs', {})

        if not outputs:
            _logger.error("히스토리에 출력 데이터 없음")
            return GenerationResult(
                success=False,
                error="ComfyUI 히스토리에서 출력을 찾을 수 없습니다.\n"
                      "워크플로우에 SaveImage 노드가 있는지 확인하세요."
            )

        # SaveImage / PreviewImage 노드 출력에서 이미지 찾기
        for node_id, node_output in outputs.items():
            images = node_output.get('images', [])
            if images:
                img_info = images[0]
                _logger.info(f"이미지 다운로드: {img_info['filename']}")

                img_response = requests.get(
                    f'{self.api_url}/view',
                    params={
                        'filename': img_info['filename'],
                        'subfolder': img_info.get('subfolder', ''),
                        'type': img_info.get('type', 'output'),
                    },
                    timeout=30
                )
                img_response.raise_for_status()

                gen_info = {
                    'prompt_id': prompt_id,
                    'filename': img_info['filename'],
                }

                _logger.info("이미지 수신 완료")
                return GenerationResult(
                    success=True,
                    image_data=img_response.content,
                    info=gen_info
                )

        _logger.error("출력 노드에 이미지 없음")
        return GenerationResult(
            success=False,
            error="ComfyUI 출력에서 이미지를 찾을 수 없습니다.\n"
                  "워크플로우에 SaveImage 또는 PreviewImage 노드가 있는지 확인하세요."
        )

    # ── 공개 API ──

    def txt2img(self, model_name: str, payload: Dict,
                progress_callback: Optional[ProgressCallback] = None) -> GenerationResult:
        """텍스트→이미지 생성"""
        try:
            _logger.info(f"=== ComfyUI txt2img 시작 ===")
            _logger.info(f"모델: {model_name}")
            _logger.info(f"워크플로우 경로: {getattr(config, 'COMFYUI_WORKFLOW_PATH', '(미설정)')}")

            workflow = self._load_workflow()
            self._apply_params(workflow, model_name, payload)
            return self._queue_and_wait(workflow, progress_callback)

        except FileNotFoundError as e:
            _logger.error(f"워크플로우 파일 없음: {e}")
            return GenerationResult(success=False, error=f"워크플로우 파일을 찾을 수 없습니다:\n{e}")
        except json.JSONDecodeError as e:
            _logger.error(f"워크플로우 JSON 파싱 실패: {e}")
            return GenerationResult(success=False, error=f"워크플로우 JSON 파싱 실패:\n{e}")
        except RuntimeError as e:
            _logger.error(f"워크플로우 처리 오류: {e}")
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            _logger.error(f"예기치 못한 오류: {e}", exc_info=True)
            return GenerationResult(success=False, error=f"ComfyUI 생성 오류: {e}")

    def _load_img2img_workflow(self) -> dict:
        """img2img 워크플로우 JSON 로드"""
        import os
        workflow_path = getattr(config, 'COMFYUI_WORKFLOW_IMG2IMG_PATH', '')
        if not workflow_path:
            raise RuntimeError(
                "ComfyUI img2img 워크플로우 파일이 설정되지 않았습니다.\n"
                "설정에서 img2img 워크플로우 JSON 파일을 선택해주세요."
            )
        if not os.path.exists(workflow_path):
            raise RuntimeError(
                f"img2img 워크플로우 파일을 찾을 수 없습니다:\n{workflow_path}"
            )
        with open(workflow_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'nodes' in data and isinstance(data['nodes'], list):
            return self._convert_web_to_api(data)
        return data

    def _upload_image(self, image_b64: str) -> str:
        """ComfyUI에 이미지 업로드 → 파일명 반환"""
        import base64
        image_bytes = base64.b64decode(image_b64)

        resp = requests.post(
            f'{self.api_url}/upload/image',
            files={'image': ('input.png', image_bytes, 'image/png')},
            data={'overwrite': 'true'},
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        filename = result.get('name', '')
        if not filename:
            raise RuntimeError("업로드 응답에 파일명이 없습니다.")
        _logger.info(f"이미지 업로드 완료: {filename}")
        return filename

    def _find_load_image_node(self, workflow: dict) -> Optional[str]:
        """LoadImage 노드 ID 찾기"""
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            if node.get('class_type') == 'LoadImage':
                return node_id
        return None

    def img2img(self, model_name: str, payload: Dict,
                progress_callback: Optional[ProgressCallback] = None) -> GenerationResult:
        """이미지→이미지 생성"""
        try:
            _logger.info("=== ComfyUI img2img 시작 ===")

            workflow = self._load_img2img_workflow()

            # 입력 이미지 업로드
            init_images = payload.get('init_images', [])
            if not init_images:
                return GenerationResult(success=False, error="입력 이미지가 없습니다.")

            uploaded_filename = self._upload_image(init_images[0])

            # LoadImage 노드에 파일명 설정
            load_img_id = self._find_load_image_node(workflow)
            if load_img_id and load_img_id in workflow:
                workflow[load_img_id]['inputs']['image'] = uploaded_filename
            else:
                _logger.warning("LoadImage 노드를 찾을 수 없습니다. 이미지가 적용되지 않을 수 있습니다.")

            # 파라미터 적용
            self._apply_params(workflow, model_name, payload)

            # denoise 설정 (img2img에서 중요)
            try:
                ks_id, ks_node = self._find_ksampler_node(workflow)
                ks_node['inputs']['denoise'] = payload.get('denoising_strength', 0.75)
            except RuntimeError:
                pass

            return self._queue_and_wait(workflow, progress_callback)

        except RuntimeError as e:
            _logger.error(f"img2img 오류: {e}")
            return GenerationResult(success=False, error=str(e))
        except Exception as e:
            _logger.error(f"img2img 예외: {e}", exc_info=True)
            return GenerationResult(success=False, error=f"ComfyUI img2img 오류: {e}")

    def upscale(self, image_b64: str, settings: Dict) -> str:
        """업스케일 (추후 구현)"""
        raise NotImplementedError(
            "ComfyUI 업스케일은 아직 지원되지 않습니다.\n"
            "워크플로우에 업스케일 노드를 추가하여 사용하세요."
        )

    def adetailer(self, image_b64: str, settings: Dict) -> str:
        """ADetailer (ComfyUI에서는 지원하지 않음)"""
        raise NotImplementedError(
            "ComfyUI에서는 ADetailer가 지원되지 않습니다.\n"
            "워크플로우에 디테일러 노드를 추가하여 사용하세요."
        )
