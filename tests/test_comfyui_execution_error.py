"""ComfyUI ``execution_error`` → the app's result text shows the exception message only (comfy-3 O4).

The fixture is the event ComfyUI 0.36 sent in the GE comfy-3 run for the flat custom sigma list
(``xs_flat_error``: Extra Schedulers custom sigmas ``[1.0, 1.0, 0.0]`` on ``ForgeNeoKSamplerCNS``): the keys of
``execution.py`` ``handle_execution_error`` (+ ``add_message``'s ``timestamp``), ``exception_message`` =
``"{ex}\\n{tips}"`` (no tips) and ``traceback`` = ``traceback.format_tb`` — its 14 frames copied from that run's
ComfyUI console; node and input names from that run's graph (``current_inputs`` object values abbreviated, as
``format_value`` would print them). Before the fix the app's result was the message plus the last three frames
(``GE_RESULT_BEFORE``, recorded in that run); now it is the message, and the whole traceback is in the app log.
"""
from __future__ import annotations

import json
import logging
import unittest
from unittest import mock

from backends.comfyui_backend import ComfyUIBackend
from backends.comfyui_execution_error import (
    UNKNOWN_ERROR,
    execution_error_log,
    execution_error_message,
    execution_error_traceback,
)
from backends.comfyui_progress import ProgressTracker

PROMPT_ID = "02319c4f-306a-45d6-ac48-49bbd0fa9b4c"
MESSAGE = (
    "Extra Schedulers: the custom scheduler's sigma list gives sigma 1 at steps 0 and 1; a step that keeps sigma where it was is wasted with Euler and makes multistep samplers such as Res Multistep divide by zero (NaN, a black image)"
)
FRAMES = [
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\execution.py", line 547, in execute\n'
    '    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)\n'
    '                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\execution.py", line 352, in get_output_data\n'
    '    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)\n'
    '                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\execution.py", line 326, in _async_map_node_over_list\n'
    '    await process_inputs(input_dict, i)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\execution.py", line 314, in process_inputs\n'
    '    result = f(**inputs)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\ai_studio_forge_parity\\generation.py", line 398, in sample\n'
    '    return _common_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\ai_studio_forge_parity\\generation.py", line 298, in _common_sample\n'
    '    samples = comfy.sample.sample(\n'
    '        model, noise, int(steps), float(cfg), sampler_name, scheduler,\n'
    '    ...<2 lines>...\n'
    '        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed),\n'
    '    )\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\comfy\\sample.py", line 80, in sample\n'
    '    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\comfy\\samplers.py", line 1413, in __init__\n'
    '    self.set_steps(steps, denoise)\n'
    '    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\comfy\\samplers.py", line 1434, in set_steps\n'
    '    self.sigmas = self.calculate_sigmas(steps).to(self.device)\n'
    '                  ~~~~~~~~~~~~~~~~~~~~~^^^^^^^\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\comfy\\samplers.py", line 1425, in calculate_sigmas\n'
    '    sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\comfy\\samplers.py", line 1385, in calculate_sigmas\n'
    '    return handler.handler(model_sampling, steps)\n'
    '           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\ai_studio_forge_parity\\extra_schedulers.py", line 592, in extra_scheduler\n'
    '    sigmas = schedule(name, steps, sigma_min, sigma_max, settings, model_sampling=model_sampling)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\ai_studio_forge_parity\\extra_schedulers.py", line 539, in schedule\n'
    '    return custom(n, sigma_min, sigma_max, settings, device)\n',
    '  File "C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\ai_studio_forge_parity\\extra_schedulers.py", line 522, in custom\n'
    '    raise CustomSchedulerError(\n'
    '    ...<2 lines>...\n'
    '    )\n',
]
EXECUTION_ERROR = {
    "prompt_id": PROMPT_ID,
    "node_id": "10",
    "node_type": "ForgeNeoKSamplerCNS",
    "executed": ["1", "2", "3", "4", "6", "9", "5", "7", "8"],
    "exception_message": MESSAGE + "\n",
    "exception_type": "C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\ai_studio_forge_parity."
                      "extra_schedulers_math.CustomSchedulerError",
    "traceback": FRAMES,
    "current_inputs": {
        "model": ["<comfy.model_patcher.ModelPatcher object at 0x0000021C5E8B7A10>"],
        "positive": ["[[tensor(...), {}]]"], "negative": ["[[tensor(...), {}]]"],
        "latent_image": ["{'samples': tensor(...)}"], "seed": [11], "steps": [28], "cfg": [5.0],
        "sampler_name": ["res_multistep"], "scheduler": ["custom"], "denoise": [1.0], "cns_enabled": [False],
    },
    "current_outputs": ["1", "2", "3", "4", "6", "9", "5", "7", "8"],
    "timestamp": 1791158035412,
}
# What the app showed for this event before the fix (GE comfy-3 results.jsonl, xs_flat_error rep 1).
GE_RESULT_BEFORE = "ComfyUI 실행 오류:\n" + MESSAGE + "\n\n" + "".join(FRAMES[-3:])


def _event(kind, **data):
    return json.dumps({"type": kind, "data": {"prompt_id": PROMPT_ID, **data}})


class ExecutionErrorTextTests(unittest.TestCase):
    def test_message_is_the_exception_without_comfys_trailing_newline(self):
        self.assertEqual(execution_error_message(EXECUTION_ERROR), MESSAGE)
        self.assertNotIn("File ", execution_error_message(EXECUTION_ERROR))

    def test_tips_stay_with_the_message(self):
        # ComfyUI puts its own tips (out of memory, CLIP shape) after the exception: they are part of the message.
        data = dict(EXECUTION_ERROR, exception_message="Allocation on device\nThis error means you ran out of "
                                                        "memory on your GPU.")
        self.assertEqual(execution_error_message(data),
                         "Allocation on device\nThis error means you ran out of memory on your GPU.")

    def test_missing_or_empty_message_is_unknown(self):
        for data in ({}, {"exception_message": ""}, {"exception_message": "\n"}, {"exception_message": None}, None):
            with self.subTest(data=data):
                self.assertEqual(execution_error_message(data), UNKNOWN_ERROR)

    def test_exception_without_text_names_its_type_and_failing_frame(self):
        # ComfyUI sends "{ex}\n{tips}" = "\n" for a bare assert / RuntimeError(): the type and the last frame
        # stand in, so the result still points at the failing code (the old result showed the frames).
        frames = ['  File "C:\\ComfyUI\\execution.py", line 496, in execute\n    output_data = get_output_data()\n',
                  '  File "C:\\ComfyUI\\comfy\\model_patcher.py", line 1766, in load\n'
                  '    assert load_device is not None\n']
        bare_assert = {"exception_message": "\n", "exception_type": "AssertionError", "traceback": frames}
        self.assertEqual(execution_error_message(bare_assert),
                         "AssertionError — model_patcher.py:1766 (load): assert load_device is not None")
        self.assertEqual(execution_error_message(dict(bare_assert, traceback="".join(frames))),
                         "AssertionError — model_patcher.py:1766 (load): assert load_device is not None")
        empty_runtime = {"exception_message": "\n", "exception_type": "builtins.RuntimeError", "traceback": []}
        self.assertEqual(execution_error_message(empty_runtime), "RuntimeError")
        self.assertEqual(execution_error_message({"exception_message": "", "traceback": frames}),
                         "model_patcher.py:1766 (load): assert load_device is not None")

    def test_log_record_keeps_the_whole_traceback(self):
        record = execution_error_log(EXECUTION_ERROR)
        self.assertTrue(record.startswith(MESSAGE + " (node 10, type ForgeNeoKSamplerCNS, exception "))
        self.assertTrue(record.endswith("".join(FRAMES).rstrip()))
        self.assertEqual(execution_error_traceback(EXECUTION_ERROR), "".join(FRAMES).rstrip())
        self.assertEqual(execution_error_traceback(dict(EXECUTION_ERROR, traceback="  File x\n")), "  File x")
        self.assertEqual(execution_error_log({"exception_message": "boom\n"}), "boom")


class ExecutionErrorResultTests(unittest.TestCase):
    workflow = {"10": {"class_type": "ForgeNeoKSamplerCNS", "inputs": {}}}

    def test_result_shows_the_message_and_the_app_log_keeps_the_traceback(self):
        ws = mock.Mock()
        ws.recv.side_effect = [
            _event("execution_start"),
            _event("execution_cached", nodes=["1", "2", "3", "4", "6", "9"]),
            json.dumps({"type": "execution_error", "data": EXECUTION_ERROR}),
        ]
        backend = ComfyUIBackend("http://127.0.0.1:8188")
        with mock.patch.object(backend, "_fetch_result_artifacts") as fetch, \
                self.assertLogs("comfyui", level=logging.ERROR) as logs:
            result = backend._wait_for_result(ws, PROMPT_ID, None, tracker=ProgressTracker(self.workflow))
        fetch.assert_not_called()
        self.assertFalse(result.success)
        self.assertEqual(result.error, "ComfyUI 실행 오류:\n" + MESSAGE)
        self.assertNotIn("File ", result.error)
        self.assertNotIn("Traceback", result.error)
        # exactly the old text with the traceback taken off
        self.assertEqual(GE_RESULT_BEFORE, result.error + "\n\n" + "".join(FRAMES[-3:]))
        logged = "\n".join(record.getMessage() for record in logs.records)
        self.assertIn(MESSAGE, logged)
        for frame in FRAMES:
            self.assertIn(frame.rstrip(), logged)


if __name__ == "__main__":
    unittest.main()
