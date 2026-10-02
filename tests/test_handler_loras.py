import unittest
from unittest.mock import patch

import handler


class HandlerLoraTests(unittest.TestCase):
    def setUp(self):
        self.queued = []
        for name, replacement in (
            ("wait_for_comfy", lambda: None),
            ("queue_workflow", self.queue),
            ("wait_result", lambda prompt_id: {"outputs": {}}),
            ("fetch_outputs", lambda record: ([], [])),
        ):
            fixture = patch.object(handler, name, replacement)
            fixture.start()
            self.addCleanup(fixture.stop)

    def queue(self, workflow, client_id):
        self.queued.append(workflow)
        return "test-job"

    def test_defaults_are_applied_and_reported(self):
        result = handler.handler({"input": {"prompt": "portrait", "seed": 42}})
        self.assertNotIn("error", result)
        self.assertEqual(result.get("lora_strengths"), {"nsfw": 1.0, "penis": 0.0, "vagina": 0.0})
        self.assertEqual(result["loras"], [
            {"name": "NSFW Qwen by TheseAlpacas V2.safetensors", "strength": 1.0},
            {"name": "qwen-image-2.1_penis_coachbate_preview1.safetensors", "strength": 0.0},
            {"name": "qwen21_v2_000002750.safetensors", "strength": 0.0},
        ])

    def test_strengths_are_forwarded_to_workflow_and_response(self):
        values = {"nsfw": 0, "penis": 0.6, "vagina": 0.4}
        result = handler.handler({"input": {"prompt": "portrait", "lora_strengths": values,
                                            "loras": [{"name": "style.safetensors", "strength": 0}]}})
        self.assertNotIn("error", result)
        self.assertEqual(result.get("lora_strengths"), values)
        self.assertEqual(self.queued[0]["builtin_loras"]["inputs"]["nsfw_strength"], 0)
        self.assertEqual(self.queued[0]["builtin_loras"]["inputs"]["penis_strength"], 0.6)
        self.assertEqual(self.queued[0]["lora_1"]["inputs"]["strength_model"], 0)

    def test_invalid_strengths_fail_before_submission(self):
        result = handler.handler({"input": {"prompt": "portrait", "lora_strengths": {"nsfw": "bad"}}})
        self.assertIn("error", result)
        self.assertEqual(self.queued, [])

    def test_legacy_filename_is_reported_once_with_effective_strength(self):
        result = handler.handler({"input": {"prompt": "portrait", "lora": [{
            "name": "NSFW Qwen by TheseAlpacas V2.safetensors", "strength": 0.3,
        }]}})
        self.assertNotIn("error", result)
        self.assertEqual(result.get("lora_strengths", {}).get("nsfw"), 0.3)
        self.assertEqual(len(result["loras"]), 3)
        self.assertNotIn("lora_1", self.queued[0])
