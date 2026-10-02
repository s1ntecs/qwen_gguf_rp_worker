import unittest

from workflow_builder import TURBO_STEPS, build_workflow, normalize_loras


class WorkflowBuilderTests(unittest.TestCase):
    def test_t2i(self):
        graph, steps = build_workflow(prompt="a cat", seed=123)
        self.assertEqual(steps, 25)
        self.assertEqual(graph["model"]["class_type"], "UnetLoaderGGUFAdvanced")
        self.assertTrue(graph["model"]["inputs"]["patch_on_device"])
        self.assertEqual(graph["latent"]["inputs"]["width"], 1024)
        self.assertEqual(graph["sampler"]["inputs"]["seed"], 123)

    def test_edit_two_images(self):
        graph, _ = build_workflow(prompt="combine", image_names=["a.png", "b.png"])
        self.assertEqual(graph["text"]["inputs"]["images.image_1"], ["image_1", 0])
        self.assertEqual(graph["text"]["inputs"]["images.image_2"], ["image_2", 0])
        self.assertEqual(graph["sampler"]["inputs"]["latent_image"], ["text", 2])

    def test_resize_primary(self):
        graph, _ = build_workflow(prompt="edit", image_names=["a.png"], width=1300, height=700)
        self.assertEqual(graph["resize_primary"]["inputs"]["width"], 1312)
        self.assertEqual(graph["resize_primary"]["inputs"]["height"], 704)
        self.assertEqual(graph["text"]["inputs"]["resolution"], 0)

    def test_lora_chain(self):
        graph, _ = build_workflow(
            prompt="portrait",
            loras=[{"name": "faces/detail.safetensors", "strength": 0.7}, "style.safetensors"],
        )
        self.assertEqual(graph["lora_1"]["inputs"]["model"], ["builtin_loras", 0])
        self.assertEqual(graph["lora_2"]["inputs"]["model"], ["lora_1", 0])
        self.assertEqual(graph["sampler"]["inputs"]["model"], ["lora_2", 0])

    def test_turbo(self):
        graph, steps = build_workflow(prompt="a cat", turbo=True, steps=99)
        self.assertEqual(steps, TURBO_STEPS)
        self.assertEqual(graph["turbo_lora"]["class_type"], "ViggleTurboLora")
        self.assertEqual(graph["sampler"]["class_type"], "SamplerCustomAdvanced")

    def test_bad_lora_path(self):
        with self.assertRaises(ValueError):
            normalize_loras([{"name": "../bad.safetensors"}])

    def test_builtin_lora_defaults_reach_sampler(self):
        graph, _ = build_workflow(prompt="portrait")
        self.assertEqual(graph["builtin_loras"]["class_type"], "QwenBuiltinLoraStack")
        self.assertEqual(graph["builtin_loras"]["inputs"], {
            "model": ["model", 0], "nsfw_strength": 1.0,
            "penis_strength": 0.0, "vagina_strength": 0.0,
        })
        self.assertEqual(graph["sampler"]["inputs"]["model"], ["builtin_loras", 0])

    def test_builtin_loras_can_all_be_disabled(self):
        graph, _ = build_workflow(prompt="portrait", lora_strengths={
            "nsfw": 0, "penis": 0, "vagina": 0,
        })
        for name in ("nsfw", "penis", "vagina"):
            self.assertEqual(graph["builtin_loras"]["inputs"][f"{name}_strength"], 0.0)

    def test_partial_strengths_preserve_other_defaults(self):
        graph, _ = build_workflow(prompt="portrait", lora_strengths={"penis": 0.65})
        self.assertEqual(graph["builtin_loras"]["inputs"]["nsfw_strength"], 1.0)
        self.assertEqual(graph["builtin_loras"]["inputs"]["penis_strength"], 0.65)
        self.assertEqual(graph["builtin_loras"]["inputs"]["vagina_strength"], 0.0)

    def test_turbo_then_builtin_then_custom_lora(self):
        graph, _ = build_workflow(prompt="portrait", turbo=True,
                                  loras=[{"name": "style.safetensors", "strength": 0.4}])
        self.assertEqual(graph["builtin_loras"]["inputs"]["model"], ["turbo_lora", 0])
        self.assertEqual(graph["lora_1"]["inputs"]["model"], ["builtin_loras", 0])
        self.assertEqual(graph["guider"]["inputs"]["model"], ["lora_1", 0])

    def test_builtin_filename_in_legacy_loras_is_not_applied_twice(self):
        graph, _ = build_workflow(prompt="portrait", loras=[{
            "name": "NSFW Qwen by TheseAlpacas V2.safetensors", "strength": 0.3,
        }])
        self.assertEqual(graph["builtin_loras"]["inputs"]["nsfw_strength"], 0.3)
        self.assertNotIn("lora_1", graph)

    def test_conflicting_builtin_strengths_are_rejected(self):
        with self.assertRaises(ValueError):
            build_workflow(prompt="portrait", lora_strengths={"nsfw": 0}, loras=[{
                "name": "NSFW Qwen by TheseAlpacas V2.safetensors", "strength": 1,
            }])

    def test_invalid_builtin_strengths_are_rejected(self):
        for values in ([1, 0, 0], {"unknown": 1}, {"nsfw": None},
                       {"nsfw": "oops"}, {"nsfw": True}, {"nsfw": 4.01},
                       {"nsfw": float("nan")}, {"nsfw": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_workflow(prompt="portrait", lora_strengths=values)

    def test_nonfinite_custom_strength_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_loras([{"name": "style.safetensors", "strength": value}])


if __name__ == "__main__":
    unittest.main()
