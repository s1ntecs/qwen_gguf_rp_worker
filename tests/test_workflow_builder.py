import unittest

from workflow_builder import TURBO_STEPS, build_workflow, normalize_loras


class WorkflowBuilderTests(unittest.TestCase):
    def test_t2i(self):
        graph, steps = build_workflow(prompt="a cat", seed=123)
        self.assertEqual(steps, 25)
        self.assertEqual(graph["model"]["class_type"], "UnetLoaderGGUF")
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
        self.assertEqual(graph["lora_1"]["inputs"]["model"], ["model", 0])
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


if __name__ == "__main__":
    unittest.main()
