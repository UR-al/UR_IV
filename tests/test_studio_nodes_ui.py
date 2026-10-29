"""The studio's UI-only nodes (panel / result view / XYZ view) and the lazy family switch."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_nodes

UI_NODES = {
    "AIStudioPanel": "state",
    "AIStudioResultView": "view",
    "AIStudioXYZView": "xyz",
}


class UiNodeTests(unittest.TestCase):
    def test_ui_nodes_have_no_outputs_and_not_output_nodes(self):
        for name, widget in UI_NODES.items():
            with self.subTest(node=name):
                cls = studio_nodes.NODE_CLASS_MAPPINGS[name]
                self.assertEqual(cls.RETURN_TYPES, ())
                self.assertFalse(getattr(cls, "OUTPUT_NODE", False))
                spec = cls.INPUT_TYPES()
                self.assertEqual(set(spec), {"required"})
                self.assertEqual(list(spec["required"]), [widget])
                kind, options = spec["required"][widget]
                self.assertEqual(kind, "STRING")
                # A hidden widget without an input socket: the studio JS draws the node, the value is its saved state.
                self.assertTrue(options["hidden"])
                self.assertTrue(options["socketless"])
                self.assertIn(name, studio_nodes.NODE_DISPLAY_NAME_MAPPINGS)


class SwitchTests(unittest.TestCase):
    def test_switch_lazy_requests_only_selected(self):
        switch = studio_nodes.AIStudioSwitch()
        self.assertEqual(switch.check_lazy_status(1, a=None, b=None), ["b"])
        self.assertEqual(switch.check_lazy_status(0, a=None, b=None), ["a"])
        self.assertEqual(getattr(switch, switch.FUNCTION)(1, a="A", b="B"), ("B",))
        self.assertEqual(getattr(switch, switch.FUNCTION)(0, a="A", b="B"), ("A",))

    def test_unselected_side_may_be_missing(self):
        # The unused family's nodes are bypassed, so their link to the switch is gone.
        switch = studio_nodes.AIStudioSwitch()
        self.assertEqual(switch.check_lazy_status(0), ["a"])
        self.assertEqual(getattr(switch, switch.FUNCTION)(0, a="A"), ("A",))

    def test_switch_inputs_are_optional_lazy_any(self):
        spec = studio_nodes.AIStudioSwitch.INPUT_TYPES()
        kind, options = spec["required"]["select"]
        self.assertEqual((kind, options["min"], options["max"], options["default"]), ("INT", 0, 1, 0))
        for name in ("a", "b"):
            with self.subTest(input=name):
                self.assertEqual(spec["optional"][name][0], "*")
                self.assertTrue(spec["optional"][name][1]["lazy"])
        self.assertEqual(studio_nodes.AIStudioSwitch.RETURN_TYPES, ("*",))
        self.assertFalse(getattr(studio_nodes.AIStudioSwitch, "OUTPUT_NODE", False))


if __name__ == "__main__":
    unittest.main()
