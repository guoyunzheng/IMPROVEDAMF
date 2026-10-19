"""Verify expert lookup without importing MuJoCo or generating demonstrations."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch, sentinel


class ExpertLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only the lookup function is exercised; optional simulation packages
        # are stubbed while loading the real generator module.
        path = Path(__file__).resolve().parents[1] / "scripts/generate_metaworld_demos.py"
        spec = importlib.util.spec_from_file_location("demo_generator_lookup_test", path)
        cls.generator = importlib.util.module_from_spec(spec)
        stubs = {name: ModuleType(name) for name in ("gymnasium", "metaworld", "numpy", "zarr")}
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(cls.generator)

    def setUp(self):
        self.parent = ModuleType("metaworld")
        self.parent.__path__ = []
        self.policies = ModuleType("metaworld.policies")
        self.policies.__path__ = []
        self.policies.ENV_POLICY_MAP = {}
        self.modules = {"metaworld": self.parent, "metaworld.policies": self.policies}

    def test_nonmatching_environment_and_expert_names_use_official_map(self):
        factory = Mock(return_value=sentinel.peg_policy)
        self.policies.ENV_POLICY_MAP["peg-insert-side-v3"] = factory
        with patch.dict(sys.modules, self.modules):
            self.assertIs(self.generator.get_policy("peg-insert-side"), sentinel.peg_policy)
        factory.assert_called_once_with()

    def test_map_accepts_normalized_names_and_existing_tasks(self):
        factory = Mock(return_value=sentinel.reach_policy)
        self.policies.ENV_POLICY_MAP["reach-v3"] = factory
        with patch.dict(sys.modules, self.modules):
            for task in ("reach", "reach-v3", "reach_v3"):
                self.assertIs(self.generator.get_policy(task), sentinel.reach_policy)
        self.assertEqual(factory.call_count, 3)

    def test_explicit_policy_name_alias_still_works(self):
        name = "metaworld.policies.sawyer_peg_insertion_side_v3_policy"
        module = ModuleType(name)
        module.SawyerPegInsertionSideV3Policy = Mock(return_value=sentinel.alias_policy)
        self.modules[name] = module
        with patch.dict(sys.modules, self.modules):
            self.assertIs(self.generator.get_policy("peg-insertion-side"), sentinel.alias_policy)

    def test_invalid_task_does_not_silently_select_another_expert(self):
        with patch.dict(sys.modules, self.modules), self.assertRaises(ModuleNotFoundError):
            self.generator.get_policy("not-a-task")


if __name__ == "__main__":
    unittest.main()
