import ast
import math
import unittest
from pathlib import Path

from experiments.tbs.optimization import (
    build_discriminative_parameter_groups,
    get_named_learning_rates,
    validate_backbone_lr_multiplier,
)


class FakeParameter:
    def __init__(self, name, requires_grad=True):
        self.name = name
        self.requires_grad = requires_grad


class FakeModule:
    def __init__(self, parameters):
        self._parameters = list(parameters)

    def parameters(self):
        return iter(self._parameters)


class FakeModel(FakeModule):
    def __init__(self, backbone_parameters, head_parameters):
        self.backbone = FakeModule(backbone_parameters)
        super().__init__([*backbone_parameters, *head_parameters])


class DiscriminativeParameterGroupTests(unittest.TestCase):
    def test_builds_non_overlapping_head_and_backbone_groups(self):
        backbone_trainable = FakeParameter("backbone_trainable")
        backbone_frozen = FakeParameter("backbone_frozen", requires_grad=False)
        head_trainable = FakeParameter("head_trainable")
        head_frozen = FakeParameter("head_frozen", requires_grad=False)
        model = FakeModel(
            [backbone_trainable, backbone_frozen],
            [head_trainable, head_frozen],
        )

        groups = build_discriminative_parameter_groups(
            model,
            base_lr=1e-4,
            backbone_lr_multiplier=0.1,
        )

        self.assertEqual([group["group_name"] for group in groups], ["head", "backbone"])
        self.assertEqual(groups[0]["lr"], 1e-4)
        self.assertEqual(groups[1]["lr"], 1e-5)
        self.assertEqual(groups[0]["params"], [head_trainable])
        self.assertEqual(groups[1]["params"], [backbone_trainable])

        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))

    def test_multiplier_one_preserves_equal_learning_rates(self):
        model = FakeModel([FakeParameter("backbone")], [FakeParameter("head")])

        groups = build_discriminative_parameter_groups(
            model,
            base_lr=2e-4,
            backbone_lr_multiplier=1.0,
        )

        self.assertEqual([group["lr"] for group in groups], [2e-4, 2e-4])

    def test_rejects_invalid_multiplier_values(self):
        invalid_values = [
            None,
            "invalid",
            False,
            True,
            0,
            -0.1,
            1.1,
            math.inf,
            math.nan,
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    validate_backbone_lr_multiplier(value)

    def test_rejects_non_positive_or_non_finite_base_learning_rate(self):
        model = FakeModel([FakeParameter("backbone")], [FakeParameter("head")])

        for value in [0, -1e-4, math.inf, math.nan]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_discriminative_parameter_groups(model, value, 0.1)

    def test_requires_trainable_parameters_in_both_groups(self):
        cases = [
            FakeModel([FakeParameter("backbone", requires_grad=False)], [FakeParameter("head")]),
            FakeModel([FakeParameter("backbone")], [FakeParameter("head", requires_grad=False)]),
        ]

        for model in cases:
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "trainable"):
                    build_discriminative_parameter_groups(model, 1e-4, 0.1)

    def test_reads_learning_rates_by_group_name(self):
        parameter_groups = [
            {"group_name": "head", "lr": 8e-5},
            {"group_name": "backbone", "lr": 8e-6},
        ]

        learning_rates = get_named_learning_rates(parameter_groups)

        self.assertEqual(learning_rates, {"head": 8e-5, "backbone": 8e-6})

    def test_rejects_missing_or_duplicate_learning_rate_groups(self):
        invalid_groups = [
            [{"group_name": "head", "lr": 1e-4}],
            [
                {"group_name": "head", "lr": 1e-4},
                {"group_name": "head", "lr": 1e-5},
                {"group_name": "backbone", "lr": 1e-5},
            ],
        ]

        for parameter_groups in invalid_groups:
            with self.subTest(parameter_groups=parameter_groups):
                with self.assertRaises(ValueError):
                    get_named_learning_rates(parameter_groups)


class Stage1ArgumentParsingTests(unittest.TestCase):
    def test_backbone_multiplier_config_default_is_not_coerced_before_validation(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "train_tbs_stage1.py"
        )
        tree = ast.parse(script_path.read_text(encoding="utf-8"))

        multiplier_argument = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--backbone_lr_multiplier"
            ):
                continue
            multiplier_argument = node
            break

        self.assertIsNotNone(multiplier_argument)
        default_node = next(
            keyword.value
            for keyword in multiplier_argument.keywords
            if keyword.arg == "default"
        )
        self.assertFalse(
            isinstance(default_node, ast.Call)
            and isinstance(default_node.func, ast.Name)
            and default_node.func.id == "float",
            "JSON defaults must reach validate_backbone_lr_multiplier without coercion",
        )


if __name__ == "__main__":
    unittest.main()
