"""Regression tests for the supernet, the sandwich rule and the ranking metrics."""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supernet import (  # noqa: E402
    BarsTask,
    LayerChoice,
    MNISTTask,
    RankingExperimentConfig,
    SandwichConfig,
    SandwichTrainer,
    SearchSpace,
    SubnetSpec,
    Supernet,
    accuracy,
    average_subnet_loss,
    build_dense_subnet,
    build_shrink_schedule,
    classification_loss,
    kendall_tau,
    make_task,
    normalize_images,
    run_ranking_experiment,
    spearman_rho,
    subnet_accuracy,
)
from supernet.dataset import DEFAULT_DATA_DIR, MNIST_FILES  # noqa: E402

TEST_SPACE = SearchSpace(
    min_depth=2,
    max_depth=4,
    embed_dims=(16, 32),
    head_dims=(8, 16),
    ffn_ratios=(2, 3),
    ffn_kernels=(1, 3),
    activations=("gelu", "silu"),
    num_classes=10,
    input_features=8,
    max_seq_len=6,
)


def make_inputs(batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch_size, TEST_SPACE.input_features, TEST_SPACE.max_seq_len,
                       generator=generator)


TEST_TASK = BarsTask(input_features=TEST_SPACE.input_features, seq_len=TEST_SPACE.max_seq_len)


def make_labels(batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, TEST_SPACE.num_classes, (batch_size,), generator=generator)


class SearchSpaceTests(unittest.TestCase):
    def test_max_subnet_is_the_expensive_corner(self) -> None:
        spec = TEST_SPACE.max_subnet()
        self.assertEqual(spec.depth, TEST_SPACE.max_depth)
        self.assertEqual(spec.embed_dim, TEST_SPACE.max_embed_dim)
        self.assertEqual(spec.num_heads, TEST_SPACE.max_embed_dim // 8)
        for layer in spec.layers:
            self.assertEqual(layer.ffn_ratio, max(TEST_SPACE.ffn_ratios))
            self.assertEqual(layer.ffn_kernel, max(TEST_SPACE.ffn_kernels))

    def test_min_subnet_is_the_cheap_corner(self) -> None:
        spec = TEST_SPACE.min_subnet()
        self.assertEqual(spec.depth, TEST_SPACE.min_depth)
        self.assertEqual(spec.embed_dim, TEST_SPACE.min_embed_dim)
        for layer in spec.layers:
            self.assertEqual(layer.ffn_ratio, min(TEST_SPACE.ffn_ratios))
            self.assertEqual(layer.ffn_kernel, min(TEST_SPACE.ffn_kernels))

    def test_sampling_is_reproducible_and_valid(self) -> None:
        left = [TEST_SPACE.sample(random.Random(3)).to_dict() for _ in range(5)]
        right = [TEST_SPACE.sample(random.Random(3)).to_dict() for _ in range(5)]
        self.assertEqual(left, right)
        other = [TEST_SPACE.sample(random.Random(4)).to_dict() for _ in range(5)]
        self.assertNotEqual(left, other)

    def test_arch_id_ignores_derived_fields(self) -> None:
        spec = TEST_SPACE.min_subnet()
        payload = spec.to_dict()
        self.assertEqual(SubnetSpec.from_dict(payload).arch_id, spec.arch_id)


class SupernetForwardTests(unittest.TestCase):
    def test_extreme_subnets_produce_logits(self) -> None:
        model = Supernet(TEST_SPACE)
        inputs = make_inputs()
        for spec in (TEST_SPACE.max_subnet(), TEST_SPACE.min_subnet()):
            logits = model(inputs, spec)
            self.assertEqual(logits.shape, (inputs.shape[0], TEST_SPACE.num_classes))
            self.assertTrue(torch.isfinite(logits).all())

    def test_unused_branch_cannot_leak(self) -> None:
        torch.manual_seed(0)
        model = Supernet(TEST_SPACE)
        spec = SubnetSpec(
            depth=2,
            embed_dim=32,
            head_dim=8,
            layers=(LayerChoice(2, 1, "gelu"), LayerChoice(2, 1, "silu")),
        )
        inputs = make_inputs()
        with torch.no_grad():
            before = model(inputs, spec)
            model.layers[0].ffn.branches["r3k3"].conv1.weight.add_(1.0)
            model.layers[0].ffn.branches["r3k3"].conv2.bias.add_(1.0)
            after = model(inputs, spec)
        self.assertTrue(torch.equal(before, after))

    def test_shallow_subnet_ignores_deeper_layers(self) -> None:
        torch.manual_seed(0)
        model = Supernet(TEST_SPACE)
        spec = SubnetSpec(
            depth=2,
            embed_dim=16,
            head_dim=8,
            layers=(LayerChoice(2, 1, "gelu"), LayerChoice(2, 1, "gelu")),
        )
        inputs = make_inputs()
        with torch.no_grad():
            before = model(inputs, spec)
            model.layers[3].attn.qkv.weight.add_(1.0)
            model.layers[3].ffn.branches["r2k1"].conv1.weight.add_(1.0)
            after = model(inputs, spec)
        self.assertTrue(torch.equal(before, after))

    def test_export_subnet_matches_shared_forward(self) -> None:
        torch.manual_seed(1)
        model = Supernet(TEST_SPACE)
        inputs = make_inputs()
        specs = [TEST_SPACE.max_subnet(), TEST_SPACE.min_subnet()]
        specs.append(
            SubnetSpec(
                depth=3,
                embed_dim=32,
                head_dim=16,
                layers=(
                    LayerChoice(3, 3, "silu"),
                    LayerChoice(2, 1, "gelu"),
                    LayerChoice(3, 1, "silu"),
                ),
            )
        )
        for spec in specs:
            exported = model.export_subnet(spec)
            with torch.no_grad():
                shared = model(inputs, spec)
                separate = exported(inputs, spec)
            self.assertTrue(
                torch.allclose(shared, separate, atol=1e-6),
                msg=f"export mismatch for {spec.arch_id}",
            )

    def test_parameter_count_matches_exported_model(self) -> None:
        model = Supernet(TEST_SPACE)
        for spec in (TEST_SPACE.max_subnet(), TEST_SPACE.min_subnet()):
            exported = model.export_subnet(spec)
            actual = sum(parameter.numel() for parameter in exported.parameters())
            self.assertEqual(model.subnet_parameter_count(spec), actual)

    def test_dense_subnet_matches_point_space_forward(self) -> None:
        spec = TEST_SPACE.min_subnet()
        model = build_dense_subnet(TEST_SPACE, spec)
        logits = model(make_inputs(), spec)
        self.assertEqual(logits.shape, (2, TEST_SPACE.num_classes))

    def test_predict_returns_label_indices(self) -> None:
        model = Supernet(TEST_SPACE)
        labels = model.predict(make_inputs(4), TEST_SPACE.min_subnet())
        self.assertEqual(labels.shape, (4,))
        self.assertTrue(bool(((labels >= 0) & (labels < TEST_SPACE.num_classes)).all()))
        self.assertTrue(model.training, msg="predict must restore training mode")


class LossTests(unittest.TestCase):
    def test_uniform_logits_give_log_classes(self) -> None:
        logits = torch.zeros(4, TEST_SPACE.num_classes)
        labels = make_labels(4)
        loss = classification_loss(logits, labels)
        self.assertAlmostEqual(float(loss), math.log(TEST_SPACE.num_classes), places=5)
        expected = float((labels == 0).to(torch.float32).mean())
        self.assertAlmostEqual(accuracy(logits, labels), expected, places=5)

    def test_accurate_logits_score_better(self) -> None:
        labels = make_labels(4)
        good = torch.nn.functional.one_hot(labels, TEST_SPACE.num_classes).float() * 10.0
        self.assertLess(float(classification_loss(good, labels)), 0.01)
        self.assertEqual(accuracy(good, labels), 1.0)

class SandwichRuleTests(unittest.TestCase):
    def _batch(self):
        return make_inputs(4), make_labels(4)

    def test_sandwich_legs_contain_the_extremes(self) -> None:
        trainer = SandwichTrainer(TEST_SPACE, SandwichConfig(n_random=2, seed=0))
        legs = trainer.build_sandwich()
        roles = [role for role, _, _ in legs]
        self.assertEqual(roles[0], "max")
        self.assertEqual(roles[1], "min")
        self.assertEqual(len(roles), 4)
        self.assertEqual(legs[0][1], TEST_SPACE.max_subnet())
        self.assertEqual(legs[1][1], TEST_SPACE.min_subnet())

    def test_training_step_touches_max_min_and_random(self) -> None:
        trainer = SandwichTrainer(TEST_SPACE, SandwichConfig(n_random=2, seed=1))
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.model.named_parameters()
        }
        inputs, labels = self._batch()
        record = trainer.train_step(inputs, labels)
        self.assertEqual(set(record.role_losses), {"max", "min", "random_0", "random_1"})
        self.assertEqual(set(record.role_accuracy), {"max", "min", "random_0", "random_1"})
        self.assertGreater(record.grad_norm, 0.0)
        deepest = f"layers.{TEST_SPACE.max_depth - 1}.attn.qkv.weight"
        self.assertFalse(torch.equal(before[deepest], trainer.model.state_dict()[deepest]))

    def test_coverage_reaches_every_depth_and_kernel(self) -> None:
        trainer = SandwichTrainer(TEST_SPACE, SandwichConfig(n_random=3, seed=2))
        batch_fn = TEST_TASK.batch_fn(4, seed=0)
        trainer.train(batch_fn, steps=20)
        self.assertEqual(
            set(trainer.stats.depth_counts),
            set(range(TEST_SPACE.min_depth, TEST_SPACE.max_depth + 1)),
        )
        self.assertEqual(set(trainer.stats.kernel_counts), set(TEST_SPACE.ffn_kernels))
        self.assertEqual(trainer.stats.subnets_trained, 20 * 5)

    def test_training_improves_the_max_subnet(self) -> None:
        space = SearchSpace(
            min_depth=2, max_depth=2, embed_dims=(16,), head_dims=(8,),
            ffn_ratios=(2,), ffn_kernels=(1,), activations=("gelu",),
            num_classes=10, input_features=8, max_seq_len=28,
        )
        task = BarsTask(noise=0.1, input_features=8, seq_len=28)
        trainer = SandwichTrainer(
            space, SandwichConfig(n_random=1, seed=3, lr=5e-3, weight_decay=0.0)
        )
        spec = space.max_subnet()
        batches = task.fixed_batches(16, 4, seed=9)
        before = average_subnet_loss(trainer.model, spec, batches)
        batch_fn = task.batch_fn(16, seed=4)
        trainer.train(batch_fn, steps=80)
        after = average_subnet_loss(trainer.model, spec, batches)
        self.assertLess(after, before)


class RankingMetricTests(unittest.TestCase):
    def test_identical_rankings_correlate_perfectly(self) -> None:
        scores = [3.0, 1.0, 2.0, 4.0]
        self.assertAlmostEqual(kendall_tau(scores, scores), 1.0)
        self.assertAlmostEqual(spearman_rho(scores, scores), 1.0)

    def test_reversed_rankings_are_anticorrelated(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(kendall_tau(scores, scores[::-1]), -1.0)
        self.assertAlmostEqual(spearman_rho(scores, scores[::-1]), -1.0)

    def test_constant_scores_have_no_correlation(self) -> None:
        self.assertEqual(kendall_tau([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]), 0.0)
        self.assertEqual(spearman_rho([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]), 0.0)

    def test_ranking_experiment_reports_consistent_fields(self) -> None:
        torch.manual_seed(0)
        space = SearchSpace(
            min_depth=2, max_depth=3, embed_dims=(16,), head_dims=(8,),
            ffn_ratios=(2,), ffn_kernels=(1, 3), activations=("gelu",),
            num_classes=10, input_features=8, max_seq_len=6,
        )
        model = Supernet(space)
        config = RankingExperimentConfig(
            num_candidates=3, standalone_steps=3, eval_batches=2, batch_size=2, seed=0
        )
        result = run_ranking_experiment(
            model, space,
            BarsTask(input_features=space.input_features, seq_len=space.max_seq_len),
            config,
        )
        self.assertEqual(result["num_candidates"], 3)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertGreaterEqual(result["kendall_tau"], -1.0)
        self.assertLessEqual(result["kendall_tau"], 1.0)
        self.assertGreaterEqual(result["regret"], 0.0)
        ranks = sorted(row["supernet_rank"] for row in result["candidates"])
        self.assertEqual(ranks, [1, 2, 3])
        self.assertTrue(0.0 <= result["best_standalone_accuracy"] <= 1.0)

    def test_subnet_accuracy_is_between_zero_and_one(self) -> None:
        model = Supernet(TEST_SPACE)
        batches = TEST_TASK.fixed_batches(4, 2, seed=1)
        value = subnet_accuracy(model, TEST_SPACE.min_subnet(), batches)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class TaskTests(unittest.TestCase):
    def test_normalize_images_turns_rows_into_tokens(self) -> None:
        images = torch.zeros(3, 5, 7, dtype=torch.uint8)
        inputs = normalize_images(images)
        self.assertEqual(tuple(inputs.shape), (3, 7, 5))
        self.assertFalse(torch.isnan(inputs).any())

    def test_bars_task_shapes_and_class_range(self) -> None:
        task = BarsTask()
        inputs, labels = task.batch(6)
        self.assertEqual(tuple(inputs.shape), (6, task.input_features, task.seq_len))
        self.assertEqual(tuple(labels.shape), (6,))
        self.assertTrue(bool(((labels >= 0) & (labels < task.num_classes)).all()))

    def test_bars_batches_are_reproducible(self) -> None:
        task = BarsTask()
        left = task.batch_fn(3, seed=5)
        right = task.batch_fn(3, seed=5)
        for step in range(3):
            for a, b in zip(left(step), right(step)):
                self.assertTrue(torch.equal(a, b))

    def test_bars_preview_covers_every_class(self) -> None:
        task = BarsTask()
        images, labels = task.preview()
        self.assertEqual(sorted(labels.tolist()), list(range(task.num_classes)))
        self.assertEqual(tuple(images.shape), (task.num_classes, task.seq_len, task.input_features))
        self.assertEqual(images.dtype, torch.uint8)

class MNISTTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cached = all((DEFAULT_DATA_DIR / name).exists() for name in MNIST_FILES)
        if not cached:
            raise unittest.SkipTest("MNIST is not cached locally; run the demo once first")
        cls.task = MNISTTask(verbose=False, train_limit=2000)

    def test_shapes_and_class_count(self) -> None:
        self.assertEqual(self.task.num_classes, 10)
        self.assertEqual(self.task.seq_len, 28)
        self.assertEqual(self.task.input_features, 28)
        inputs, labels = self.task.batch(4)
        self.assertEqual(tuple(inputs.shape), (4, 28, 28))
        self.assertTrue(bool(((labels >= 0) & (labels < 10)).all()))

    def test_held_out_batches_differ_from_training_batches(self) -> None:
        train_inputs, _ = self.task.batch(4)
        val_inputs, _ = self.task.fixed_batches(4, 1, seed=0)[0]
        self.assertFalse(torch.allclose(train_inputs, val_inputs))

    def test_preview_returns_one_image_per_class(self) -> None:
        images, labels = self.task.preview()
        self.assertEqual(labels.tolist(), list(range(10)))
        self.assertEqual(tuple(images.shape), (10, 28, 28))


class ReportTests(unittest.TestCase):
    def test_snippets_come_from_the_real_source(self) -> None:
        from supernet.report import read_snippet

        code = read_snippet("supernet/modules.py", "elastic-conv")
        self.assertIn("def forward", code)
        self.assertIn("F.conv1d", code)
        self.assertNotIn("snippet:", code)
        self.assertIn("self.weight[:out_channels, :in_channels]", code)

    def test_all_report_snippets_resolve(self) -> None:
        from supernet.report import read_snippet

        for relative_path, name in (
            ("supernet/modules.py", "elastic-conv"),
            ("supernet/modules.py", "elastic-ffn"),
            ("supernet/supernet.py", "supernet-forward"),
            ("supernet/sandwich.py", "sandwich-build"),
            ("supernet/sandwich.py", "sandwich-step"),
        ):
            self.assertTrue(read_snippet(relative_path, name).strip())

    def test_highlight_marks_tokens_and_escapes_html(self) -> None:
        from supernet.report import highlight

        marked = highlight('def f(x):\n    return "s" + 1  # note')
        for token_class in ("tok-keyword", "tok-string", "tok-number", "tok-comment"):
            self.assertIn(token_class, marked)
        self.assertIn("&lt;script&gt;", highlight("x = '<script>'"))

    def test_missing_snippet_raises(self) -> None:
        from supernet.report import read_snippet

        with self.assertRaises(ValueError):
            read_snippet("supernet/modules.py", "not-a-snippet")


class SchematicFigureTests(unittest.TestCase):
    """The slicing figure must come from the spec, not from placeholder numbers."""

    def test_cell_size_keeps_the_grid_readable(self) -> None:
        try:
            from supernet.viz import _cell_size
        except ImportError:
            self.skipTest("matplotlib not installed")
        self.assertEqual(_cell_size(128, 96), 8)
        self.assertEqual(_cell_size(128, 128), 8)
        self.assertEqual(_cell_size(96, 64), 8)

    def test_schematic_is_rendered_from_the_spec(self) -> None:
        try:
            from supernet.viz import plot_slicing_schematic
        except ImportError:
            self.skipTest("matplotlib not installed")
        import tempfile

        spec = {
            "embed_dim": 96,
            "head_dim": 32,
            "depth": 3,
            "layers": [
                {"ffn_ratio": 2, "ffn_kernel": 1, "activation": "gelu"},
                {"ffn_ratio": 4, "ffn_kernel": 5, "activation": "silu"},
                {"ffn_ratio": 3, "ffn_kernel": 3, "activation": "gelu"},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "slicing.png"
            path = plot_slicing_schematic(
                spec, out, max_embed_dim=128,
                ffn_ratios=(2, 3, 4), ffn_kernels=(1, 3, 5),
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 4000)


class ProgressiveShrinkingTests(unittest.TestCase):
    def test_fewer_than_two_stages_means_off(self) -> None:
        self.assertEqual(build_shrink_schedule(TEST_SPACE, 0), ())
        self.assertEqual(build_shrink_schedule(TEST_SPACE, 1), ())

    def test_first_stage_unlocks_only_the_largest(self) -> None:
        first = build_shrink_schedule(TEST_SPACE, 4)[0]
        self.assertEqual(first.min_depth, TEST_SPACE.max_depth)
        self.assertEqual(first.embed_dims, (TEST_SPACE.max_embed_dim,))
        self.assertEqual(first.ffn_ratios, (max(TEST_SPACE.ffn_ratios),))
        self.assertEqual(first.ffn_kernels, (max(TEST_SPACE.ffn_kernels),))

    def test_last_stage_is_the_full_space(self) -> None:
        last = build_shrink_schedule(TEST_SPACE, 4)[-1]
        self.assertEqual(last.min_depth, TEST_SPACE.min_depth)
        self.assertEqual(tuple(sorted(last.embed_dims)), tuple(sorted(TEST_SPACE.embed_dims)))
        self.assertEqual(tuple(sorted(last.ffn_ratios)), tuple(sorted(TEST_SPACE.ffn_ratios)))
        self.assertEqual(tuple(sorted(last.ffn_kernels)), tuple(sorted(TEST_SPACE.ffn_kernels)))
        self.assertAlmostEqual(last.until, 1.0)

    def test_frontier_only_widens(self) -> None:
        stages = build_shrink_schedule(TEST_SPACE, 4)
        depths = [stage.min_depth for stage in stages]
        self.assertEqual(depths, sorted(depths, reverse=True))
        for smaller, bigger in zip(stages, stages[1:]):
            self.assertLessEqual(len(smaller.embed_dims), len(bigger.embed_dims))
            self.assertTrue(set(smaller.embed_dims) <= set(bigger.embed_dims))
            self.assertTrue(set(smaller.ffn_kernels) <= set(bigger.ffn_kernels))

    def test_early_stage_trains_only_the_largest_subnets(self) -> None:
        trainer = SandwichTrainer(
            TEST_SPACE, SandwichConfig(n_random=2, seed=0, shrink_stages=4)
        )
        trainer.total_steps = 100
        trainer.step_index = 0
        for _, spec, _ in trainer.build_sandwich():
            self.assertEqual(spec.depth, TEST_SPACE.max_depth)
            self.assertEqual(spec.embed_dim, TEST_SPACE.max_embed_dim)
            for layer in spec.layers:
                self.assertEqual(layer.ffn_ratio, max(TEST_SPACE.ffn_ratios))
                self.assertEqual(layer.ffn_kernel, max(TEST_SPACE.ffn_kernels))

    def test_last_stage_reaches_the_cheapest_subnet(self) -> None:
        trainer = SandwichTrainer(
            TEST_SPACE, SandwichConfig(n_random=0, seed=0, shrink_stages=4)
        )
        trainer.total_steps = 100
        trainer.step_index = 99
        _, minimum, _ = trainer.build_sandwich()[1]
        self.assertEqual(minimum, TEST_SPACE.min_subnet())

    def test_each_step_records_its_stage(self) -> None:
        trainer = SandwichTrainer(
            TEST_SPACE, SandwichConfig(n_random=1, seed=0, shrink_stages=2)
        )
        history = trainer.train(TEST_TASK.batch_fn(4, seed=0), steps=6)
        self.assertEqual(history[0].stage, 0)
        self.assertEqual(history[-1].stage, 1)


class DistillationTests(unittest.TestCase):
    def _record(self, distill_weight: float):
        trainer = SandwichTrainer(
            TEST_SPACE, SandwichConfig(n_random=1, seed=0, distill_weight=distill_weight)
        )
        return trainer.train_step(make_inputs(4), make_labels(4))

    def test_off_by_default(self) -> None:
        self.assertEqual(SandwichConfig().distill_weight, 0.0)

    def test_distillation_changes_only_the_non_teacher_legs(self) -> None:
        plain = self._record(0.0).role_losses
        distilled = self._record(0.5).role_losses
        self.assertAlmostEqual(plain["max"], distilled["max"], places=6)
        self.assertNotAlmostEqual(plain["random_0"], distilled["random_0"], places=4)

    def test_teacher_is_the_max_leg(self) -> None:
        trainer = SandwichTrainer(
            TEST_SPACE, SandwichConfig(n_random=1, seed=0, distill_weight=1.0)
        )
        inputs, labels = make_inputs(4), make_labels(4)
        before = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
        record = trainer.train_step(inputs, labels)
        trainer.model.load_state_dict(before)   # undo the step to recompute the teacher loss
        with torch.no_grad():
            expected = float(
                classification_loss(trainer.model(inputs, TEST_SPACE.max_subnet()), labels)
            )
        self.assertAlmostEqual(record.role_losses["max"], expected, places=4)


if __name__ == "__main__":
    unittest.main()
