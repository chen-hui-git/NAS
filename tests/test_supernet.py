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
    CharLMTask,
    InductionTask,
    InductionTaskConfig,
    LayerChoice,
    RankingExperimentConfig,
    SandwichConfig,
    SandwichTrainer,
    SearchSpace,
    SubnetSpec,
    Supernet,
    average_subnet_loss,
    build_dense_subnet,
    induction_batch,
    kendall_tau,
    masked_next_token_loss,
    run_ranking_experiment,
    spearman_rho,
)

TEST_SPACE = SearchSpace(
    min_depth=2,
    max_depth=4,
    embed_dims=(16, 32),
    head_dims=(8, 16),
    ffn_ratios=(2, 3),
    ffn_kernels=(1, 3),
    activations=("gelu", "silu"),
    vocab_size=16,
    max_seq_len=12,
)

TEST_CONFIG = InductionTaskConfig(num_pairs=2, num_keys=6, num_values=4)
TEST_TASK = InductionTask(TEST_CONFIG)


def make_tokens(batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 16, (batch_size, TEST_CONFIG.num_tokens), generator=generator)


class SearchSpaceTests(unittest.TestCase):
    def test_rejects_depth_outside_the_space(self) -> None:
        spec = SubnetSpec(
            depth=5,
            embed_dim=16,
            head_dim=8,
            layers=tuple(LayerChoice(2, 1, "gelu") for _ in range(5)),
        )
        with self.assertRaises(ValueError):
            spec.validate(TEST_SPACE)

    def test_rejects_unknown_branch(self) -> None:
        spec = SubnetSpec(
            depth=2,
            embed_dim=16,
            head_dim=8,
            layers=(LayerChoice(5, 1, "gelu"), LayerChoice(2, 1, "gelu")),
        )
        with self.assertRaises(ValueError):
            spec.validate(TEST_SPACE)

    def test_rejects_layer_count_mismatch(self) -> None:
        spec = SubnetSpec(
            depth=3,
            embed_dim=16,
            head_dim=8,
            layers=tuple(LayerChoice(2, 1, "gelu") for _ in range(2)),
        )
        with self.assertRaises(ValueError):
            spec.validate(TEST_SPACE)

    def test_rejects_non_divisible_head_dim(self) -> None:
        spec = SubnetSpec(
            depth=2,
            embed_dim=16,
            head_dim=32,
            layers=tuple(LayerChoice(2, 1, "gelu") for _ in range(2)),
        )
        with self.assertRaises(ValueError):
            spec.validate(TEST_SPACE)

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
        self.assertNotEqual(left, [TEST_SPACE.sample(random.Random(4)).to_dict() for _ in range(5)])


class SupernetForwardTests(unittest.TestCase):
    def test_extreme_subnets_produce_logits(self) -> None:
        model = Supernet(TEST_SPACE)
        tokens = make_tokens()
        for spec in (TEST_SPACE.max_subnet(), TEST_SPACE.min_subnet()):
            logits = model(tokens, spec)
            self.assertEqual(
                logits.shape, (tokens.shape[0], TEST_SPACE.vocab_size, tokens.shape[1])
            )
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
        tokens = make_tokens()
        with torch.no_grad():
            before = model(tokens, spec)
            model.layers[0].ffn.branches["r3k3"].conv1.weight.add_(1.0)
            model.layers[0].ffn.branches["r3k3"].conv2.bias.add_(1.0)
            after = model(tokens, spec)
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
        tokens = make_tokens()
        with torch.no_grad():
            before = model(tokens, spec)
            model.layers[3].attn.qkv.weight.add_(1.0)
            model.layers[3].ffn.branches["r2k1"].conv1.weight.add_(1.0)
            after = model(tokens, spec)
        self.assertTrue(torch.equal(before, after))

    def test_logits_never_depend_on_future_tokens(self) -> None:
        """Causality regression: a kernel>1 Conv1d must not read token t+1."""
        torch.manual_seed(0)
        model = Supernet(TEST_SPACE)
        spec = TEST_SPACE.max_subnet()
        self.assertGreater(max(spec_layer.ffn_kernel for spec_layer in spec.layers), 1)
        tokens = make_tokens()
        cut = 5
        with torch.no_grad():
            baseline = model(tokens, spec)
            perturbed_tokens = tokens.clone()
            perturbed_tokens[:, cut] = (perturbed_tokens[:, cut] + 1) % TEST_SPACE.vocab_size
            perturbed = model(perturbed_tokens, spec)
        self.assertTrue(
            torch.allclose(baseline[:, :, :cut], perturbed[:, :, :cut], atol=1e-6),
            msg="logits before the changed token must not move",
        )
        self.assertFalse(
            torch.allclose(baseline[:, :, cut], perturbed[:, :, cut], atol=1e-6),
            msg="the changed token must affect its own position",
        )

    def test_export_subnet_matches_shared_forward(self) -> None:
        torch.manual_seed(1)
        model = Supernet(TEST_SPACE)
        tokens = make_tokens()
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
                shared = model(tokens, spec)
                separate = exported(tokens, spec)
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
        tokens = make_tokens()
        logits = model(tokens, spec)
        self.assertEqual(
            logits.shape, (tokens.shape[0], TEST_SPACE.vocab_size, tokens.shape[1])
        )


class LossTests(unittest.TestCase):
    def test_masked_mean_matches_uniform_prediction(self) -> None:
        logits = torch.zeros(2, 16, TEST_CONFIG.num_tokens)
        tokens = make_tokens()
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[0, 1] = True
        loss = masked_next_token_loss(logits, tokens, mask)
        self.assertAlmostEqual(float(loss), math.log(16), places=5)

    def test_unmasked_positions_do_not_change_the_loss(self) -> None:
        logits = torch.zeros(2, 16, TEST_CONFIG.num_tokens)
        tokens = make_tokens()
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[0, 2] = True
        baseline = masked_next_token_loss(logits, tokens, mask)
        perturbed = logits.clone()
        perturbed[1] += 5.0
        self.assertTrue(torch.allclose(baseline, masked_next_token_loss(perturbed, tokens, mask)))


class InductionTaskTests(unittest.TestCase):
    def test_scored_positions_predict_the_paired_value(self) -> None:
        config = InductionTaskConfig(num_pairs=3, num_keys=8, num_values=4)
        generator = torch.Generator().manual_seed(11)
        tokens, targets, mask = induction_batch(config, 4, generator)
        self.assertTrue(torch.equal(tokens, targets))
        self.assertEqual(tokens.shape, (4, config.num_tokens))
        pairs = config.num_pairs
        for sample in range(4):
            lookup = {}
            for pair in range(pairs):
                key = int(tokens[sample, 2 * pair])
                value = int(tokens[sample, 2 * pair + 1])
                lookup[key] = value
            scored = mask[sample].nonzero().flatten().tolist()
            self.assertEqual(len(scored), pairs)
            for position in scored:
                self.assertGreaterEqual(position, 2 * pairs)
                key = int(tokens[sample, position])
                self.assertEqual(int(tokens[sample, position + 1]), lookup[key])

    def test_batches_are_reproducible_from_a_seed(self) -> None:
        left = TEST_TASK.batch_fn(3, seed=5)
        right = TEST_TASK.batch_fn(3, seed=5)
        for step in range(3):
            for a, b in zip(left(step), right(step)):
                self.assertTrue(torch.equal(a, b))


class SandwichRuleTests(unittest.TestCase):
    def test_sandwich_legs_contain_the_extremes(self) -> None:
        trainer = SandwichTrainer(TEST_SPACE, SandwichConfig(n_random=2, seed=0))
        legs = trainer.build_sandwich()
        roles = [role for role, _, _ in legs]
        self.assertEqual(roles[0], "max")
        self.assertEqual(roles[1], "min")
        self.assertEqual(len(roles), 4)
        self.assertEqual(legs[0][1], TEST_SPACE.max_subnet())
        self.assertEqual(legs[1][1], TEST_SPACE.min_subnet())

    def test_training_step_touches_max_min_and_random_branches(self) -> None:
        trainer = SandwichTrainer(TEST_SPACE, SandwichConfig(n_random=2, seed=1))
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.model.named_parameters()
        }
        batch_fn = TEST_TASK.batch_fn(2, seed=0)
        tokens, targets, mask = batch_fn(0)
        record = trainer.train_step(tokens, targets, mask)
        self.assertEqual(set(record.role_losses), {"max", "min", "random_0", "random_1"})
        self.assertGreater(record.grad_norm, 0.0)
        # The deepest layer is only reachable through the max leg.
        deepest = f"layers.{TEST_SPACE.max_depth - 1}.attn.qkv.weight"
        self.assertFalse(torch.equal(before[deepest], trainer.model.state_dict()[deepest]))

    def test_coverage_reaches_every_depth_and_kernel(self) -> None:
        trainer = SandwichTrainer(TEST_SPACE, SandwichConfig(n_random=3, seed=2))
        batch_fn = TEST_TASK.batch_fn(2, seed=0)
        trainer.train(batch_fn, steps=20)
        self.assertEqual(
            set(trainer.stats.depth_counts),
            set(range(TEST_SPACE.min_depth, TEST_SPACE.max_depth + 1)),
        )
        self.assertEqual(set(trainer.stats.kernel_counts), set(TEST_SPACE.ffn_kernels))
        self.assertEqual(trainer.stats.subnets_trained, 20 * 5)

    def test_training_reduces_the_max_subnet_loss(self) -> None:
        trainer = SandwichTrainer(
            TEST_SPACE, SandwichConfig(n_random=1, seed=3, lr=5e-3, weight_decay=0.0)
        )
        batch_fn = TEST_TASK.batch_fn(8, seed=4)
        spec = TEST_SPACE.max_subnet()
        batches = TEST_TASK.fixed_batches(8, 4, seed=9)
        before = average_subnet_loss(trainer.model, spec, batches)
        trainer.train(batch_fn, steps=30)
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
            min_depth=2,
            max_depth=3,
            embed_dims=(16,),
            head_dims=(8,),
            ffn_ratios=(2,),
            ffn_kernels=(1, 3),
            activations=("gelu",),
            vocab_size=16,
            max_seq_len=12,
        )
        model = Supernet(space)
        config = RankingExperimentConfig(
            num_candidates=3, standalone_steps=3, eval_batches=2, batch_size=2, seed=0
        )
        result = run_ranking_experiment(model, space, TEST_TASK, config)
        self.assertEqual(result["num_candidates"], 3)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertGreaterEqual(result["kendall_tau"], -1.0)
        self.assertLessEqual(result["kendall_tau"], 1.0)
        self.assertGreaterEqual(result["regret"], 0.0)
        ranks = sorted(row["supernet_rank"] for row in result["candidates"])
        self.assertEqual(ranks, [1, 2, 3])


class CharLanguageTaskTests(unittest.TestCase):
    """Uses the public corpus when it is present; never touches the network here."""

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        corpus = Path(__file__).resolve().parents[1] / "data" / "tiny_shakespeare.txt"
        if not corpus.exists():
            raise unittest.SkipTest("tiny shakespeare corpus not cached locally")
        cls.task = CharLMTask(seq_len=16, verbose=False)

    def test_vocabulary_covers_the_corpus(self) -> None:
        self.assertEqual(self.task.vocab_size, len(self.task.chars))
        self.assertGreater(self.task.vocab_size, 40)

    def test_mask_scores_every_position_but_the_last(self) -> None:
        tokens, targets, mask = self.task.batch(3)
        self.assertTrue(torch.equal(tokens, targets))
        self.assertEqual(tokens.shape, (3, 16))
        self.assertTrue(bool(mask[:, :-1].all()))
        self.assertFalse(bool(mask[:, -1].any()))

    def test_valid_batches_come_from_the_validation_split(self) -> None:
        tokens, _, _ = self.task.fixed_batches(2, 1, seed=0)[0]
        val = set(int(value) for value in self.task.val_ids[:200000].tolist())
        train = set(int(value) for value in self.task.train_ids[:200000].tolist())
        self.assertTrue(set(int(value) for value in tokens.flatten().tolist()) <= val)
        self.assertNotEqual(val, train)

    def test_batches_are_reproducible(self) -> None:
        left = self.task.batch_fn(2, seed=7)
        right = self.task.batch_fn(2, seed=7)
        for step in range(2):
            for a, b in zip(left(step), right(step)):
                self.assertTrue(torch.equal(a, b))


class GenerationTests(unittest.TestCase):
    def test_generate_extends_the_prompt(self) -> None:
        torch.manual_seed(0)
        model = Supernet(TEST_SPACE)
        spec = TEST_SPACE.min_subnet()
        prompt = make_tokens(1)[0]
        out = model.generate(spec, prompt, max_new_tokens=5)
        self.assertEqual(out.shape, (1, prompt.numel() + 5))
        self.assertTrue(torch.equal(out[0, : prompt.numel()], prompt))

    def test_generate_restores_training_mode(self) -> None:
        model = Supernet(TEST_SPACE)
        model.train()
        model.generate(TEST_SPACE.min_subnet(), make_tokens(1)[0], max_new_tokens=2)
        self.assertTrue(model.training)


class TaskInterfaceTests(unittest.TestCase):
    def test_induction_task_exposes_shapes(self) -> None:
        self.assertEqual(TEST_TASK.vocab_size, TEST_CONFIG.vocab_size)
        self.assertEqual(TEST_TASK.seq_len, TEST_CONFIG.num_tokens)
        tokens, _, mask = TEST_TASK.batch(2)
        self.assertEqual(tokens.shape, (2, TEST_CONFIG.num_tokens))
        self.assertEqual(mask.dtype, torch.bool)

    def test_unknown_task_name_is_rejected(self) -> None:
        from supernet import make_task

        with self.assertRaises(ValueError):
            make_task("mnist-tokens")


class ReportTests(unittest.TestCase):
    def test_snippets_come_from_the_real_source(self) -> None:
        from supernet.report import read_snippet

        code = read_snippet("supernet/modules.py", "elastic-conv")
        self.assertIn("def forward", code)
        self.assertIn("F.conv1d", code)
        self.assertNotIn("snippet:", code)
        self.assertIn("self.weight[:out_channels, :in_channels]", code)

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


if __name__ == "__main__":
    unittest.main()
