"""Analytic and shape tests for the balanced-partition estimand."""
import unittest
import numpy as np
import torch
from common import *
from prepare import make_group, NOUN_CANDIDATES, ADJECTIVE_CANDIDATES, validate_row
from run import conditions
from analyze import bootstrap_draws, estimate, components

class BalancedPartitionTests(unittest.TestCase):
    def setUp(self):
        self.row = make_group(0, "toy", 731, NOUN_CANDIDATES, ADJECTIVE_CANDIDATES)
        self.rng = np.random.default_rng(47)

    def test_each_dose_is_partition_of_same_donor_grid(self):
        expected = set(range(64))
        for k in DOSES:
            observed = []
            for bank in [b for b in self.row["banks"] if b["k"] == k]:
                observed += [slot * 8 + column for slot, column in enumerate(bank["columns"])]
                self.assertEqual(len(set(bank["columns"])), k)
                self.assertEqual(sorted(np.bincount(bank["columns"])[np.bincount(bank["columns"]) > 0]), [8 // k] * k)
            self.assertEqual(set(observed), expected)
            self.assertEqual(len(observed), 64)

    def test_global_resample_and_centroid_are_invariant(self):
        x = self.rng.normal(size=(130, 5))
        names, vectors, recipients, banks = conditions(torch.tensor(x), self.row)
        # A deliberately nonlinear downstream function; global resample equality
        # is due to the partition, not to an affine model assumption.
        nonlinear = lambda a: np.exp(.1 * a).sum(axis=-1)
        for direction in ["restore", "destroy"]:
            pooled_values, grand_centroids = [], []
            for k in DOSES:
                bs = [b for b in banks if b["direction"] == direction and b["k"] == k]
                pooled_values.append(np.mean([nonlinear(vectors[b["indices"]].numpy()).mean() for b in bs]))
                grand_centroids.append(np.mean([vectors[b["mean"]].numpy() for b in bs], axis=0))
            np.testing.assert_allclose(pooled_values, pooled_values[0], rtol=0, atol=1e-14)
            np.testing.assert_allclose(grand_centroids, np.broadcast_to(grand_centroids[0], (4, 5)), rtol=0, atol=1e-14)

    def test_affine_map_commutes_bankwise(self):
        x = self.rng.normal(size=(130, 5))
        _, vectors, _, banks = conditions(torch.tensor(x), self.row)
        weights = self.rng.normal(size=5)
        output = vectors.numpy() @ weights + 3.2
        for bank in banks:
            self.assertAlmostEqual(output[bank["mean"]], output[bank["indices"]].mean(), places=12)

    def test_fixed_support_different_grouping_changes_quadratic_gap(self):
        x = np.zeros((130, 1))
        for i in range(8):
            for j in range(8):
                for side in [0, 1]:
                    x[self.row["grid"][i][j][side]] = j - 3.5
        _, vectors, _, banks = conditions(torch.tensor(x), self.row)
        output = vectors.numpy()[:, 0] ** 2
        gaps = {k: np.mean([abs(output[b["mean"]] - output[b["indices"]].mean()) for b in banks if b["direction"] == "restore" and b["k"] == k]) for k in DOSES}
        self.assertEqual(gaps[1], 0.)
        self.assertEqual(gaps[8], 5.25)

    def test_causal_prefix_grid_leaves_all_anchor_banks_equal(self):
        contexts = self.rng.normal(size=(8, 5))
        x = self.rng.normal(size=(130, 5))
        for i in range(8):
            for j in range(8):
                for side in [0, 1]:
                    x[self.row["grid"][i][j][side]] = contexts[i] + side
        _, vectors, _, banks = conditions(torch.tensor(x), self.row)
        for direction in ["restore", "destroy"]:
            means = [vectors[b["mean"]].numpy() for b in banks if b["direction"] == direction]
            np.testing.assert_array_equal(means, np.broadcast_to(means[0], (32, 5)))

    def test_primary_indexing_preserves_recipient_axis(self):
        a = self.rng.normal(size=(64, 8, 2, 4, 8, 2))
        end = np.take(a, [1, 3, 5], axis=1)
        primary = abs(end[:, :, :, -1, :, 0]) - abs(end[:, :, :, 0, :, 0])
        self.assertEqual(primary.shape, (64, 3, 2, 8))
        reference = np.array([np.mean([abs(a[g, s, d, 3, r, 0]) - abs(a[g, s, d, 0, r, 0]) for s in [1, 3, 5] for d in range(2) for r in range(8)]) for g in range(64)])
        np.testing.assert_allclose(primary.mean(axis=(1, 2, 3)), reference)

    def test_group_bootstrap_does_not_inflate_sample_size(self):
        rows = [{"template": i % 2, "class": (i // 2) % 2} for i in range(64)]
        draws = bootstrap_draws(rows, count=31)
        self.assertEqual(draws.shape, (31, 64))
        values = np.broadcast_to(np.arange(64.)[:, None, None], (64, 3, 8))
        result = estimate(values, draws, alpha=.025)
        self.assertEqual(result["n_recipient_groups"], 64)
        self.assertEqual(result["confidence"], .975)
        self.assertEqual(result["mean"], 31.5)

    def test_saved_generated_design_invariants(self):
        for split in ["calibration", "development", "test"]:
            for row in read(ROOT / f"data/{split}.json"):
                self.assertTrue(validate_row(row))

if __name__ == "__main__":
    unittest.main()
