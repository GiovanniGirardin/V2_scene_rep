"""Paper-level invariants for the MST and SLT implementations.

Run without extra test dependencies:
    venv/bin/python -m unittest tests/test_scene_rep_components.py
"""

from __future__ import annotations

import copy
import unittest

import torch

from scene_rep.models.mst_encoder import MSTEncoder
from scene_rep.models.slt import SequentialLatentTransformer


def small_config() -> dict:
    return {
        "model": {
            "motion_dim": 5,
            "waypoint_dim": 3,
            "d_model": 16,
            "latent_dim": 16,
            "nhead": 4,
            "dropout": 0.0,
        },
        "observation": {
            "max_neighbors": 2,
            "max_candidate_routes": 2,
            "waypoint_len": 3,
        },
    }


def observation() -> dict[str, torch.Tensor]:
    """One batch with padded history, padded routes, and an absent agent."""
    torch.manual_seed(7)
    b, a, h, r, w = 2, 3, 4, 2, 3
    motion = torch.zeros(b, a, h, 5)
    motion_mask = torch.zeros(b, a, h)
    motion[:, 0, -2:] = torch.randn(b, 2, 5)
    motion[:, 1, -1:] = torch.randn(b, 1, 5)
    motion_mask[:, 0, -2:] = 1
    motion_mask[:, 1, -1:] = 1

    waypoints = torch.zeros(b, a, r, w, 3)
    waypoint_mask = torch.zeros(b, a, r, w)
    waypoints[:, 0, 0] = torch.randn(b, w, 3)
    waypoints[:, 0, 1, :2] = torch.randn(b, 2, 3)
    waypoints[:, 1, 0] = torch.randn(b, w, 3)
    waypoint_mask[:, 0, 0] = 1
    waypoint_mask[:, 0, 1, :2] = 1
    waypoint_mask[:, 1, 0] = 1

    return {
        "motion": motion,
        "motion_mask": motion_mask,
        "waypoints": waypoints,
        "waypoint_mask": waypoint_mask,
        "agent_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32),
        "route_mask": torch.tensor(
            [[[1, 1], [1, 0], [0, 0]], [[1, 1], [1, 0], [0, 0]]],
            dtype=torch.float32,
        ),
    }


class Capture(torch.nn.Module):
    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block
        self.query: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, context: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.query = query.detach().clone()
        self.output = self.block(query, context, **kwargs)
        return self.output


class MSTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = MSTEncoder(small_config()).eval()

    def test_shape_and_padding_invariance(self) -> None:
        obs = observation()
        reference = self.model(obs)
        self.assertEqual(tuple(reference.shape), (2, 16))
        self.assertTrue(torch.isfinite(reference).all())

        changed = copy.deepcopy(obs)
        changed["motion"][changed["motion_mask"] == 0] = 1e4
        changed["waypoints"][changed["waypoint_mask"] == 0] = -1e4
        actual = self.model(changed)
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)

    def test_aggregation_queries_ego_motion_not_ego_route(self) -> None:
        spy = Capture(self.model.ego_agent_cross)
        self.model.ego_agent_cross = spy
        obs = observation()
        self.model(obs)
        query_before = spy.query

        changed = copy.deepcopy(obs)
        # Change only valid ego route points. Eq. (9) requires the aggregation
        # query to remain D^ego_M, not the ego cross-modal route feature.
        changed["waypoints"][:, 0, :, :, :] += 100.0
        self.model(changed)
        torch.testing.assert_close(spy.query, query_before, rtol=0, atol=1e-6)

    def test_output_level_has_aggregation_residual(self) -> None:
        aggregation_spy = Capture(self.model.ego_agent_cross)
        output_spy = Capture(self.model.ego_route_cross)
        self.model.ego_agent_cross = aggregation_spy
        self.model.ego_route_cross = output_spy
        latent = self.model(observation())
        self.assertIsNotNone(aggregation_spy.output)
        self.assertIsNotNone(output_spy.output)
        expected = output_spy.output.squeeze(1) + aggregation_spy.output.squeeze(1)
        torch.testing.assert_close(latent, expected, rtol=0, atol=1e-6)


class SLTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.model = SequentialLatentTransformer(
            latent_dim=16,
            action_dim=2,
            future_horizon=3,
            nhead=4,
            dropout=0.0,
            projector_dim=12,
            predictor_dim=99,
        ).eval()

    def test_predictor_and_sequence_alignment(self) -> None:
        self.assertIsInstance(self.model.predictor, torch.nn.Linear)
        latents = torch.randn(2, 4, 16)
        actions = torch.randn(2, 4, 2)
        prediction = self.model(latents, actions)
        self.assertEqual(tuple(prediction.shape), (2, 3, 16))

    def test_causality_and_similarity_gradient(self) -> None:
        latents = torch.randn(2, 4, 16, requires_grad=True)
        actions = torch.randn(2, 4, 2)
        prediction = self.model(latents, actions)
        altered = latents.detach().clone()
        altered[:, 3] += 1e3
        altered_prediction = self.model(altered, actions)
        # The final input is not available to any returned prediction.
        torch.testing.assert_close(prediction, altered_prediction, rtol=0, atol=1e-6)

        loss, _ = self.model.compute_loss(latents, actions)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.isfinite(latents.grad).all())


if __name__ == "__main__":
    unittest.main()
