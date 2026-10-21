"""Live routing boundaries, using invented model replies and no network."""

from copy import deepcopy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import live


class LiveTests(unittest.TestCase):
    def test_self_hosted_and_unrelated_endpoints_do_not_receive_cloud_environment_key(self):
        with patch.dict(live.os.environ, {"SIE_API_KEY": "invented-cloud-key", "SIE_BASE_URL": "https://api.superlinked.com"}):
            with live.sie_client("http://localhost:8080", self_hosted=True) as local:
                self.assertFalse(local._api_key)
            with live.sie_client("https://unrelated.example.test") as unrelated:
                self.assertFalse(unrelated._api_key)
            with live.sie_client("https://api.superlinked.com") as cloud:
                self.assertEqual(cloud._api_key, "invented-cloud-key")

    def test_always_arm_bypasses_encoder_and_native_inference(self):
        plan = {"pair": {"arms": {"cheap": {"model": "cheap"}, "strong": {"model": "strong"}}}}
        for method in ("embedding", "lora", "base"):
            with (
                patch.object(live, "checked_model", return_value=(plan, {"mode": "always_cheap"}, {})),
                patch.object(live, "embed") as embed,
                patch.object(live.paired_native, "checkpoint_probabilities") as native,
            ):
                result, _ = live.route("unused", "invented question", method, 1)
            self.assertEqual(result["arm"], "cheap")
            self.assertIsNone(result["embedding"])
            embed.assert_not_called()
            native.assert_not_called()

    def test_live_encoder_preserves_instruction_dimensions_and_normalization(self):
        client = MagicMock()
        client.get_model.return_value = {"revision": live.core.REVISION, "dims": {"dense": 2560}}
        client.last_model_revision = "e" * 64
        client.encode.return_value = [
            {
                "id": "new",
                "model": live.core.MODEL,
                "dense": [2.0] * 2560,
                "request": {"id": "request-1", "credits_debited": 2},
            }
        ]
        vector, receipt = live.embed(client, "invented question", "new")
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        self.assertEqual(receipt["observed_cloud_usd"], "0.00002")
        self.assertEqual(client.encode.call_args.kwargs["instruction"], live.core.INSTRUCTION)
        self.assertTrue(client.encode.call_args.kwargs["is_query"])
        client.get_model.return_value["revision"] = "different"
        with self.assertRaises(ValueError):
            live.embed(client, "invented question", "new")
        self.assertEqual(client.encode.call_count, 1)

    def test_selfhosted_encoder_does_not_claim_free_cloud_credits(self):
        client = MagicMock()
        client.get_model.return_value = {"revision": live.core.REVISION, "dims": {"dense": 2560}}
        client.encode.return_value = [{"id": "new", "model": live.core.MODEL, "dense": [1.0] * 2560}]
        _, receipt = live.embed(client, "invented question", "new", self_hosted=True)
        self.assertIsNone(receipt["observed_credits"])
        self.assertIsNone(receipt["observed_cloud_usd"])
        with self.assertRaises(ValueError):
            live.embed(client, "invented question", "new", self_hosted=False)

    def pair(self):
        return {
            "arms": {
                "cheap": {
                    "arm": "luna",
                    "model": "gpt-6-luna",
                    "settings": deepcopy(live.answer_models.OPENAI_SETTINGS),
                }
            },
            "request_builder_source_sha256": live.core.sha(live.answer_models.__file__),
            "prices_usd_per_million": deepcopy(live.answer_models.PRICES),
        }

    def item(self):
        return {"id": "invented", "request": "What is2+2?", "response_contract": {"type": "integer"}}

    def test_incomplete_answer_is_charged_once_and_not_retried(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.post.return_value.json.return_value = {
            "model": "gpt-6-luna",
            "service_tier": "default",
            "status": "incomplete",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 15}},
        }
        with (
            patch.object(live.httpx, "Client", return_value=client),
            patch.dict(live.os.environ, {"OPENAI_API_KEY": "invented"}),
        ):
            result = live.solve(self.item(), "cheap", self.pair())
        self.assertEqual(client.post.call_count, 1)
        self.assertIsNone(result["answer"])
        self.assertFalse(result["valid_answer"])
        self.assertGreater(float(result["cost"]["usd_estimate"]), 0)

    def test_answer_setting_change_rejected_before_network(self):
        pair = self.pair()
        pair["arms"]["cheap"]["settings"]["max_output_tokens"] = 16
        with patch.object(live.httpx, "Client") as client:
            with self.assertRaises(ValueError):
                live.solve(self.item(), "cheap", pair)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
