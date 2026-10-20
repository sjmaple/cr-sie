"""The worker must not become healthy after a failed or partial preload."""

import unittest

import healthcheck


class Client:
    def __init__(self, missing=None, wrong_revision=False):
        self.missing = missing
        self.wrong_revision = wrong_revision
        self.calls = []

    def get_model(self, model):
        self.calls.append(model)
        revision = next(revision for rows in healthcheck.MODELS.values() for name, revision in rows if name == model)
        return {"name": model, "loaded": model != self.missing, "revision": "changed" if self.wrong_revision else revision,
                "dims": {"dense": 2560}}


class HealthcheckTests(unittest.TestCase):
    def test_cheap_requires_both_loaded_profiles(self):
        client = Client(missing="Qwen/Qwen3.5-4B:no-spec")
        with self.assertRaises(RuntimeError):
            healthcheck.check(client, "cheap")
        self.assertEqual(client.calls, [model for model, _ in healthcheck.MODELS["cheap"]])

    def test_failed_encoder_preload_does_not_pass(self):
        with self.assertRaises(RuntimeError):
            healthcheck.check(Client(missing="Qwen/Qwen3-Embedding-4B"), "encoder")

    def test_wrong_revision_does_not_pass(self):
        with self.assertRaises(RuntimeError):
            healthcheck.check(Client(wrong_revision=True), "strong")

    def test_wrong_profile_with_same_revision_does_not_pass(self):
        class WrongProfileClient(Client):
            def get_model(self, model):
                return {**super().get_model(model), "name": "Qwen/Qwen3.5-4B"}

        with self.assertRaises(RuntimeError):
            healthcheck.check(WrongProfileClient(), "cheap")

    def test_pinned_loaded_profiles_pass(self):
        for arm in healthcheck.MODELS:
            healthcheck.check(Client(), arm)


if __name__ == "__main__":
    unittest.main()
