"""Optional Apple GPU check for dropout gradient correctness; no models are downloaded."""

import unittest

from finetune import runtime_settings


class MPSTrainingTests(unittest.TestCase):
    def test_runtime_profile_preserves_dropout_gradients(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("Install the optional training dependencies to check MPS gradients")
        if not torch.backends.mps.is_available():
            self.skipTest("An available Apple MPS GPU is required")

        from torch.utils.checkpoint import checkpoint

        def forward_backward(use_checkpoint):
            torch.manual_seed(89)
            model = torch.nn.Sequential(
                torch.nn.Linear(32, 32),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(32, 8),
            ).to(device="mps", dtype=torch.float32)
            inputs = torch.linspace(-1, 1, 256, dtype=torch.float32).reshape(8, 32).to("mps").requires_grad_()
            # Reset dropout RNG after identical parameter initialization in both runs.
            torch.manual_seed(137)
            output = checkpoint(model, inputs, use_reentrant=False) if use_checkpoint else model(inputs)
            output.square().mean().backward()
            gradients = {"inputs": inputs.grad.detach().cpu()}
            gradients.update({name: parameter.grad.detach().cpu() for name, parameter in model.named_parameters()})
            return output.detach().cpu(), gradients

        expected_output, expected_gradients = forward_backward(False)
        actual_output, actual_gradients = forward_backward(runtime_settings("mps")["gradient_checkpointing"])
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
        for name, expected in expected_gradients.items():
            with self.subTest(gradient=name):
                self.assertTrue(torch.isfinite(expected).all().item())
                self.assertGreater(expected.abs().sum().item(), 0)
                # Torch 2.9.1 can reproduce the forward output while checkpoint
                # recomputation uses different MPS dropout masks in backward.
                torch.testing.assert_close(actual_gradients[name], expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
