import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    exact_categorical_kl = None
    IMPORT_ERROR = exc
else:
    from notagen_runtime.notagen_replay import exact_categorical_kl

    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"torch unavailable: {IMPORT_ERROR}")
class NotaGenReplayUtilsTests(unittest.TestCase):
    def test_exact_categorical_kl_is_zero_for_identical_distributions(self):
        logits = torch.tensor([[0.0, 1.0, -2.0], [3.0, -1.0, 0.5]])
        log_dists = logits.log_softmax(dim=-1)
        actual = exact_categorical_kl(log_dists, log_dists)
        self.assertLess(abs(float(actual)), 1e-7)

    def test_exact_categorical_kl_matches_manual_full_vocab_formula(self):
        policy_logits = torch.tensor([[0.0, 1.0, -2.0], [3.0, -1.0, 0.5]])
        reference_logits = torch.tensor([[0.3, 0.7, -1.0], [2.0, -0.5, 1.5]])
        policy_log_dists = policy_logits.log_softmax(dim=-1)
        reference_log_dists = reference_logits.log_softmax(dim=-1)
        expected = (
            policy_log_dists.exp() * (policy_log_dists - reference_log_dists)
        ).sum(dim=-1).mean()
        actual = exact_categorical_kl(policy_log_dists, reference_log_dists)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
