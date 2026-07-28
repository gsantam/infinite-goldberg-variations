from __future__ import annotations

import unittest

try:
    from notagen_runtime.notagen_wrapper import PATCH_SIZE, fit_repatch_context_to_limit
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local torch install
    PATCH_SIZE = 16
    fit_repatch_context_to_limit = None
    _IMPORT_SKIP_REASON = f"notagen wrapper dependencies unavailable: {exc}"
else:
    _IMPORT_SKIP_REASON = ""


class _FakePatchilizer:
    def encode_generate(self, text: str) -> list[list[int]]:
        flat = [ord(char) % 128 for char in text]
        return [flat[index : index + PATCH_SIZE] for index in range(0, len(flat), PATCH_SIZE)]


@unittest.skipIf(fit_repatch_context_to_limit is None, _IMPORT_SKIP_REASON)
class RolloverContextTest(unittest.TestCase):
    def test_fit_repatch_context_shrinks_to_safe_full_patch_limit(self) -> None:
        text = "\n".join(
            [
                "X:1",
                "M:3/4",
                "L:1/8",
                "K:G",
                "[r:0/3][V:1]" + "A" * 40,
                "[r:1/2][V:1]" + "B" * 40,
                "[r:2/1][V:1]" + "C" * 40,
                "[r:3/0][V:1]" + "D" * 40,
                "",
            ]
        )

        flat_ids, cut_index = fit_repatch_context_to_limit(
            text,
            _FakePatchilizer(),
            max_context_tokens=64,
            preferred_cut_index=4,
        )

        self.assertLess(cut_index, 4)
        self.assertLessEqual(len(flat_ids), 64)
        self.assertEqual(len(flat_ids) % PATCH_SIZE, 0)
        self.assertGreaterEqual(len(flat_ids), PATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
