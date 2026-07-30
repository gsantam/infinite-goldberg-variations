from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from scripts.run_notagen_sft_epoch_sampling import (
    _chunk_generated_flat_ids,
    aggregate_meter_duration_ratio_monitor,
    evaluate_checkpoint_samples,
    meter_duration_ratio_counts_for_text,
    shuffled_prefix_specs_for_epoch,
    summarize_meter_duration_ratio_counts,
)


def test_meter_duration_ratio_counts_for_text_counts_half_and_exact_bars() -> None:
    abc_text = "\n".join(
        [
            "L:1/8",
            "M:3/4",
            "K:G",
            "V:1",
            "[r:0/1][V:1]C2 D|",
            "[r:1/0][V:1]C2 D2 E2|",
            "",
        ]
    )

    counts = meter_duration_ratio_counts_for_text(abc_text)

    assert counts[Fraction(1, 2)] == 1
    assert counts[Fraction(1, 1)] == 1


def test_summarize_meter_duration_ratio_counts_reports_key_fractions() -> None:
    summary = summarize_meter_duration_ratio_counts(
        Counter({Fraction(1, 2): 2, Fraction(1, 1): 3, Fraction(2, 1): 1})
    )

    assert summary["voice_bar_count"] == 6
    assert summary["half_bar_fraction"] == 2 / 6
    assert summary["exact_bar_fraction"] == 3 / 6
    assert summary["double_bar_fraction"] == 1 / 6
    assert summary["top_ratios"][0]["ratio"] == "1"


def test_aggregate_meter_duration_ratio_monitor_groups_by_prompt_meter_and_length() -> None:
    monitor = aggregate_meter_duration_ratio_monitor(
        [
            {
                "prefix_name": "variation-03_G",
                "prompt_meter": "3/4",
                "prompt_default_length": "1/16",
                "meter_duration_ratio_counts": {"1/2": 4, "1": 1},
            },
            {
                "prefix_name": "variation-04_G",
                "prompt_meter": "3/8",
                "prompt_default_length": "1/8",
                "meter_duration_ratio_counts": {"1": 5},
            },
        ]
    )

    assert monitor["overall"]["voice_bar_count"] == 10
    assert monitor["overall"]["half_bar_fraction"] == 0.4
    by_prompt = {row["name"]: row for row in monitor["by_prompt"]}
    assert by_prompt["variation-03_G"]["prompt_meter"] == "3/4"
    assert by_prompt["variation-03_G"]["half_bar_fraction"] == 0.8
    by_meter = {row["name"]: row for row in monitor["by_meter"]}
    assert by_meter["3/8"]["exact_bar_fraction"] == 1.0


def test_chunk_generated_flat_ids_completes_partial_prompt_patch_first() -> None:
    chunks = _chunk_generated_flat_ids(list(range(30)), prompt_token_count=10)

    assert [len(chunk) for chunk in chunks] == [6, 16, 8]
    assert [token for chunk in chunks for token in chunk] == list(range(30))


def test_prefix_shuffle_seed_keeps_eval_prompt_order_fixed_across_epochs() -> None:
    specs = [{"prefix": f"variation-{index:02d}.abc"} for index in range(10)]

    epoch_1 = shuffled_prefix_specs_for_epoch(specs, epoch=1, prefix_shuffle_seed=17)
    epoch_8 = shuffled_prefix_specs_for_epoch(specs, epoch=8, prefix_shuffle_seed=17)

    assert epoch_1 == epoch_8
    assert epoch_1 != specs


def test_prefix_shuffle_defaults_to_epoch_dependent_order() -> None:
    specs = [{"prefix": f"variation-{index:02d}.abc"} for index in range(10)]

    epoch_1 = shuffled_prefix_specs_for_epoch(specs, epoch=1, prefix_shuffle_seed=None)
    epoch_8 = shuffled_prefix_specs_for_epoch(specs, epoch=8, prefix_shuffle_seed=None)

    assert epoch_1 != epoch_8


def test_pretrained_baseline_zero_sample_row_has_same_summary_shape() -> None:
    args = SimpleNamespace(samples_per_epoch=0, prefix_shuffle_seed=0)

    row = evaluate_checkpoint_samples(
        args=args,
        row_type="pretrained_baseline",
        epoch=0,
        checkpoint=Path("base.pth"),
        checkpoint_is_rolling=False,
        rolling_checkpoint=None,
        epoch_checkpoint=None,
        losses=None,
        prefix_specs=None,
        variation_refs=[],
        exact_kl_reference_checkpoint=Path("base.pth"),
        max_generated_patches=1024,
        samples_dir=Path("samples"),
        logs_dir=Path("logs"),
        scores_dir=Path("scores"),
    )

    assert row["epoch"] == 0
    assert row["row_type"] == "pretrained_baseline"
    assert row["checkpoint"] == "base.pth"
    assert row["losses"] is None
    assert row["samples"] == []
    assert row["mean_reward"] is None
    assert row["mean_effective_similarity_reward"] is None
    assert row["prefix_shuffle_seed"] == 0
