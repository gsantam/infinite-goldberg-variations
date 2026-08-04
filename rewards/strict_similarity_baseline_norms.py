from __future__ import annotations

# Global base NotaGen-large normalization constants for the 21 Aria-matching
# PPO prompts. These are computed from 198 successful base-model samples after
# merging 216 total sampled rows: an initial 21 * 8 baseline run with 150
# successful rows plus a 48-successful-row top-up for underrepresented prompts.
#
# Source rows:
# data/processed/notagen/remote_runs/SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z/current_reward_rescore_20260803_aria_matching_bars/base_strict_similarity_norms_base8_20260804T090153Z
# data/processed/notagen/remote_runs/SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z/current_reward_rescore_20260803_aria_matching_bars/base_strict_similarity_norms_topup_20260804T
# data/processed/notagen/remote_runs/SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z/current_reward_rescore_20260803_aria_matching_bars/base_strict_similarity_norms_merged_20260804T
#
# Initial reproduction command:
# python scripts/sample_base_strict_similarity_norms.py \
#   --eligible-prompts data/processed/goldberg/structure/aria_matching_prompt_names.txt \
#   --samples-per-prompt 8
#
# The raw metrics live in rewards.strict_similarity; these constants
# only calibrate "how many base-model standard deviations above baseline" a
# generated sample is.

STRICT_SIMILARITY_BASELINE_METRICS = (
    "strict_symbolic_similarity",
    "strict_aligned_root_bass",
    "strict_cadence_root_bass",
    "strict_root_bass_bigram_weighted_jaccard",
    "strict_root_bass_fourgram_weighted_jaccard",
    "strict_dtw_combined_narrow",
)

STRICT_SIMILARITY_BASELINE_PROVENANCE = {
    "eligible_prompt_count": 21,
    "total_sample_rows": 216,
    "successful_sample_rows": 198,
    "failed_sample_rows": 18,
    "initial_base8_rows": 168,
    "initial_base8_successful_rows": 150,
    "topup_rows": 48,
    "topup_successful_rows": 48,
    "source_dirs": (
        "data/processed/notagen/remote_runs/SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z/current_reward_rescore_20260803_aria_matching_bars/base_strict_similarity_norms_base8_20260804T090153Z",
        "data/processed/notagen/remote_runs/SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z/current_reward_rescore_20260803_aria_matching_bars/base_strict_similarity_norms_topup_20260804T",
        "data/processed/notagen/remote_runs/SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z/current_reward_rescore_20260803_aria_matching_bars/base_strict_similarity_norms_merged_20260804T",
    ),
}

STRICT_SIMILARITY_GLOBAL_NORMS = {
    "strict_symbolic_similarity": {
        "mean": 0.33856960756146337,
        "std": 0.04654469368865775,
        "std_safe": 0.04654469368865775,
        "n": 198,
    },
    "strict_aligned_root_bass": {
        "mean": 0.30784406565656564,
        "std": 0.07221815834544558,
        "std_safe": 0.07221815834544558,
        "n": 198,
    },
    "strict_cadence_root_bass": {
        "mean": 0.4163510101010101,
        "std": 0.22060392252517244,
        "std_safe": 0.22060392252517244,
        "n": 198,
    },
    "strict_root_bass_bigram_weighted_jaccard": {
        "mean": 0.05956008944453005,
        "std": 0.03285211198816913,
        "std_safe": 0.03285211198816913,
        "n": 198,
    },
    "strict_root_bass_fourgram_weighted_jaccard": {
        "mean": 0.0012468038783828258,
        "std": 0.005197175911981873,
        "std_safe": 0.005197175911981873,
        "n": 198,
    },
    "strict_dtw_combined_narrow": {
        "mean": 0.7699289935349171,
        "std": 0.02106780939958876,
        "std_safe": 0.02106780939958876,
        "n": 198,
    },
}

# Sparse n-gram metrics can have near-zero empirical std. Use these floors only
# when converting components to z-scores; the raw constants above preserve the
# measured std.
STRICT_SIMILARITY_Z_STD_FLOORS = {
    "strict_symbolic_similarity": 1e-4,
    "strict_aligned_root_bass": 1e-4,
    "strict_cadence_root_bass": 1e-4,
    "strict_root_bass_bigram_weighted_jaccard": 1e-4,
    "strict_root_bass_fourgram_weighted_jaccard": 0.02,
    "strict_dtw_combined_narrow": 1e-4,
}
