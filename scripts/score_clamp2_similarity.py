#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig


def load_clamp2(clamp2_dir: Path):
    sys.path.insert(0, str(clamp2_dir))
    from config import (  # type: ignore
        CLAMP2_HIDDEN_SIZE,
        CLAMP2_LOAD_M3,
        CLAMP2_WEIGHTS_PATH,
        M3_HIDDEN_SIZE,
        PATCH_LENGTH,
        PATCH_NUM_LAYERS,
        PATCH_SIZE,
        TEXT_MODEL_NAME,
    )
    from utils import CLaMP2Model, M3Patchilizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m3_config = BertConfig(
        vocab_size=1,
        hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE // 64,
        intermediate_size=M3_HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )
    model = CLaMP2Model(
        m3_config,
        text_model_name=TEXT_MODEL_NAME,
        hidden_size=CLAMP2_HIDDEN_SIZE,
        load_m3=CLAMP2_LOAD_M3,
    ).to(device)
    checkpoint = torch.load(clamp2_dir / CLAMP2_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, M3Patchilizer(), device, PATCH_LENGTH, PATCH_SIZE


@torch.no_grad()
def music_feature(path: Path, model, patchilizer, device, patch_length: int, patch_size: int) -> tuple[np.ndarray, int]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    item = "".join(line for line in lines if not (line.startswith("%") and not line.startswith("%%")))
    input_data = torch.tensor(patchilizer.encode(item, add_special_patches=True), dtype=torch.long)

    segment_list = [input_data[i : i + patch_length] for i in range(0, len(input_data), patch_length)]
    if not segment_list:
        segment_list = [input_data]
    segment_list[-1] = input_data[-patch_length:]

    features = []
    weights = []
    for segment in segment_list:
        mask = torch.tensor([1] * segment.size(0), dtype=torch.long)
        pad = torch.ones((patch_length - segment.size(0), patch_size), dtype=torch.long) * patchilizer.pad_token_id
        mask = torch.cat((mask, torch.zeros(patch_length - segment.size(0), dtype=torch.long)), 0)
        segment = torch.cat((segment, pad), 0)
        feature = model.get_music_features(
            music_inputs=segment.unsqueeze(0).to(device),
            music_masks=mask.unsqueeze(0).to(device),
            get_normalized=True,
        ).squeeze(0)
        features.append(feature)
        weights.append(int(mask.sum().item()))

    stacked = torch.stack(features, dim=0)
    weight_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype).view(-1, 1)
    pooled = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()
    pooled = torch.nn.functional.normalize(pooled, dim=0)
    return pooled.detach().cpu().numpy(), int(input_data.shape[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clamp2-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()

    model, patchilizer, device, patch_length, patch_size = load_clamp2(args.clamp2_dir)
    ref_features = []
    references = []
    for reference in args.reference:
        ref_feature, ref_patches = music_feature(reference, model, patchilizer, device, patch_length, patch_size)
        ref_features.append(ref_feature)
        references.append(
            {
                "name": reference.stem,
                "path": str(reference),
                "patches": ref_patches,
            }
        )
    ref_feature = np.mean(np.stack(ref_features, axis=0), axis=0)
    ref_feature = ref_feature / np.linalg.norm(ref_feature)

    rows = []
    for candidate in args.candidates:
        candidate_feature, patches = music_feature(candidate, model, patchilizer, device, patch_length, patch_size)
        sim = cosine(ref_feature, candidate_feature)
        rows.append(
            {
                "name": candidate.stem,
                "path": str(candidate),
                "patches": patches,
                "cosine_similarity_to_reference": sim,
                "cosine_distance_to_reference": 1.0 - sim,
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "references": references,
                "reference_count": len(references),
                "reference_mode": "single" if len(references) == 1 else "normalized_centroid",
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in rows:
        print(
            row["name"],
            f"similarity={row['cosine_similarity_to_reference']:.6f}",
            f"distance={row['cosine_distance_to_reference']:.6f}",
            f"patches={row['patches']}",
        )


if __name__ == "__main__":
    main()
