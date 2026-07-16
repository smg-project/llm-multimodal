# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate Inkling dMel parity fixtures from the Python reference."""

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from vllm.transformers_utils.processors.inkling import (
    InklingAudioEncoderParams,
    _dmel_bins,
)

SYNTH_BLOCK = 800
SYNTH_STEPS = 12


def synth_samples(seed: int, num_samples: int) -> np.ndarray:
    state = seed & 0xFFFFFFFF
    samples = np.empty(num_samples, dtype=np.float32)
    for index in range(num_samples):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        value = (state >> 16) & 0xFFFF
        if value >= 0x8000:
            value -= 0x10000
        exponent = 8 - 2 * ((index // SYNTH_BLOCK) % SYNTH_STEPS)
        samples[index] = (
            np.float32(value)
            / np.float32(32768.0)
            * np.float32(2.0**exponent)
        )
    return samples


def make_case(
    name: str,
    seed: int,
    num_samples: int,
    params: InklingAudioEncoderParams,
    source_sample_rate: int | None = None,
) -> dict:
    source_sample_rate = source_sample_rate or params.sample_rate
    audio = torch.from_numpy(np.ascontiguousarray(synth_samples(seed, num_samples)))
    if source_sample_rate != params.sample_rate:
        audio = torchaudio.functional.resample(
            audio,
            source_sample_rate,
            params.sample_rate,
        )
    bins = _dmel_bins(audio, params)
    return {
        "name": name,
        "params": {
            "sample_rate": params.sample_rate,
            "n_mels": params.n_mels,
            "num_dmel_bins": params.num_dmel_bins,
            "dmel_min_value": params.dmel_min_value,
            "dmel_max_value": params.dmel_max_value,
        },
        "synth": {
            "seed": seed,
            "num_samples": num_samples,
            "sample_rate": source_sample_rate,
        },
        "num_frames": int(bins.shape[0]),
        "n_mels": int(bins.shape[1]),
        "expected_bins": bins.flatten().tolist(),
    }


def main() -> None:
    cases = [
        make_case("defaults_1s", 42, 16_123, InklingAudioEncoderParams()),
        make_case(
            "inkling_config_range_1s",
            42,
            16_123,
            InklingAudioEncoderParams(dmel_min_value=-1.5, dmel_max_value=2.0),
        ),
        make_case("short_clip_3_frames", 7, 2_100, InklingAudioEncoderParams()),
        make_case(
            "resample_48khz",
            99,
            4_923,
            InklingAudioEncoderParams(),
            source_sample_rate=48_000,
        ),
    ]
    output = Path(__file__).with_name("inkling_dmel_parity.json")
    output.write_text(json.dumps(cases) + "\n")
    print(f"wrote {output} with {len(cases)} cases")


if __name__ == "__main__":
    main()
