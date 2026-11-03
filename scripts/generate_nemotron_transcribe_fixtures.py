#!/usr/bin/env python3
r"""Generate independent Nemotron Transcribe fixtures by executing native vLLM.

Reference: vLLM a900ac810ecaedc0c2c5e9a8887e353cf1a5c350. Supply that checkout
and the public model config containing NeMo perception/pe_encoder_config:

    python scripts/generate_nemotron_transcribe_fixtures.py \
        --vllm-source /path/to/vllm --model-config /path/to/config.json \
        --output tests/fixtures/nemotron_transcribe_native.json

Requires Git on PATH, Python 3.12, NumPy, torch, torchaudio and transformers. Recorded versions:
NumPy 2.2.6, torch 2.13.0+cu130, torchaudio 2.11.0+cu130, transformers 5.14.1
(from nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0). No GPU or weights are needed.
Waveforms can differ with different numerical-library versions; provenance
records the versions and hashes of every executed native source file.

Original function bodies are AST-extracted and executed, rather than copying
preprocessing formulas. Processing classes lose only framework bases and unused
methods; context getters, list predicates and result containers are plumbing
stubs. The actual native config resolution, audio normalization, torchaudio
resampling, waveform padding and token counting execute unchanged. Six PCM16 WAV
inputs are written in the nemotron_transcribe/ directory beside the output JSON,
matching the Rust tests' fixture layout. The source checkout is executable
code and must be trusted. This covers CPU preprocessing, not model inference.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import subprocess
import types
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, cast, Literal

cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
cli.add_argument("--vllm-source", type=Path, required=True, help="Native vLLM git checkout root")
cli.add_argument("--model-config", type=Path, required=True, help="Public NeMo model config.json")
cli.add_argument("--output", type=Path, required=True, help="Output fixture JSON; WAV files go in the adjacent nemotron_transcribe/ directory")
args = cli.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
wav_dir = args.output.parent / "nemotron_transcribe"
wav_dir.mkdir(parents=True, exist_ok=True)
native_commit = subprocess.check_output(
    ["git", "-C", str(args.vllm_source), "rev-parse", "HEAD"], text=True
).strip()

import numpy as np
import numpy.typing as npt
import torch
import torchaudio
import transformers
from transformers import BatchFeature

SOURCE = args.vllm_source / 'vllm'
selected = {}
ns = dict(globals())
def extract(relative, names):
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.ClassDef) and isinstance(names, dict):
                node.bases = []
                node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in names[node.name]]
            nodes.append(node)
    if set(x.name for x in nodes) != set(names):
        raise RuntimeError(f'Missing extraction in {path}: {names}')
    entry = selected.setdefault(relative, {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'symbols': []})
    entry['symbols'].extend(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, 'exec'), ns)

extract('transformers_utils/configs/nemotron3_5_transcribe.py', ['_encoder_config', '_preprocessor_config', '_checked_online', 'resolve_audio_config'])
ns['_SUPPORTED_SPEAKER_FEATURE_MODES'] = ('thresholded', 'continuous')
raw = json.loads(args.model_config.read_text())
config = ns['resolve_audio_config'](raw['perception'], raw['pe_encoder_config'])
extract('utils/torch_utils.py', ['set_default_torch_num_threads'])
extract('multimodal/audio.py', ['ChannelReduction', 'AudioSpec', 'normalize_audio', '_get_torchaudio_resampler', 'resample_audio_torchaudio', 'AudioResampler'])
# Plumbing wrappers only: output container and list predicate are unrelated to DSP.
ns['is_list_of'] = lambda value, kind: isinstance(value, list) and all(isinstance(v, kind) for v in value)
ns['AudioProcessorItems'] = lambda value: value
extract('multimodal/parse.py', {'MultiModalDataParser': ['is_embeddings', '_get_audio_with_sr', '_parse_audio_data']})
extract('model_executor/models/nemotron3_5_transcribe.py', ['_num_audio_embeddings'])
extract('model_executor/models/nemotron3_5_transcribe.py', {'Nemotron3_5TranscribeProcessingInfo': ['get_min_num_samples', 'get_num_audio_tokens'], 'Nemotron3_5TranscribeMultiModalProcessor': ['_call_hf_processor']})
info = ns['Nemotron3_5TranscribeProcessingInfo']()
info.get_hf_config = lambda: types.SimpleNamespace(audio_config=config)
processor = ns['Nemotron3_5TranscribeMultiModalProcessor']()
processor.info = info
parser = ns['MultiModalDataParser']()
parser.audio_resampler = ns['AudioResampler'](target_sr=config['preprocessor']['sample_rate'])
parser.target_channels = 1
parser.expected_hidden_size = None
parser.embedding_fields = {}

cases=[]
def add_case(name, data, rate=16000, constant=False):
    data=np.asarray(data, dtype=np.float32)
    normalized=parser._parse_audio_data((data, rate))
    result=processor._call_hf_processor({'audio': normalized}, {})
    signal=result['audio_signal'][0]
    length=int(result['audio_signal_length'][0])
    case={'name': name,'sample_rate': rate,'expected_samples': signal.tolist(), 'expected_length':length,'expected_tokens':info.get_num_audio_tokens(length),'expected_dtype':str(signal.dtype),'length_dtype':str(result['audio_signal_length'].dtype),'length_device':str(result['audio_signal_length'].device)}
    if constant:
        case.update(input_num_samples=len(data), constant_sample=float(data[0]))
        case.pop('expected_samples')
        case['expected_constant_samples']=float(data[0])
        case['expected_zero_padding']=max(0,length-len(data))
    else:
        case['input_samples']=data.tolist()
    cases.append(case)

for n in [1,159,160,161,1279,1280,1281,1439,1440,2559,2560,2561,16000]:
    add_case(f'mono16k_constant_{n}', np.full(n,.25,dtype=np.float32), constant=True)
add_case('mono16k_signed_pcm',[-1,-.75,-.5,-.25,0,.25,.5,.75,32767/32768])
add_case('stereo16k_channels_time', [[-.5,.25,.75,0],[.5,.75,-.25,-1]])
add_case('stereo16k_time_channels', [[-.5,.5],[.25,.75],[.75,-.25],[0,-1]])
for rate,n in [(8000,97),(44100,317),(48000,481),(22050,223)]:
    signal=np.sin(np.arange(n,dtype=np.float32)*np.float32(.137))*np.float32(.5)
    add_case(f'mono{rate}_sine_{n}',signal,rate)
    add_case(f'stereo{rate}_sine_{n}',np.stack([signal,-signal*.5]),rate)
# Portable 16-bit PCM WAV inputs: quantize first, then oracle gets exactly decoded samples.
for rate, channels, n in [(16000,1,159),(16000,1,1440),(16000,2,161),(8000,1,97),(44100,2,317),(48000,1,481)]:
    base=(np.sin(np.arange(n,dtype=np.float32)*np.float32(.137))*12000).astype('<i2')
    pcm=base[:,None] if channels==1 else np.stack([base,(-base//2).astype('<i2')],axis=1)
    name=f'pcm16wav_{rate}_{channels}ch_{n}'
    filename=name+'.wav'
    with wave.open(str(wav_dir / filename),'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())
    decoded=pcm[:,0].astype(np.float32)/32768 if channels==1 else pcm.T.astype(np.float32)/32768
    add_case(name,decoded,rate)
    cases[-1]['input_wav']=filename
# True native variable-length list output, no cross-clip padding.
batch_arrays=[np.array([.5],dtype=np.float32),np.full(161,-.25,dtype=np.float32),np.full(1280,.125,dtype=np.float32)]
batch_result=processor._call_hf_processor({'audio':batch_arrays},{})
batches=[{'name':'three_variable_clips','case_names':['custom_one','custom_161','custom_1280'],'input_lengths':[len(x) for x in batch_arrays],'expected_lengths':batch_result['audio_signal_length'].tolist(),'expected_shapes':[list(x.shape) for x in batch_result['audio_signal']],'expected_tokens':[info.get_num_audio_tokens(int(n)) for n in batch_result['audio_signal_length']]}]
output={'provenance':{'native_vllm_commit':native_commit,'model_config_sha256':hashlib.sha256(args.model_config.read_bytes()).hexdigest(),'executed_native_symbols':selected,'torch':torch.__version__,'torchaudio':torchaudio.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'method':'AST-extracted original function/class definitions compiled unchanged; class framework bases removed, HF config getter and parser output plumbing stubbed; no mathematical formulas or DSP reimplemented'},'config':config,'cases':cases,'batches':batches}
args.output.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'cases':len(cases),'batches':batches,'resolved_preprocessor':config['preprocessor'],'asr_subsampling_factor':config['asr_encoder']['subsampling_factor'],'case_summary':[{k:v for k,v in c.items() if k in ['name','expected_length','expected_tokens']} for c in cases]},indent=2))
