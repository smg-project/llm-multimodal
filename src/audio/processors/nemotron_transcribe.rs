//! Native Nemotron-3.5-Transcribe waveform inputs. The model owns the mel
//! frontend; the host only resamples, minimally pads and counts embeddings.

use std::sync::Arc;

use ndarray::Array2;
use serde_json::Value;
use thiserror::Error;

use crate::{
    audio::{transforms::bandlimited_resample, AudioPreProcessor, DecodedAudio},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

pub(crate) const NEMOTRON_AUDIO_LENGTH_KEY: &str = "audio_signal_length";

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid {field}: {reason}")]
pub(crate) struct NemotronTranscribeConfigError {
    pub field: String,
    pub reason: String,
}

/// Geometry consumed by vLLM's native audio tower, resolved from either its
/// normalized config or the published NeMo checkpoint's perception config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NemotronTranscribeAudioParams {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub subsampling_factor: usize,
}

impl NemotronTranscribeAudioParams {
    pub fn from_config(config: &Value) -> Result<Self, NemotronTranscribeConfigError> {
        let invalid = |field: &str, reason: &str| NemotronTranscribeConfigError {
            field: field.to_string(),
            reason: reason.to_string(),
        };
        let positive = |value: Option<&Value>, field: &str| {
            value
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| invalid(field, "expected a positive integer"))
        };
        // vLLM re-resolves perception when present, even if audio_config exists.
        let (sample_rate, hop_length, n_fft, factor) = if config
            .get("perception")
            .is_some_and(|p| !p.is_null() && p.as_object().is_none_or(|p| !p.is_empty()))
        {
            let pre = &config["perception"]["preprocessor"];
            let default_rate = Value::from(16_000);
            let sample_rate = positive(
                pre.get("sample_rate").or(Some(&default_rate)),
                "perception.preprocessor.sample_rate",
            )?;
            let stride = pre
                .get("window_stride")
                .map_or(Some(0.01), Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    invalid(
                        "perception.preprocessor.window_stride",
                        "expected positive seconds",
                    )
                })?;
            let hop = stride * sample_rate as f64;
            if hop < 1.0 || hop >= usize::MAX as f64 {
                return Err(invalid(
                    "perception.preprocessor.window_stride",
                    "hop length is out of range",
                ));
            }
            let n_fft = if pre
                .get("n_fft")
                .is_none_or(|v| v.is_null() || v.as_u64() == Some(0))
            {
                let window = pre
                    .get("window_size")
                    .map_or(Some(0.02), Value::as_f64)
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        invalid(
                            "perception.preprocessor.window_size",
                            "expected positive seconds",
                        )
                    })?;
                let samples = window * sample_rate as f64;
                if samples < 1.0 || samples >= usize::MAX as f64 {
                    return Err(invalid(
                        "perception.preprocessor.window_size",
                        "window is out of range",
                    ));
                }
                (samples as usize)
                    .checked_next_power_of_two()
                    .ok_or_else(|| {
                        invalid("perception.preprocessor.window_size", "FFT size overflows")
                    })?
            } else {
                positive(pre.get("n_fft"), "perception.preprocessor.n_fft")?
            };
            let default_factor = Value::from(4);
            let factor = positive(
                config
                    .pointer("/pe_encoder_config/asr_encoder_cfg/subsampling_factor")
                    .or(Some(&default_factor)),
                "pe_encoder_config.asr_encoder_cfg.subsampling_factor",
            )?;
            (sample_rate, hop as usize, n_fft, factor)
        } else {
            let pre = &config["audio_config"]["preprocessor"];
            (
                positive(
                    pre.get("sample_rate"),
                    "audio_config.preprocessor.sample_rate",
                )?,
                positive(
                    pre.get("hop_length"),
                    "audio_config.preprocessor.hop_length",
                )?,
                positive(pre.get("n_fft"), "audio_config.preprocessor.n_fft")?,
                positive(
                    config.pointer("/audio_config/asr_encoder/subsampling_factor"),
                    "audio_config.asr_encoder.subsampling_factor",
                )?,
            )
        };
        // The published native family consumes 16 kHz mono. Refuse other
        // towers instead of silently treating an unrelated SpeechLM as this model.
        if sample_rate != 16_000 {
            return Err(invalid(
                "sample_rate",
                "Nemotron Transcribe requires 16000 Hz",
            ));
        }
        if n_fft % 2 != 0 {
            return Err(invalid("n_fft", "expected an even FFT size"));
        }
        Ok(Self {
            sample_rate,
            hop_length,
            n_fft,
            subsampling_factor: factor,
        })
    }

    /// Native `_num_audio_embeddings`: STFT frame floor followed by the
    /// feature-stacking ceiling. Minimal padding supplies at least one frame.
    pub fn token_count(&self, samples: usize) -> usize {
        (samples.max(self.hop_length) / self.hop_length).div_ceil(self.subsampling_factor)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NemotronTranscribeAudioProcessor {
    params: NemotronTranscribeAudioParams,
}

impl NemotronTranscribeAudioProcessor {
    pub fn from_config(config: &Value) -> Result<Self, NemotronTranscribeConfigError> {
        Ok(Self {
            params: NemotronTranscribeAudioParams::from_config(config)?,
        })
    }

    /// Batched waveforms right-padded to a common width. The native tower
    /// slices each row to its true post-minimum-pad `audio_signal_length`.
    #[cfg(test)]
    fn preprocess_decoded(
        &self,
        clips: &[DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_refs(clips.iter())
    }

    fn preprocess_refs<'a>(
        &self,
        clips: impl ExactSizeIterator<Item = &'a DecodedAudio>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if clips.len() == 0 {
            return Err(TransformError::EmptyBatch);
        }
        let mut samples = Vec::new();
        let mut lengths = Vec::with_capacity(clips.len());
        let mut counts = Vec::with_capacity(clips.len());
        let mut sizes = Vec::with_capacity(clips.len());
        for (index, clip) in clips.enumerate() {
            if clip.sample_rate == 0
                || clip.samples.is_empty()
                || clip.samples.iter().any(|s| !s.is_finite())
            {
                return Err(TransformError::ShapeError(format!("invalid audio clip at batch index {index}: expected finite samples and a positive sample rate")));
            }
            let start = samples.len();
            if clip.sample_rate == self.params.sample_rate {
                samples.extend_from_slice(&clip.samples);
            } else {
                samples.extend(bandlimited_resample(
                    &clip.samples,
                    clip.sample_rate,
                    self.params.sample_rate,
                )?);
            }
            let length = (samples.len() - start).max(self.params.hop_length);
            let size = u32::try_from(length)
                .map_err(|_| TransformError::ShapeError("audio length exceeds u32".to_string()))?;
            samples.resize(start + length, 0.0);
            lengths.push(i64::from(size));
            sizes.push((size, 1));
            counts.push(self.params.token_count(length));
        }
        let width = lengths.iter().copied().max().unwrap_or(0) as usize;
        let mut batch = Array2::zeros((lengths.len(), width));
        let mut offset = 0;
        for (mut row, &length) in batch.outer_iter_mut().zip(&lengths) {
            let length = length as usize;
            for (target, &sample) in row.iter_mut().zip(&samples[offset..offset + length]) {
                *target = sample;
            }
            offset += length;
        }
        Ok(
            PreprocessedEncoderInputs::new(batch, counts, sizes).with_extra(
                NEMOTRON_AUDIO_LENGTH_KEY,
                ModelSpecificValue::int_1d(lengths),
            ),
        )
    }
}

impl AudioPreProcessor for NemotronTranscribeAudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_refs(clips.iter().map(|clip| clip.decoded()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> Value {
        json!({"audio_config": {"preprocessor": {"sample_rate":16000,"hop_length":160,"n_fft":512},"asr_encoder":{"subsampling_factor":8}}})
    }
    fn clip(samples: usize, rate: usize) -> DecodedAudio {
        DecodedAudio {
            samples: vec![0.25; samples],
            sample_rate: rate,
        }
    }

    #[test]
    fn native_frame_and_subsampling_boundaries() {
        let params = NemotronTranscribeAudioParams::from_config(&config()).unwrap();
        for (samples, expected) in [
            (0, 1),
            (1, 1),
            (159, 1),
            (160, 1),
            (319, 1),
            (1279, 1),
            (1280, 1),
            (1439, 1),
            (1440, 2),
            (15999, 13),
            (16000, 13),
            (480000, 375),
        ] {
            assert_eq!(params.token_count(samples), expected, "samples={samples}");
        }
        // Even non-power-of-two FFTs cancel in the native center-pad arithmetic.
        let mut varied = config();
        varied["audio_config"]["preprocessor"]["n_fft"] = json!(400);
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&varied)
                .unwrap()
                .token_count(1440),
            2
        );
    }

    #[test]
    fn public_salm_config_takes_precedence_over_legacy_audio_config() {
        let config = json!({"audio_config": {"model_type":"nemotron3_5_transcribe_audio"},"perception":{"preprocessor":{"sample_rate":16000,"window_stride":0.01,"window_size":0.025,"n_fft":512}},"pe_encoder_config":{"asr_encoder_cfg":{"subsampling_factor":8}}});
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&config).unwrap(),
            NemotronTranscribeAudioParams {
                sample_rate: 16000,
                hop_length: 160,
                n_fft: 512,
                subsampling_factor: 8
            }
        );
        let mut fallback = config;
        fallback["perception"]["preprocessor"]["n_fft"] = Value::Null;
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&fallback)
                .unwrap()
                .n_fft,
            512
        );
        fallback["perception"]["preprocessor"]
            .as_object_mut()
            .unwrap()
            .remove("sample_rate");
        fallback["pe_encoder_config"]["asr_encoder_cfg"]
            .as_object_mut()
            .unwrap()
            .remove("subsampling_factor");
        let defaults = NemotronTranscribeAudioParams::from_config(&fallback).unwrap();
        assert_eq!(defaults.sample_rate, 16_000);
        assert_eq!(defaults.subsampling_factor, 4);
    }

    #[test]
    fn batch_padding_preserves_true_lengths_samples_and_counts() {
        let processor = NemotronTranscribeAudioProcessor::from_config(&config()).unwrap();
        let output = processor
            .preprocess_decoded(&[clip(1, 16000), clip(1440, 16000)])
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[2, 1440]);
        assert_eq!(output.feature_token_counts, [1, 2]);
        assert_eq!(output.item_sizes, [(160, 1), (1440, 1)]);
        match &output.model_specific[NEMOTRON_AUDIO_LENGTH_KEY] {
            ModelSpecificValue::IntTensor { data, shape } => {
                assert_eq!(data, &[160, 1440]);
                assert_eq!(shape, &[2]);
            }
            _ => panic!("expected int64 lengths"),
        }
        let values = output.encoder_input.as_f32();
        assert_eq!(values[[0, 0]], 0.25);
        assert!(values
            .index_axis(ndarray::Axis(0), 0)
            .iter()
            .skip(1)
            .all(|v| *v == 0.0));
        assert!(values
            .index_axis(ndarray::Axis(0), 1)
            .iter()
            .all(|v| *v == 0.25));
    }

    #[test]
    fn resamples_before_counting_and_rejects_invalid_audio() {
        let processor = NemotronTranscribeAudioProcessor::from_config(&config()).unwrap();
        let output = processor
            .preprocess_decoded(&[clip(8000, 8000), clip(48000, 48000)])
            .unwrap();
        assert_eq!(output.feature_token_counts, [13, 13]);
        match &output.model_specific[NEMOTRON_AUDIO_LENGTH_KEY] {
            ModelSpecificValue::IntTensor { data, shape } => {
                assert_eq!(data, &[16000, 16000]);
                assert_eq!(shape, &[2]);
            }
            _ => panic!("expected int64 lengths"),
        }
        assert!(processor.preprocess_decoded(&[]).is_err());
        for bad in [
            clip(0, 16000),
            clip(4, 0),
            DecodedAudio {
                samples: vec![f32::NAN],
                sample_rate: 16000,
            },
            DecodedAudio {
                samples: vec![f32::INFINITY],
                sample_rate: 16000,
            },
        ] {
            assert!(processor.preprocess_decoded(&[bad]).is_err());
        }
    }

    #[test]
    fn rejects_malformed_geometry() {
        for (path, value) in [
            ("/audio_config/preprocessor/sample_rate", json!(0)),
            ("/audio_config/preprocessor/sample_rate", json!(24000)),
            ("/audio_config/preprocessor/hop_length", json!(0)),
            ("/audio_config/preprocessor/n_fft", json!(511)),
            ("/audio_config/asr_encoder/subsampling_factor", json!(-1)),
            ("/audio_config/asr_encoder/subsampling_factor", json!("8")),
        ] {
            let mut config = config();
            *config.pointer_mut(path).unwrap() = value;
            assert!(
                NemotronTranscribeAudioParams::from_config(&config).is_err(),
                "{path}"
            );
        }
        assert!(NemotronTranscribeAudioParams::from_config(&json!({})).is_err());
    }

    #[tokio::test]
    async fn decoder_downmixes_stereo_before_waveform_preparation() {
        let frames = 160usize;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + frames as u32 * 4).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&64000u32.to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(frames as u32 * 4).to_le_bytes());
        for _ in 0..frames {
            wav.extend_from_slice(&8192i16.to_le_bytes());
            wav.extend_from_slice(&(-8192i16).to_le_bytes());
        }
        let decoded = crate::audio::decode_audio_mono_f32(&wav).await.unwrap();
        assert_eq!(decoded.samples.len(), frames);
        let output = NemotronTranscribeAudioProcessor::from_config(&config())
            .unwrap()
            .preprocess_decoded(&[decoded])
            .unwrap();
        assert!(output.encoder_input.as_f32().iter().all(|s| s.abs() < 1e-6));
    }
}

#[cfg(test)]
mod native_parity {
    use super::*;

    #[tokio::test]
    async fn matches_native_vllm_waveform_oracle() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/nemotron_transcribe_native.json"
        ))
        .unwrap();
        let processor = NemotronTranscribeAudioProcessor::from_config(
            &serde_json::json!({"audio_config":fixture["config"]}),
        )
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let rate = case["sample_rate"].as_u64().unwrap() as usize;
            let decoded = if let Some(wav) = case["input_wav"].as_str() {
                let bytes = std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("tests/fixtures/nemotron_transcribe")
                        .join(wav),
                )
                .unwrap();
                crate::audio::decode_audio_mono_f32(&bytes).await.unwrap()
            } else if let Some(count) = case["input_num_samples"].as_u64() {
                DecodedAudio {
                    samples: vec![case["constant_sample"].as_f64().unwrap() as f32; count as usize],
                    sample_rate: rate,
                }
            } else {
                let input = case["input_samples"].as_array().unwrap();
                if input[0].is_array() {
                    // Exercise the production decoder's channel reduction with
                    // lossless float WAV, not an independently averaged input.
                    let channels_first = input.len() == 2;
                    let frames = if channels_first {
                        input[0].as_array().unwrap().len()
                    } else {
                        input.len()
                    };
                    let mut wav = Vec::new();
                    wav.extend_from_slice(b"RIFF");
                    wav.extend_from_slice(&(36 + frames as u32 * 8).to_le_bytes());
                    wav.extend_from_slice(b"WAVEfmt ");
                    wav.extend_from_slice(&16u32.to_le_bytes());
                    wav.extend_from_slice(&3u16.to_le_bytes());
                    wav.extend_from_slice(&2u16.to_le_bytes());
                    wav.extend_from_slice(&(rate as u32).to_le_bytes());
                    wav.extend_from_slice(&(rate as u32 * 8).to_le_bytes());
                    wav.extend_from_slice(&8u16.to_le_bytes());
                    wav.extend_from_slice(&32u16.to_le_bytes());
                    wav.extend_from_slice(b"data");
                    wav.extend_from_slice(&(frames as u32 * 8).to_le_bytes());
                    let interleaved: Vec<f32> = if channels_first {
                        input[0]
                            .as_array()
                            .unwrap()
                            .iter()
                            .zip(input[1].as_array().unwrap())
                            .flat_map(|(left, right)| {
                                [
                                    left.as_f64().unwrap() as f32,
                                    right.as_f64().unwrap() as f32,
                                ]
                            })
                            .collect()
                    } else {
                        input
                            .iter()
                            .flat_map(|frame| {
                                frame
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|v| v.as_f64().unwrap() as f32)
                            })
                            .collect()
                    };
                    for sample in interleaved {
                        wav.extend_from_slice(&sample.to_le_bytes());
                    }
                    crate::audio::decode_audio_mono_f32(&wav).await.unwrap()
                } else {
                    DecodedAudio {
                        samples: input.iter().map(|v| v.as_f64().unwrap() as f32).collect(),
                        sample_rate: rate,
                    }
                }
            };
            let output = processor.preprocess_decoded(&[decoded]).unwrap();
            let length = case["expected_length"].as_u64().unwrap() as usize;
            assert_eq!(output.encoder_input.shape(), &[1, length], "{name}");
            assert_eq!(
                output.feature_token_counts,
                [case["expected_tokens"].as_u64().unwrap() as usize],
                "{name}"
            );
            match &output.model_specific[NEMOTRON_AUDIO_LENGTH_KEY] {
                ModelSpecificValue::IntTensor { data, shape } => {
                    assert_eq!(data, &[length as i64], "{name}");
                    assert_eq!(shape, &[1]);
                }
                _ => panic!("expected int64 lengths"),
            }
            let expected = if let Some(samples) = case["expected_samples"].as_array() {
                samples
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            } else {
                let mut samples = vec![
                    case["constant_sample"].as_f64().unwrap() as f32;
                    case["input_num_samples"].as_u64().unwrap() as usize
                ];
                samples.resize(length, 0.0);
                samples
            };
            let mut max_error = 0.0f32;
            for (index, (actual, expected)) in output
                .encoder_input
                .as_f32()
                .iter()
                .zip(expected)
                .enumerate()
            {
                max_error = max_error.max((actual - expected).abs());
                // Float32 sinc kernels use a different summation order from
                // torch's convolution; same-rate waveform values remain exact.
                let tolerance = if rate == 16_000 { 0.0 } else { 1e-5 };
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{name} sample {index}: {actual} != {expected}"
                );
            }
            eprintln!("{name}: max waveform error {max_error:e}");
        }
    }
}
