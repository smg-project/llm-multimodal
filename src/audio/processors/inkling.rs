//! Inkling audio preprocessing.
//!
//! Converts decoded mono PCM into the model's quantized dMel features. Media
//! fetching, hashing, and decode are owned by `MediaConnector`; this processor
//! starts from the decoded samples stored in [`AudioClip`].

use std::sync::Arc;

use ndarray::Array2;
use realfft::RealFftPlanner;
use serde_json::Value;

use crate::{
    audio::{
        transforms::{bandlimited_resample, hann_window, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
    vision::PreProcessorConfig,
};

/// Maximum number of dMel frames accepted for one clip (~10 minutes at 20 Hz).
const MAX_AUDIO_TOKENS: usize = 12_000;

/// Parameters used to convert decoded audio into Inkling dMel bins.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingAudioParams {
    pub sample_rate: usize,
    pub window_size_multiplier: f64,
    pub n_fft: Option<usize>,
    pub n_mels: usize,
    pub num_dmel_bins: usize,
    pub dmel_min_value: f64,
    pub dmel_max_value: f64,
    pub audio_token_duration_s: f64,
}

impl Default for InklingAudioParams {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            window_size_multiplier: 2.0,
            n_fft: None,
            n_mels: 80,
            num_dmel_bins: 16,
            dmel_min_value: -7.0,
            dmel_max_value: 2.0,
            audio_token_duration_s: 0.05,
        }
    }
}

impl InklingAudioParams {
    fn from_model_config(model_config: &Value) -> Self {
        let mut params = Self::default();
        let Some(audio_config) = model_config.get("audio_config") else {
            return params;
        };

        if let Some(value) = audio_config.get("n_mel_bins").and_then(Value::as_u64) {
            params.n_mels = value as usize;
        }
        if let Some(value) = audio_config.get("mel_vocab_size").and_then(Value::as_u64) {
            params.num_dmel_bins = value as usize;
        }
        if let Some(value) = audio_config.get("dmel_min_value").and_then(Value::as_f64) {
            params.dmel_min_value = value;
        }
        if let Some(value) = audio_config.get("dmel_max_value").and_then(Value::as_f64) {
            params.dmel_max_value = value;
        }
        params
    }

    fn hop_length(&self) -> Result<usize, TransformError> {
        to_exact_int(
            self.audio_token_duration_s * self.sample_rate as f64,
            "audio_token_duration_s * sample_rate",
        )
    }

    fn window_size(&self) -> Result<usize, TransformError> {
        to_exact_int(
            self.audio_token_duration_s * self.window_size_multiplier * self.sample_rate as f64,
            "audio_token_duration_s * window_size_multiplier * sample_rate",
        )
    }

    fn validate(&self) -> Result<(), TransformError> {
        let hop_length = self.hop_length()?;
        let window_size = self.window_size()?;
        let n_fft = self.n_fft.unwrap_or(window_size);
        if self.sample_rate == 0
            || hop_length == 0
            || window_size == 0
            || n_fft == 0
            || self.n_mels == 0
            || self.num_dmel_bins == 0
        {
            return Err(shape_error(
                "Inkling audio dimensions and sample rate must be positive",
            ));
        }
        if window_size > n_fft {
            return Err(shape_error(format!(
                "Inkling audio window size {window_size} exceeds n_fft {n_fft}"
            )));
        }
        if self.dmel_min_value > self.dmel_max_value {
            return Err(shape_error(format!(
                "Inkling dMel minimum {} exceeds maximum {}",
                self.dmel_min_value, self.dmel_max_value
            )));
        }
        Ok(())
    }
}

/// Pure-Rust Inkling dMel processor.
#[derive(Debug, Clone)]
pub struct InklingAudioProcessor {
    params: InklingAudioParams,
}

impl Default for InklingAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingAudioProcessor {
    pub fn new() -> Self {
        Self::with_params(InklingAudioParams::default())
    }

    pub fn with_params(params: InklingAudioParams) -> Self {
        Self { params }
    }

    pub fn from_configs(model_config: &Value, _preprocessor_config: &PreProcessorConfig) -> Self {
        Self::with_params(InklingAudioParams::from_model_config(model_config))
    }

    pub fn params(&self) -> &InklingAudioParams {
        &self.params
    }

    pub fn preprocess_decoded_clips(
        &self,
        clips: Vec<DecodedAudio>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let clips = clips.iter().collect::<Vec<_>>();
        self.preprocess_decoded_clip_refs(&clips)
    }

    fn preprocess_decoded_clip_refs(
        &self,
        clips: &[&DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.params.validate()?;
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let hop_length = self.params.hop_length()?;
        let mut all_bins = Vec::new();
        let mut token_counts = Vec::with_capacity(clips.len());
        let mut item_sizes = Vec::with_capacity(clips.len());

        for (index, clip) in clips.iter().copied().enumerate() {
            if clip.sample_rate == 0 {
                return Err(shape_error("decoded audio sample rate must be positive"));
            }
            if clip.samples.iter().any(|sample| !sample.is_finite()) {
                return Err(shape_error("decoded audio contains a non-finite sample"));
            }
            let predicted_num_frames = resampled_frame_count(
                clip.samples.len(),
                clip.sample_rate,
                self.params.sample_rate,
                hop_length,
            )?;
            validate_audio_token_limit(index, predicted_num_frames)?;

            let resampled;
            let samples = if clip.sample_rate == self.params.sample_rate {
                clip.samples.as_slice()
            } else {
                resampled = bandlimited_resample(
                    clip.samples.as_slice(),
                    clip.sample_rate,
                    self.params.sample_rate,
                )?;
                resampled.as_slice()
            };
            let num_frames = frame_count(samples.len(), hop_length);
            debug_assert_eq!(num_frames, predicted_num_frames);
            validate_audio_token_limit(index, num_frames)?;

            let dmel = dmel_bins(samples, &self.params)?;
            debug_assert_eq!(dmel.num_frames, num_frames);
            all_bins.extend(dmel.bins);
            token_counts.push(num_frames);
            item_sizes.push((
                u32::try_from(self.params.n_mels)
                    .map_err(|_| shape_error("Inkling n_mels exceeds u32"))?,
                u32::try_from(num_frames)
                    .map_err(|_| shape_error("Inkling audio frame count exceeds u32"))?,
            ));
        }

        let total_frames = token_counts.iter().sum();
        let encoder_input = Array2::from_shape_vec((total_frames, self.params.n_mels), all_bins)
            .map_err(|error| {
                shape_error(format!(
                    "failed to create Inkling dMel input [{total_frames}, {}]: {error}",
                    self.params.n_mels
                ))
            })?;
        let num_audio_tokens = token_counts
            .iter()
            .map(|&count| {
                i64::try_from(count)
                    .map_err(|_| shape_error("Inkling audio frame count exceeds i64"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes).with_extra(
                "num_audio_tokens",
                ModelSpecificValue::int_1d(num_audio_tokens),
            ),
        )
    }
}

impl AudioPreProcessor for InklingAudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let decoded = clips.iter().map(|clip| clip.decoded()).collect::<Vec<_>>();
        self.preprocess_decoded_clip_refs(&decoded)
    }
}

struct DmelBins {
    bins: Vec<f32>,
    num_frames: usize,
}

fn dmel_bins(audio: &[f32], params: &InklingAudioParams) -> Result<DmelBins, TransformError> {
    params.validate()?;
    let hop_length = params.hop_length()?;
    let window_size = params.window_size()?;
    let n_fft = params.n_fft.unwrap_or(window_size);
    let num_frames = frame_count(audio.len(), hop_length);
    if audio.is_empty() {
        return Ok(DmelBins {
            bins: Vec::new(),
            num_frames,
        });
    }

    let left_pad = n_fft.saturating_sub(hop_length);
    let right_pad = num_frames * hop_length - audio.len();
    let mut padded = Vec::with_capacity(left_pad + audio.len() + right_pad);
    padded.extend(std::iter::repeat_n(0.0_f32, left_pad));
    padded.extend_from_slice(audio);
    padded.extend(std::iter::repeat_n(0.0_f32, right_pad));

    let mut window = vec![0.0_f32; n_fft];
    let window_offset = (n_fft - window_size) / 2;
    window[window_offset..window_offset + window_size].copy_from_slice(&hann_window(window_size));

    let fft_bins = n_fft / 2 + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut frame = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut magnitude = vec![0.0_f32; num_frames * fft_bins];

    for frame_index in 0..num_frames {
        let start = frame_index * hop_length;
        for (index, slot) in frame.iter_mut().enumerate() {
            *slot = padded[start + index] * window[index];
        }
        fft.process(&mut frame, &mut spectrum)
            .map_err(|error| shape_error(format!("Inkling audio FFT failed: {error}")))?;
        for (bin, value) in spectrum.iter().enumerate() {
            magnitude[frame_index * fft_bins + bin] = value
                .re
                .mul_add(value.re, value.im * value.im)
                .max(1e-10)
                .sqrt();
        }
    }

    let basis = mel_basis(params.sample_rate, n_fft, params.n_mels);
    let bin_centers = linspace(
        params.dmel_min_value,
        params.dmel_max_value,
        params.num_dmel_bins,
    );
    let mut bins = vec![0.0_f32; num_frames * params.n_mels];

    for frame_index in 0..num_frames {
        let frame_magnitude = &magnitude[frame_index * fft_bins..(frame_index + 1) * fft_bins];
        for mel_index in 0..params.n_mels {
            let filter = &basis[mel_index * fft_bins..(mel_index + 1) * fft_bins];
            let mel = filter
                .iter()
                .zip(frame_magnitude)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            let value =
                (mel.max(1e-10).log10() as f64).clamp(params.dmel_min_value, params.dmel_max_value);
            bins[frame_index * params.n_mels + mel_index] =
                nearest_center(value, &bin_centers) as f32;
        }
    }

    Ok(DmelBins { bins, num_frames })
}

fn frame_count(sample_count: usize, hop_length: usize) -> usize {
    sample_count.div_ceil(hop_length)
}

fn resampled_frame_count(
    sample_count: usize,
    source_sample_rate: usize,
    target_sample_rate: usize,
    hop_length: usize,
) -> Result<usize, TransformError> {
    if source_sample_rate == 0 || target_sample_rate == 0 {
        return Err(shape_error("audio resampling rates must be positive"));
    }

    let target_samples = (sample_count as u128)
        .checked_mul(target_sample_rate as u128)
        .ok_or_else(|| shape_error("resampled audio length overflow"))?
        .div_ceil(source_sample_rate as u128);
    let target_samples = usize::try_from(target_samples)
        .map_err(|_| shape_error("resampled audio length exceeds usize"))?;
    Ok(frame_count(target_samples, hop_length))
}

fn validate_audio_token_limit(clip_index: usize, num_frames: usize) -> Result<(), TransformError> {
    if num_frames > MAX_AUDIO_TOKENS {
        return Err(shape_error(format!(
            "Audio clip {clip_index} produces {num_frames} tokens, exceeding the maximum of \
             {MAX_AUDIO_TOKENS} (~10 min at 20 tokens/s). Provide a shorter clip."
        )));
    }
    Ok(())
}

fn to_exact_int(value: f64, name: &str) -> Result<usize, TransformError> {
    const TOLERANCE: f64 = 1e-6;
    let rounded = value.round();
    if !value.is_finite() || (value - rounded).abs() > TOLERANCE || rounded < 0.0 {
        return Err(shape_error(format!(
            "{name} must resolve to an integer sample count, got {value}"
        )));
    }
    Ok(rounded as usize)
}

fn nearest_center(value: f64, centers: &[f64]) -> usize {
    let mut best_index = 0;
    let mut best_distance = f64::INFINITY;
    for (index, center) in centers.iter().enumerate() {
        let distance = (value - center).abs();
        if distance < best_distance {
            best_index = index;
            best_distance = distance;
        }
    }
    best_index
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    (0..count)
        .map(|index| {
            if index + 1 == count {
                stop
            } else {
                start + index as f64 * step
            }
        })
        .collect()
}

fn shape_error(message: impl Into<String>) -> TransformError {
    TransformError::ShapeError(message.into())
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use serde::Deserialize;

    use super::*;

    fn decoded(sample_count: usize, sample_rate: usize) -> DecodedAudio {
        DecodedAudio {
            samples: vec![0.0; sample_count],
            sample_rate,
        }
    }

    #[test]
    fn frame_count_matches_token_rate() {
        let params = InklingAudioParams::default();
        for sample_count in [0, 1, 799, 800, 801, 16_000, 16_001] {
            let output = dmel_bins(&vec![0.0; sample_count], &params).unwrap();
            assert_eq!(output.num_frames, sample_count.div_ceil(800));
            assert_eq!(output.bins.len(), output.num_frames * params.n_mels);
        }
    }

    #[test]
    fn silence_is_uniform_and_low() {
        let params = InklingAudioParams::default();
        let output = dmel_bins(&vec![0.0; 1_600], &params).unwrap();
        let (first, second) = output.bins.split_at(params.n_mels);
        assert_eq!(first, second);
        assert!(first.iter().copied().fold(0.0_f32, f32::max) < 8.0);
    }

    #[test]
    fn tone_energy_reaches_a_higher_dmel_bin() {
        let params = InklingAudioParams::default();
        let audio = (0..16_000)
            .map(|index| {
                (2.0 * PI * 1_000.0 * index as f64 / params.sample_rate as f64).sin() as f32
            })
            .collect::<Vec<_>>();
        let output = dmel_bins(&audio, &params).unwrap();
        let frame = &output.bins[10 * params.n_mels..11 * params.n_mels];
        assert!(frame.iter().copied().fold(0.0_f32, f32::max) > 8.0);
    }

    #[test]
    fn preprocesses_empty_clip_as_zero_rows() {
        let output = InklingAudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(0, 16_000)])
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[0, 80]);
        assert_eq!(output.feature_token_counts, vec![0]);
    }

    #[test]
    fn preprocesses_and_flattens_multiple_clips() {
        let output = InklingAudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(800, 16_000), decoded(1_600, 16_000)])
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[3, 80]);
        assert_eq!(output.feature_token_counts, vec![1, 2]);
        assert_eq!(output.item_sizes, vec![(80, 1), (80, 2)]);
        assert!(matches!(
            output.model_specific.get("num_audio_tokens"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data == &vec![1, 2] && shape == &vec![2]
        ));
    }

    #[test]
    fn resamples_48khz_audio_before_dmel() {
        let output = InklingAudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(4_800, 48_000)])
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[2, 80]);
        assert_eq!(output.feature_token_counts, vec![2]);
    }

    #[test]
    fn enforces_per_clip_token_limit() {
        assert!(validate_audio_token_limit(0, MAX_AUDIO_TOKENS).is_ok());
        let error = validate_audio_token_limit(2, MAX_AUDIO_TOKENS + 1).unwrap_err();
        assert!(error.to_string().contains("Audio clip 2"));
        assert!(error.to_string().contains("12000"));
    }

    #[test]
    fn rejects_oversized_clip_before_resampling() {
        let error = InklingAudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(601, 1)])
            .unwrap_err();

        assert!(error.to_string().contains("12020 tokens"));
        assert!(error.to_string().contains("12000"));
    }

    #[test]
    fn rejects_non_finite_samples() {
        for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = InklingAudioProcessor::new()
                .preprocess_decoded_clips(vec![DecodedAudio {
                    samples: vec![sample],
                    sample_rate: 16_000,
                }])
                .unwrap_err();
            assert!(error.to_string().contains("non-finite sample"));
        }
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let error = InklingAudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(1, 0)])
            .unwrap_err();
        assert!(error.to_string().contains("sample rate must be positive"));
    }

    #[test]
    fn reads_nested_audio_config() {
        let processor = InklingAudioProcessor::from_configs(
            &serde_json::json!({
                "audio_config": {
                    "n_mel_bins": 64,
                    "mel_vocab_size": 32,
                    "dmel_min_value": -1.5,
                    "dmel_max_value": 1.75
                }
            }),
            &PreProcessorConfig::default(),
        );
        assert_eq!(processor.params().n_mels, 64);
        assert_eq!(processor.params().num_dmel_bins, 32);
        assert_eq!(processor.params().dmel_min_value, -1.5);
        assert_eq!(processor.params().dmel_max_value, 1.75);
        assert_eq!(processor.params().sample_rate, 16_000);
    }

    #[derive(Deserialize)]
    struct ParityCase {
        name: String,
        params: ParityParams,
        synth: ParitySynth,
        num_frames: usize,
        n_mels: usize,
        expected_bins: Vec<i32>,
    }

    #[derive(Deserialize)]
    struct ParityParams {
        sample_rate: usize,
        n_mels: usize,
        num_dmel_bins: usize,
        dmel_min_value: f64,
        dmel_max_value: f64,
    }

    #[derive(Deserialize)]
    struct ParitySynth {
        seed: u32,
        num_samples: usize,
        sample_rate: usize,
    }

    fn synth_samples(seed: u32, num_samples: usize) -> Vec<f32> {
        const SYNTH_BLOCK: usize = 800;
        const SYNTH_STEPS: usize = 12;

        let mut state = seed;
        (0..num_samples)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = (state >> 16) as u16 as i16;
                let exponent = 8 - 2 * ((index / SYNTH_BLOCK) % SYNTH_STEPS) as i32;
                f32::from(value) / 32_768.0 * (2.0_f32).powi(exponent)
            })
            .collect()
    }

    #[test]
    fn dmel_matches_python_reference() {
        let cases: Vec<ParityCase> =
            serde_json::from_str(include_str!("fixtures/inkling_dmel_parity.json")).unwrap();
        assert!(!cases.is_empty());

        for case in cases {
            let params = InklingAudioParams {
                sample_rate: case.params.sample_rate,
                n_mels: case.params.n_mels,
                num_dmel_bins: case.params.num_dmel_bins,
                dmel_min_value: case.params.dmel_min_value,
                dmel_max_value: case.params.dmel_max_value,
                ..Default::default()
            };
            let samples = synth_samples(case.synth.seed, case.synth.num_samples);
            let samples = if case.synth.sample_rate == params.sample_rate {
                samples
            } else {
                bandlimited_resample(&samples, case.synth.sample_rate, params.sample_rate).unwrap()
            };
            let output = dmel_bins(&samples, &params).unwrap();
            let actual = output
                .bins
                .iter()
                .map(|value| *value as i32)
                .collect::<Vec<_>>();

            assert_eq!(output.num_frames, case.num_frames, "{}", case.name);
            assert_eq!(params.n_mels, case.n_mels, "{}", case.name);
            assert_eq!(actual, case.expected_bins, "{}", case.name);
        }
    }
}
