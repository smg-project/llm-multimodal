//! Qwen2-VL family image processors.
//!
//! This module provides the Qwen2-VL processor which wraps the shared
//! `QwenVLProcessorBase` with Qwen2-VL specific default parameters.
//!
//! # Key Features
//!
//! - **Smart Resize**: Resizes images to fit within min/max pixel bounds while
//!   preserving aspect ratio and aligning to patch boundaries
//! - **Dynamic Token Count**: Token count depends on actual image dimensions
//! - **image_grid_thw**: Returns (T, H, W) grid dimensions for position encoding
//!
//! # Qwen2-VL Parameters
//!
//! - patch_size: 14
//! - merge_size: 2
//! - factor: 28 (patch_size * merge_size)
//! - normalization: CLIP mean/std

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase};
use crate::video::{FrameSampler, VideoDecodeError, VideoMetadata};
use crate::vision::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::TransformError,
    video_processor::{PreprocessedVideos, VideoPreProcessor},
};

/// CLIP normalization mean values used by Qwen2-VL models.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values used by Qwen2-VL models.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default minimum pixels (256 * 28 * 28 = 200,704)
pub const DEFAULT_MIN_PIXELS: usize = 256 * 28 * 28;

/// Default maximum pixels (1280 * 28 * 28 = 1,003,520)
pub const DEFAULT_MAX_PIXELS: usize = 1280 * 28 * 28;

/// Default patch size
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default merge size for token reduction
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (for video frames)
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Default minimum sampled frames for Qwen2-VL videos.
pub const DEFAULT_MIN_FRAMES: usize = 4;

/// Default maximum sampled frames for Qwen2-VL videos.
pub const DEFAULT_MAX_FRAMES: usize = 768;

/// Qwen2-VL-specific frame sampler mirroring transformers' `sample_frames`.
#[derive(Debug, Clone)]
pub struct Qwen2VLFrameSampler {
    pub do_sample_frames: bool,
    pub temporal_patch_size: usize,
    pub min_frames: usize,
    pub max_frames: usize,
    pub num_frames: Option<usize>,
    pub fps: Option<f64>,
}

impl Default for Qwen2VLFrameSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2VLFrameSampler {
    pub fn new() -> Self {
        Self {
            do_sample_frames: false,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            min_frames: DEFAULT_MIN_FRAMES,
            max_frames: DEFAULT_MAX_FRAMES,
            num_frames: None,
            fps: None,
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self, VideoDecodeError> {
        let sampler = Self {
            do_sample_frames: config.do_sample_frames.unwrap_or(false),
            temporal_patch_size: config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
            min_frames: config.min_frames.unwrap_or(DEFAULT_MIN_FRAMES),
            max_frames: config.max_frames.unwrap_or(DEFAULT_MAX_FRAMES),
            num_frames: config.num_frames,
            fps: config.fps,
        };
        sampler.validate()?;
        Ok(sampler)
    }

    fn validate(&self) -> Result<(), VideoDecodeError> {
        if self.temporal_patch_size == 0 {
            return Err(VideoDecodeError::InvalidSampling(
                "`temporal_patch_size` must be greater than zero".into(),
            ));
        }
        if self.min_frames > self.max_frames {
            return Err(VideoDecodeError::InvalidSampling(
                "`min_frames` must be less than or equal to `max_frames`".into(),
            ));
        }
        if self.num_frames.is_some() && self.fps.is_some() {
            return Err(VideoDecodeError::InvalidSampling(
                "`num_frames` and `fps` are mutually exclusive".into(),
            ));
        }
        if matches!(self.fps, Some(fps) if !fps.is_finite() || fps <= 0.0) {
            return Err(VideoDecodeError::InvalidSampling(
                "`fps` must be a positive finite number".into(),
            ));
        }
        Ok(())
    }

    fn round_to_multiple(value: usize, multiple: usize) -> usize {
        let quotient = value / multiple;
        let remainder = value % multiple;
        let doubled_remainder = remainder * 2;

        let rounded_quotient = if doubled_remainder < multiple {
            quotient
        } else if doubled_remainder > multiple {
            quotient + 1
        } else if quotient % 2 == 0 {
            quotient
        } else {
            quotient + 1
        };

        rounded_quotient * multiple
    }
}

impl FrameSampler for Qwen2VLFrameSampler {
    fn sample_indices(&self, meta: &VideoMetadata) -> Result<Vec<usize>, VideoDecodeError> {
        self.validate()?;

        if meta.total_frames == 0 {
            return Ok(Vec::new());
        }

        if !self.do_sample_frames {
            return Ok((0..meta.total_frames).collect());
        }

        let total_frames = meta.total_frames;
        let temporal_patch_size = self.temporal_patch_size;

        let num_frames = if let Some(num_frames) = self.num_frames {
            let rounded = Self::round_to_multiple(num_frames, temporal_patch_size);
            if rounded > total_frames {
                return Err(VideoDecodeError::InvalidSampling(format!(
                    "inferred `num_frames={rounded}` exceeds `total_num_frames={total_frames}`"
                )));
            }
            rounded
        } else if let Some(target_fps) = self.fps {
            if meta.fps <= 0.0 {
                return Err(VideoDecodeError::InvalidSampling(
                    "sampling with `fps` requires source video fps metadata".into(),
                ));
            }

            let capped_max_frames =
                (self.max_frames.min(total_frames) / temporal_patch_size) * temporal_patch_size;
            let inferred = total_frames as f64 / meta.fps * target_fps;
            let bounded = inferred
                .max(self.min_frames as f64)
                .min(capped_max_frames as f64)
                .min(total_frames as f64);
            ((bounded / temporal_patch_size as f64).floor() as usize) * temporal_patch_size
        } else {
            total_frames
        };

        if num_frames == 0 {
            return Ok(Vec::new());
        }

        let step = total_frames as f64 / num_frames as f64;
        Ok((0..num_frames)
            .map(|idx| (idx as f64 * step).floor() as usize)
            .collect())
    }
}

/// Qwen2-VL image processor.
///
/// This is a thin wrapper around `QwenVLProcessorBase` with Qwen2-VL
/// specific default parameters:
/// - patch_size: 14
/// - merge_size: 2
/// - CLIP normalization mean/std
#[derive(Debug, Clone)]
pub struct Qwen2VLProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Qwen2VLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2VLProcessor {
    /// Create a new Qwen2-VL processor with default settings.
    ///
    /// Defaults:
    /// - patch_size: 14
    /// - merge_size: 2
    /// - min_pixels: 200,704 (256 * 28 * 28)
    /// - max_pixels: 1,003,520 (1280 * 28 * 28)
    /// - temporal_patch_size: 2
    /// - normalization: CLIP mean/std
    pub fn new() -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: DEFAULT_MAX_PIXELS,
                temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
                max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
                temporal_patch_size: config
                    .temporal_patch_size
                    .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    /// Get the factor for dimension alignment.
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.inner.get_factor()
    }

    /// Smart resize algorithm for Qwen2-VL.
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        self.inner.smart_resize(height, width)
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        self.inner.calculate_grid_thw(height, width, num_frames)
    }

    /// Calculate the number of image tokens after merge.
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        self.inner
            .calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }
}

impl Deref for Qwen2VLProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ImagePreProcessor for Qwen2VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        self.inner.preprocess(images, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

impl VideoPreProcessor for Qwen2VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        videos: &[Vec<DynamicImage>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedVideos, TransformError> {
        self.inner.preprocess_video(videos, config)
    }

    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        num_frames: u32,
        _config: &PreProcessorConfig,
    ) -> usize {
        self.inner
            .calculate_num_video_tokens(width, height, num_frames)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::{image_processor::ModelSpecificValue, preprocessor_config::PatchSize};

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_qwen2_vl_processor_default() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels(), DEFAULT_MIN_PIXELS);
        assert_eq!(processor.max_pixels(), DEFAULT_MAX_PIXELS);
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = Qwen2VLProcessor::new();

        // Image that's already within bounds
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        // Should be aligned to factor (28)
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);

        // Should be within bounds
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_too_large() {
        let processor = Qwen2VLProcessor::new();

        // Very large image
        let (h, w) = processor.smart_resize(3000, 3000).unwrap();

        // Should be scaled down
        assert!(h * w <= processor.max_pixels());
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_too_small() {
        let processor = Qwen2VLProcessor::new();

        // Small image (but above minimum dimension)
        let (h, w) = processor.smart_resize(100, 100).unwrap();

        // Should be scaled up to min_pixels
        assert!(h * w >= processor.min_pixels());
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_aspect_ratio_preserved() {
        let processor = Qwen2VLProcessor::new();

        // 2:1 aspect ratio
        let (h, w) = processor.smart_resize(400, 800).unwrap();

        // Aspect ratio should be approximately preserved
        let original_ratio = 800.0 / 400.0;
        let new_ratio = w as f64 / h as f64;
        assert!((new_ratio - original_ratio).abs() < 0.5);
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = Qwen2VLProcessor::new();

        // 300:1 aspect ratio - should fail
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_smart_resize_small_dimension_clamps_to_factor() {
        let processor = Qwen2VLProcessor::new();

        // Dimension smaller than factor (28) should be clamped up, not rejected
        let (h, w) = processor.smart_resize(10, 100).unwrap();
        assert!(h >= 28);
        assert!(w >= 28);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_calculate_grid_thw_image() {
        let processor = Qwen2VLProcessor::new();

        // 448x448 image (16x16 grid patches)
        let (t, h, w) = processor.calculate_grid_thw(448, 448, 1);

        assert_eq!(t, 1); // Single image
        assert_eq!(h, 448 / 14); // 32
        assert_eq!(w, 448 / 14); // 32
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = Qwen2VLProcessor::new();

        // With merge_size=2, tokens = (t * h * w) / 4
        let tokens = processor.calculate_tokens_from_grid(1, 32, 32);
        assert_eq!(tokens, (32 * 32) / 4); // 256
    }

    #[test]
    fn test_qwen2_vl_preprocess() {
        let processor = Qwen2VLProcessor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(CLIP_MEAN.to_vec()),
            image_std: Some(CLIP_STD.to_vec()),
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            min_pixels: Some(DEFAULT_MIN_PIXELS),
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        // Disambiguate: Qwen2VLProcessor implements both Image- and VideoPreProcessor.
        let result = ImagePreProcessor::preprocess(&processor, &[image], &config).unwrap();

        // pixel_values is patchified: [total_patches, patch_features]
        assert_eq!(result.pixel_values.ndim(), 2);
        assert!(result.pixel_values.shape()[0] > 0); // total_patches > 0

        // Check pixel values are normalized
        let flat = result.pixel_values_flat();
        // After normalization with CLIP mean/std, gray (0.5) should be near 0
        // (0.5 - 0.48) / 0.27 ≈ 0.07
        assert!(flat.iter().all(|&v| v.abs() < 1.0)); // Should be normalized

        // Check image_grid_thw and patches_per_image are present
        assert!(result.model_specific.contains_key("image_grid_thw"));
        assert!(result.model_specific.contains_key("patches_per_image"));

        // Verify token count is reasonable
        assert!(result.num_img_tokens[0] > 0);
    }

    #[test]
    fn test_qwen2_vl_preprocess_multiple() {
        let processor = Qwen2VLProcessor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = ImagePreProcessor::preprocess(&processor, &images, &config).unwrap();

        // Both images processed
        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);

        // pixel_values is 2D [total_patches, patch_features]
        assert_eq!(result.pixel_values.ndim(), 2);

        // Check grid_thw shape
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("image_grid_thw")
        {
            assert_eq!(shape, &[2, 3]); // 2 images, 3 values (T, H, W) each
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected image_grid_thw to be IntTensor");
        }

        // Check patches_per_image
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("patches_per_image")
        {
            assert_eq!(shape, &[2]); // 2 images
            assert_eq!(data.len(), 2);
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.pixel_values.shape()[0]);
        } else {
            panic!("Expected patches_per_image to be IntTensor");
        }
    }

    #[test]
    fn test_qwen2_vl_from_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(16),
                width: Some(16),
            }),
            merge_size: Some(4),
            min_pixels: Some(100000),
            max_pixels: Some(500000),
            temporal_patch_size: Some(4),
            ..Default::default()
        };

        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.patch_size(), 16);
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.min_pixels(), 100000);
        assert_eq!(processor.max_pixels(), 500000);
        assert_eq!(processor.temporal_patch_size(), 4);
    }

    #[test]
    fn test_model_name() {
        let processor = Qwen2VLProcessor::new();
        // Both Image- and VideoPreProcessor define model_name; pick one explicitly.
        assert_eq!(ImagePreProcessor::model_name(&processor), "qwen2-vl");
        assert_eq!(VideoPreProcessor::model_name(&processor), "qwen2-vl");
    }

    #[test]
    fn test_default_mean_std() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(ImagePreProcessor::default_mean(&processor), CLIP_MEAN);
        assert_eq!(ImagePreProcessor::default_std(&processor), CLIP_STD);
    }

    fn video_config() -> PreProcessorConfig {
        PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(CLIP_MEAN.to_vec()),
            image_std: Some(CLIP_STD.to_vec()),
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            min_pixels: Some(DEFAULT_MIN_PIXELS),
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            ..Default::default()
        }
    }

    fn sample_video_metadata() -> VideoMetadata {
        VideoMetadata {
            duration_secs: 4.0,
            total_frames: 24,
            fps: 6.0,
            width: 640,
            height: 480,
            codec: "TEST".into(),
        }
    }

    #[test]
    fn test_qwen2_vl_preprocess_video_even_frames() {
        let processor = Qwen2VLProcessor::new();
        let config = video_config();

        // 4 frames -> grid_t = 4 / temporal_patch_size(2) = 2
        let frames: Vec<DynamicImage> = (0..4)
            .map(|i| create_test_image(448, 448, Rgb([10 * i as u8, 20, 30])))
            .collect();
        let result = VideoPreProcessor::preprocess(&processor, &[frames], &config).unwrap();

        // pixel_values is patchified 2D [total_patches, patch_features]
        assert_eq!(result.pixel_values.ndim(), 2);

        let (gt, gh, gw) = processor.calculate_grid_thw(448, 448, 4);
        assert_eq!(gt, 2);
        assert_eq!(result.pixel_values.shape()[0], gt * gh * gw);

        // patch_features = C(3) * temporal(2) * patch(14) * patch(14)
        assert_eq!(result.pixel_values.shape()[1], 3 * 2 * 14 * 14);

        // video_grid_thw present with shape [num_videos, 3]
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("video_grid_thw")
        {
            assert_eq!(shape, &[1, 3]);
            assert_eq!(data, &[gt as i64, gh as i64, gw as i64]);
        } else {
            panic!("Expected video_grid_thw to be IntTensor");
        }

        // video_sizes records (width, height, frames)
        assert_eq!(result.video_sizes, vec![(448, 448, 4)]);
        assert_eq!(result.num_video_tokens.len(), 1);
        assert_eq!(
            result.num_video_tokens[0],
            processor.calculate_tokens_from_grid(gt, gh, gw)
        );
    }

    #[test]
    fn test_qwen2_vl_preprocess_video_odd_frames_padded() {
        let processor = Qwen2VLProcessor::new();
        let config = video_config();

        // 3 frames -> padded to 4 -> grid_t = 2
        let frames: Vec<DynamicImage> = (0..3)
            .map(|_| create_test_image(448, 448, Rgb([128, 128, 128])))
            .collect();
        let result = VideoPreProcessor::preprocess(&processor, &[frames], &config).unwrap();

        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("video_grid_thw")
        {
            assert_eq!(data[0], 2); // grid_t after padding 3 -> 4
        } else {
            panic!("Expected video_grid_thw");
        }

        // Original frame count is preserved in video_sizes (not the padded count).
        assert_eq!(result.video_sizes[0].2, 3);
    }

    #[test]
    fn test_qwen2_vl_preprocess_video_batch() {
        let processor = Qwen2VLProcessor::new();
        let config = video_config();

        let video_a: Vec<DynamicImage> = (0..2)
            .map(|_| create_test_image(448, 448, Rgb([100, 100, 100])))
            .collect();
        let video_b: Vec<DynamicImage> = (0..4)
            .map(|_| create_test_image(224, 336, Rgb([50, 60, 70])))
            .collect();
        let result =
            VideoPreProcessor::preprocess(&processor, &[video_a, video_b], &config).unwrap();

        assert_eq!(result.num_videos(), 2);
        assert_eq!(result.num_video_tokens.len(), 2);

        // patches_per_video sums to the total patch rows.
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("patches_per_video")
        {
            assert_eq!(shape, &[2]);
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.pixel_values.shape()[0]);
        } else {
            panic!("Expected patches_per_video to be IntTensor");
        }
    }

    #[test]
    fn test_qwen2_vl_calculate_num_video_tokens_matches_preprocess() {
        let processor = Qwen2VLProcessor::new();
        let config = video_config();

        let frames: Vec<DynamicImage> = (0..5)
            .map(|_| create_test_image(640, 480, Rgb([128, 128, 128])))
            .collect();
        let n_frames = frames.len() as u32;
        let result = VideoPreProcessor::preprocess(&processor, &[frames], &config).unwrap();

        let predicted =
            VideoPreProcessor::calculate_num_tokens(&processor, 640, 480, n_frames, &config);
        assert_eq!(predicted, result.num_video_tokens[0]);
    }

    #[test]
    fn test_qwen2_vl_frame_sampler_uses_all_frames_by_default() {
        let sampler = Qwen2VLFrameSampler::new();
        assert_eq!(
            sampler.sample_indices(&sample_video_metadata()).unwrap(),
            (0..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_qwen2_vl_frame_sampler_rounds_num_frames_like_transformers() {
        let sampler = Qwen2VLFrameSampler::from_preprocessor_config(&PreProcessorConfig {
            do_sample_frames: Some(true),
            num_frames: Some(5),
            temporal_patch_size: Some(2),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            sampler.sample_indices(&sample_video_metadata()).unwrap(),
            vec![0, 6, 12, 18]
        );
    }

    #[test]
    fn test_qwen2_vl_frame_sampler_supports_fps_sampling() {
        let sampler = Qwen2VLFrameSampler::from_preprocessor_config(&PreProcessorConfig {
            do_sample_frames: Some(true),
            fps: Some(1.5),
            temporal_patch_size: Some(2),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            sampler.sample_indices(&sample_video_metadata()).unwrap(),
            vec![0, 4, 8, 12, 16, 20]
        );
    }

    #[test]
    fn test_qwen2_vl_frame_sampler_rejects_conflicting_config() {
        let err = Qwen2VLFrameSampler::from_preprocessor_config(&PreProcessorConfig {
            do_sample_frames: Some(true),
            num_frames: Some(8),
            fps: Some(2.0),
            ..Default::default()
        })
        .unwrap_err();

        assert!(matches!(err, VideoDecodeError::InvalidSampling(_)));
    }

    #[test]
    fn test_qwen2_vl_preprocess_video_empty_batch_errors() {
        let processor = Qwen2VLProcessor::new();
        let config = video_config();
        let videos: Vec<Vec<DynamicImage>> = Vec::new();
        assert!(VideoPreProcessor::preprocess(&processor, &videos, &config).is_err());
    }
}
