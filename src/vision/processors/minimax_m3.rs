//! MiniMax-M3 VL image processor.
//!
//! Ported from HuggingFace `MiniMaxM3VLImageProcessor`. The patchify pipeline
//! (rescale → normalize → reshape into `[grid_t, grid_h, grid_w, ...]` patches)
//! is identical to Qwen2-VL, so we reuse [`QwenVLProcessorBase`] for it.
//!
//! MiniMax-M3 uses Qwen-style smart resize with `max_pixels = 672 * 672`.

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

/// CLIP normalization mean values used by MiniMax-M3 VL.
pub const MINIMAX_M3_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values used by MiniMax-M3 VL.
pub const MINIMAX_M3_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default vision encoder patch size.
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default spatial merge size (token reduction).
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (for video frames; images repeat the single frame).
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Default minimum pixels (4 * 28 * 28 = 3,136).
pub const DEFAULT_MIN_PIXELS: usize = 4 * 28 * 28;

/// Default maximum pixels (672 * 672 = 451,584).
pub const DEFAULT_MAX_PIXELS: usize = 672 * 672;

/// MiniMax-M3 VL image processor.
///
/// Wraps [`QwenVLProcessorBase`] for the shared smart-resize, patchify, and grid
/// logic.
#[derive(Debug, Clone)]
pub struct MiniMaxM3Processor {
    inner: QwenVLProcessorBase,
}

impl Default for MiniMaxM3Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMaxM3Processor {
    /// Create a new MiniMax-M3 processor with default settings.
    ///
    /// Defaults:
    /// - patch_size: 14
    /// - merge_size: 2
    /// - temporal_patch_size: 2
    /// - min_pixels: 3,136
    /// - max_pixels: 451,584
    /// - normalization: CLIP mean/std
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
            DEFAULT_MIN_PIXELS,
            DEFAULT_MAX_PIXELS,
        )
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                video_min_pixels: min_pixels,
                video_max_pixels: max_pixels,
                video_resize_mode: QwenVideoResizeMode::TotalVolume,
                temporal_patch_size,
                mean: MINIMAX_M3_MEAN,
                std: MINIMAX_M3_STD,
                model_name: "minimax-m3",
            }),
        }
    }

    /// Create a processor from a HuggingFace preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let patch_size = config.get_patch_size(DEFAULT_PATCH_SIZE);
        let merge_size = config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
        let temporal_patch_size = config
            .temporal_patch_size
            .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);
        let min_pixels = config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS);
        let max_pixels = config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS);
        Self::with_config(
            patch_size,
            merge_size,
            temporal_patch_size,
            min_pixels,
            max_pixels,
        )
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }

    /// Get the factor for dimension alignment (`patch_size * merge_size`).
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.inner.get_factor()
    }

    /// Smart resize. Returns `(new_height, new_width)`.
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        self.inner.smart_resize(height, width)
    }

    /// Calculate the grid dimensions `(grid_t, grid_h, grid_w)` for an image.
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

    /// Build the effective processor for a request, applying any structural
    /// overrides from `config`; otherwise reuse the existing defaults.
    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.patch_size.is_some()
            || config.merge_size.is_some()
            || config.min_pixels.is_some()
            || config.max_pixels.is_some()
            || config.temporal_patch_size.is_some()
            || config.size.is_some()
        {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }
}

impl Deref for MiniMaxM3Processor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for MiniMaxM3Processor {
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
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let processor = self.with_preprocessor_config(config);
        processor.inner.preprocess(images, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        self.inner.calculate_num_tokens(width, height, _config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::processor::ModelSpecificValue;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_defaults() {
        let p = MiniMaxM3Processor::new();
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.temporal_patch_size(), 2);
        assert_eq!(p.get_factor(), 28); // 14 * 2
        assert_eq!(p.min_pixels(), DEFAULT_MIN_PIXELS);
        assert_eq!(p.max_pixels(), DEFAULT_MAX_PIXELS);
    }

    #[test]
    fn test_mean_std() {
        let p = MiniMaxM3Processor::new();
        assert_eq!(p.default_mean(), MINIMAX_M3_MEAN);
        assert_eq!(p.default_std(), MINIMAX_M3_STD);
    }

    #[test]
    fn test_model_name() {
        assert_eq!(MiniMaxM3Processor::new().model_name(), "minimax-m3");
    }

    #[test]
    fn test_resize_within_bounds_aligns_up() {
        let p = MiniMaxM3Processor::new();
        // 100x100 -> rounded to 28 multiples -> 112x112.
        let (h, w) = p.smart_resize(100, 100).unwrap();
        assert_eq!(h, 112);
        assert_eq!(w, 112);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_resize_exact_max_size() {
        let p = MiniMaxM3Processor::new();
        let (h, w) = p.smart_resize(672, 672).unwrap();
        assert_eq!((h, w), (672, 672));
        // Sanity: 672x672 -> grid 48x48 -> 576 tokens == config image_seq_length.
        let (t, gh, gw) = p.calculate_grid_thw(h, w, 1);
        assert_eq!(p.calculate_tokens_from_grid(t, gh, gw), 576);
    }

    #[test]
    fn test_resize_scales_down_by_max_pixels() {
        let p = MiniMaxM3Processor::new();

        // 500x500 -> 504x504 -> grid 36x36 -> 324 tokens.
        assert_eq!(p.smart_resize(500, 500).unwrap(), (504, 504));
        // 4000x3000 (w x h): scaled by max_pixels -> grid 40x54.
        let (h, w) = p.smart_resize(3000, 4000).unwrap();
        assert_eq!((h, w), (560, 756));
        let (t, gh, gw) = p.calculate_grid_thw(h, w, 1);
        assert_eq!((gh, gw), (40, 54));
        assert_eq!(p.calculate_tokens_from_grid(t, gh, gw), 540);
    }

    #[test]
    fn test_calculate_num_tokens() {
        let config = PreProcessorConfig::default();
        // 500x500 -> 324 tokens (verified against HF).
        let p = MiniMaxM3Processor::new();
        assert_eq!(p.calculate_num_tokens(500, 500, &config), 324);
        // 4000x3000 -> grid 40x54 -> (40*54)/4 = 540.
        assert_eq!(p.calculate_num_tokens(4000, 3000, &config), 540);
    }

    #[test]
    fn test_preprocess_single() {
        let p = MiniMaxM3Processor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(MINIMAX_M3_MEAN.to_vec()),
            image_std: Some(MINIMAX_M3_STD.to_vec()),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = p.preprocess(&[image], &config).unwrap();

        // pixel_values is patchified: [total_patches, patch_features].
        assert_eq!(result.encoder_input.ndim(), 2);
        assert_eq!(result.encoder_input.shape()[1], 3 * 2 * 14 * 14); // 1176
        assert!(result.encoder_input.shape()[0] > 0);

        assert!(result.model_specific.contains_key("image_grid_thw"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_preprocess_multiple() {
        let p = MiniMaxM3Processor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = p.preprocess(&images, &config).unwrap();

        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);
        assert_eq!(result.encoder_input.ndim(), 2);

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("image_grid_thw")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected image_grid_thw to be IntTensor");
        }

        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("patches_per_image")
        {
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.encoder_input.shape()[0]);
        } else {
            panic!("Expected patches_per_image to be IntTensor");
        }
    }

    #[test]
    fn test_preprocess_empty_batch_errors() {
        let p = MiniMaxM3Processor::new();
        let config = PreProcessorConfig::default();
        assert!(p.preprocess(&[], &config).is_err());
    }

    #[test]
    fn test_from_preprocessor_config() {
        let config = PreProcessorConfig {
            merge_size: Some(2),
            temporal_patch_size: Some(2),
            ..Default::default()
        };
        let p = MiniMaxM3Processor::from_preprocessor_config(&config);
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.max_pixels(), DEFAULT_MAX_PIXELS);
    }
}
