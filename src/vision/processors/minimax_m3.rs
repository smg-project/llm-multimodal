//! MiniMax-M3 VL image processor.
//!
//! Ported from HuggingFace `MiniMaxM3VLImageProcessor`. The model documents this
//! as "Copied from Qwen2VLImageProcessorFast with resize changed to vLLM style":
//! the patchify pipeline (rescale → normalize → reshape into
//! `[grid_t, grid_h, grid_w, ...]` patches) is identical to Qwen2-VL, so we reuse
//! [`QwenVLProcessorBase`] for it. The only difference is the resize step.
//!
//! # vLLM-style resize (`get_hw_multiple_of`)
//!
//! Unlike Qwen's smart-resize (which targets a min/max *pixel* budget), MiniMax:
//!
//! 1. Rounds each dimension **up** to a multiple of `patch_size * merge_size`.
//! 2. If either dimension exceeds `max_size` (width, height), scales the image
//!    down to fit while preserving aspect ratio, then re-aligns (rounds up) to
//!    the factor.
//!
//! There is no lower (min-pixels) bound. `max_size` is inferred from the
//! processor's `size` (default `{height: 672, width: 672}`) and must itself be
//! divisible by the factor.

use image::{imageops::FilterType, DynamicImage, GenericImageView};

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase};
use crate::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{pil_to_filter, resize, to_tensor, to_tensor_and_normalize, TransformError},
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

/// Default vLLM-style resize bound as `(max_width, max_height)`.
///
/// Inferred from the HF default `size = {"height": 672, "width": 672}`. Must be
/// divisible by the factor (`patch_size * merge_size`); 672 / 28 = 24.
pub const DEFAULT_MAX_SIZE: (usize, usize) = (672, 672);

/// Round `x` up to the nearest multiple of `multiple`.
#[inline]
fn ceil_to_multiple(x: usize, multiple: usize) -> usize {
    if multiple == 0 || x % multiple == 0 {
        x
    } else {
        x + (multiple - x % multiple)
    }
}

/// MiniMax-M3 VL image processor.
///
/// Wraps [`QwenVLProcessorBase`] for the shared patchify/grid logic and overrides
/// the resize with the vLLM-style [`Self::vllm_resize`].
#[derive(Debug, Clone)]
pub struct MiniMaxM3Processor {
    inner: QwenVLProcessorBase,
    /// vLLM-style resize bound as `(max_width, max_height)`.
    max_size: (usize, usize),
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
    /// - max_size: (672, 672)
    /// - normalization: CLIP mean/std
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
            DEFAULT_MAX_SIZE,
        )
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        max_size: (usize, usize),
    ) -> Self {
        Self {
            // min_pixels / max_pixels are unused: MiniMax never calls smart_resize.
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels: 0,
                max_pixels: usize::MAX,
                temporal_patch_size,
                mean: MINIMAX_M3_MEAN,
                std: MINIMAX_M3_STD,
                model_name: "minimax-m3",
            }),
            max_size,
        }
    }

    /// Create a processor from a HuggingFace preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let patch_size = config.get_patch_size(DEFAULT_PATCH_SIZE);
        let merge_size = config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
        let temporal_patch_size = config
            .temporal_patch_size
            .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);
        // `get_target_size` returns (height, width); max_size is (width, height).
        let max_size = config
            .get_target_size()
            .map(|(h, w)| (w as usize, h as usize))
            .unwrap_or(DEFAULT_MAX_SIZE);
        Self::with_config(patch_size, merge_size, temporal_patch_size, max_size)
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

    /// Get the vLLM-style resize bound as `(max_width, max_height)`.
    pub fn max_size(&self) -> (usize, usize) {
        self.max_size
    }

    /// Get the factor for dimension alignment (`patch_size * merge_size`).
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.inner.get_factor()
    }

    /// Compute the target `(new_width, new_height)`, both multiples of `factor`,
    /// scaled to fit within `max_size`. Mirrors HF `get_hw_multiple_of` for the
    /// `(max_w, max_h)` tuple case (the only one MiniMax uses).
    fn get_hw_multiple_of(&self, width: usize, height: usize, factor: usize) -> (usize, usize) {
        let (max_w, max_h) = self.max_size;
        let mut new_w = ceil_to_multiple(width, factor);
        let mut new_h = ceil_to_multiple(height, factor);

        if new_w > max_w || new_h > max_h {
            // Scale down to fit within max_size while maintaining aspect ratio.
            // (new_w * max_w) // new_w == max_w, kept explicit to match HF.
            let new_w_ = max_w.min(new_w * max_h / new_h);
            let new_h_ = (new_h * max_w / new_w).min(max_h);
            // Re-align (round up) to the factor.
            new_w = ceil_to_multiple(new_w_, factor);
            new_h = ceil_to_multiple(new_h_, factor);
        }

        (new_w, new_h)
    }

    /// vLLM-style resize. Returns `(new_height, new_width)`, both multiples of the
    /// alignment factor and bounded by `max_size`.
    pub fn vllm_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = self.get_factor();
        let (new_w, new_h) = self.get_hw_multiple_of(width, height, factor);
        (new_h, new_w)
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

    /// Pick the resampling filter, defaulting to BICUBIC (the HF default) when the
    /// config doesn't specify one.
    fn resize_filter(config: &PreProcessorConfig) -> FilterType {
        match config.resampling {
            Some(r) => pil_to_filter(Some(r)),
            None => FilterType::CatmullRom, // BICUBIC
        }
    }
}

impl ImagePreProcessor for MiniMaxM3Processor {
    fn default_mean(&self) -> [f64; 3] {
        MINIMAX_M3_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        MINIMAX_M3_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let image_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();

        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let filter = Self::resize_filter(config);

        let patch_size = self.patch_size();
        let temporal_patch_size = self.temporal_patch_size();
        let patch_features = 3 * temporal_patch_size * patch_size * patch_size;

        let mut all_patches: Vec<f32> = Vec::new();
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut num_img_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (w, h) = image.dimensions();
            let (target_h, target_w) = self.vllm_resize(h as usize, w as usize);

            // Resize to the image's own target size (skip if dimensions match).
            let (tw32, th32) = (target_w as u32, target_h as u32);
            let needs_resize = config.do_resize.unwrap_or(true) && (w != tw32 || h != th32);
            let resized;
            let img_ref = if needs_resize {
                resized = resize(image, tw32, th32, filter);
                &resized
            } else {
                image
            };

            // Grid dimensions based on the target size (T=1 for a single image).
            let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(target_h, target_w, 1);
            grid_thw_data.push(grid_t as i64);
            grid_thw_data.push(grid_h as i64);
            grid_thw_data.push(grid_w as i64);

            let num_patches = grid_t * grid_h * grid_w;
            num_img_tokens.push(self.calculate_tokens_from_grid(grid_t, grid_h, grid_w));

            // Convert to tensor [C, H, W] (+ normalize) in one fused pass.
            let tensor = if config.do_normalize.unwrap_or(true) {
                to_tensor_and_normalize(img_ref, &mean, &std)
            } else {
                to_tensor(img_ref)
            };

            // Patchify directly into the shared buffer. The single image frame is
            // repeated across `temporal_patch_size` inside `patchify_into`, which
            // matches HF's "repeat last frame to fill the temporal dim".
            self.inner
                .patchify_into(&tensor, grid_t, grid_h, grid_w, &mut all_patches)?;
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let pixel_values =
            ndarray::Array2::from_shape_vec((total_patches, patch_features), all_patches).map_err(
                |e| {
                    TransformError::ShapeError(format!(
                        "Failed to create patchified pixel_values [{total_patches}, {patch_features}]: {e}"
                    ))
                },
            )?;

        let result =
            PreprocessedImages::new_dynamic(pixel_values.into_dyn(), num_img_tokens, image_sizes)
                .with_extra(
                    "image_grid_thw",
                    ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                );

        Ok(result)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        let (new_height, new_width) = self.vllm_resize(height as usize, width as usize);
        let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(new_height, new_width, 1);
        self.calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }

    fn model_name(&self) -> &'static str {
        "minimax-m3"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Dynamic resolution: no fixed output size.
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::image_processor::ModelSpecificValue;

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
        assert_eq!(p.max_size(), (672, 672));
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
        // 100x100 -> ceil to 28 multiples -> 112x112 (no scaling, under 672).
        let (h, w) = p.vllm_resize(100, 100);
        assert_eq!(h, 112);
        assert_eq!(w, 112);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_resize_exact_max_size() {
        let p = MiniMaxM3Processor::new();
        // 672x672 is already factor-aligned and at the bound: unchanged.
        let (h, w) = p.vllm_resize(672, 672);
        assert_eq!((h, w), (672, 672));
        // Sanity: 672x672 -> grid 48x48 -> 576 tokens == config image_seq_length.
        let (t, gh, gw) = p.calculate_grid_thw(h, w, 1);
        assert_eq!(p.calculate_tokens_from_grid(t, gh, gw), 576);
    }

    #[test]
    fn test_resize_scales_down_preserving_aspect() {
        let p = MiniMaxM3Processor::new();
        // 800x600 (w x h). ceil -> 812x616, exceeds max_w=672 -> scale down.
        // Expected (height, width) = (532, 672); see hand-computation in the port.
        let (h, w) = p.vllm_resize(600, 800);
        assert_eq!(w, 672);
        assert_eq!(h, 532);
        assert!(w <= 672 && h <= 672);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_calculate_num_tokens() {
        let p = MiniMaxM3Processor::new();
        let config = PreProcessorConfig::default();
        // 672x672 -> 576 tokens.
        assert_eq!(p.calculate_num_tokens(672, 672, &config), 576);
        // 800x600 -> grid 38x48 -> (38*48)/4 = 456.
        assert_eq!(p.calculate_num_tokens(800, 600, &config), 456);
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
        assert_eq!(result.pixel_values.ndim(), 2);
        assert_eq!(result.pixel_values.shape()[1], 3 * 2 * 14 * 14); // 1176
        assert!(result.pixel_values.shape()[0] > 0);

        assert!(result.model_specific.contains_key("image_grid_thw"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.num_img_tokens[0] > 0);
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

        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);
        assert_eq!(result.pixel_values.ndim(), 2);

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
            assert_eq!(total as usize, result.pixel_values.shape()[0]);
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
        let mut size = std::collections::HashMap::new();
        size.insert("height".to_string(), 1008u32);
        size.insert("width".to_string(), 672u32);
        let config = PreProcessorConfig {
            merge_size: Some(2),
            temporal_patch_size: Some(2),
            size: Some(size),
            ..Default::default()
        };
        let p = MiniMaxM3Processor::from_preprocessor_config(&config);
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        // max_size is (width, height) = (672, 1008).
        assert_eq!(p.max_size(), (672, 1008));
    }
}
