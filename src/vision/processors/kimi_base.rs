//! Shared MoonViT NaViT processor for the Kimi vision model family.
//!
//! K2.5 and K3 provide their own settings and implement `VisionPreProcessor`.
//! This module owns their shared resize, alpha compositing, padding,
//! normalization and patch extraction pipeline.
//!
//! 1. Compute scale to fit within patch limits (never upscale)
//! 2. Resize with BICUBIC interpolation
//! 3. Zero-pad to make dimensions divisible by factor (patch_size * merge_size)
//! 4. Normalize with [0.5, 0.5, 0.5] mean/std
//! 5. Extract patches as [N, C, patch_size, patch_size]
//!
//! Kimi resizes then zero-pads to make dimensions divisible by the alignment
//! factor (patch_size * merge_size). The model was trained with zero-padded
//! images, so using direct resize-to-aligned would degrade image quality.

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::Array3;
use serde::Deserialize;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs},
    scratch,
    transforms::{self, TransformError},
};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct TransparentBgConfig {
    pattern: String,
    chessboard_square_size: usize,
    chessboard_square_on_top_left: bool,
    chessboard_white_value: u8,
    chessboard_gray_value: u8,
}

impl Default for TransparentBgConfig {
    fn default() -> Self {
        Self {
            pattern: "black".to_string(),
            chessboard_square_size: 16,
            chessboard_square_on_top_left: true,
            chessboard_white_value: 255,
            chessboard_gray_value: 200,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransparentBgFillStage {
    BeforeResize,
    AfterResize,
}

#[derive(Debug, Clone)]
pub(crate) struct MoonViTConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub in_patch_limit: usize,
    pub patch_limit_on_one_side: usize,
    pub fixed_output_tokens: Option<usize>,
    pub transparent_bg_config: Option<TransparentBgConfig>,
    pub transparent_bg_fill_stage: TransparentBgFillStage,
}

/// Kimi-K2.5 resize configuration for a single image.
pub(crate) struct ResizeConfig {
    new_width: usize,
    new_height: usize,
    pad_width: usize,
    pad_height: usize,
    num_tokens: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct KimiMoonViTProcessor {
    mean: [f64; 3],
    std: [f64; 3],
    patch_size: usize,
    merge_size: usize,
    in_patch_limit: usize,
    patch_limit_on_one_side: usize,
    fixed_output_tokens: Option<usize>,
    transparent_bg_config: Option<TransparentBgConfig>,
    transparent_bg_fill_stage: TransparentBgFillStage,
}

impl KimiMoonViTProcessor {
    pub(crate) fn with_preprocessor_config(mut self, config: &PreProcessorConfig) -> Self {
        self.mean = config.get_image_mean();
        self.std = config.get_image_std();
        self
    }

    pub(crate) fn new(config: MoonViTConfig) -> Self {
        Self {
            mean: PreProcessorConfig::default().get_image_mean(),
            std: PreProcessorConfig::default().get_image_std(),
            patch_size: config.patch_size,
            merge_size: config.merge_size,
            in_patch_limit: config.in_patch_limit,
            patch_limit_on_one_side: config.patch_limit_on_one_side,
            fixed_output_tokens: config.fixed_output_tokens,
            transparent_bg_config: config.transparent_bg_config,
            transparent_bg_fill_stage: config.transparent_bg_fill_stage,
        }
    }

    pub(crate) fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub(crate) fn merge_size(&self) -> usize {
        self.merge_size
    }

    #[cfg(test)]
    pub(crate) fn in_patch_limit(&self) -> usize {
        self.in_patch_limit
    }

    #[cfg(test)]
    pub(crate) fn patch_limit_on_one_side(&self) -> usize {
        self.patch_limit_on_one_side
    }

    #[inline]
    pub(crate) fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Compute resize dimensions and padding, matching HF `navit_resize_image`.
    ///
    /// Never upscales (scale capped at 1.0). Pads with zeros to align to factor.
    pub(crate) fn compute_resize_config(&self, width: usize, height: usize) -> ResizeConfig {
        let ps = self.patch_size;
        let patches_w = (width / ps).max(1) as f64;
        let patches_h = (height / ps).max(1) as f64;

        let s1 = (self.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let s2 = (self.patch_limit_on_one_side * ps) as f64 / width as f64;
        let s3 = (self.patch_limit_on_one_side * ps) as f64 / height as f64;
        let scale = f64::min(1.0, f64::min(s1, f64::min(s2, s3)));

        let new_w = ((width as f64 * scale) as usize).max(1);
        let new_h = ((height as f64 * scale) as usize).max(1);
        let new_w = new_w.min(self.patch_limit_on_one_side * ps);
        let new_h = new_h.min(self.patch_limit_on_one_side * ps);

        let factor = self.factor();
        let pad_width = (factor - new_w % factor) % factor;
        let pad_height = (factor - new_h % factor) % factor;

        let token_height = (new_h + pad_height) / factor;
        let token_width = (new_w + pad_width) / factor;
        let num_tokens = self
            .fixed_output_tokens
            .unwrap_or(token_height * token_width);

        ResizeConfig {
            new_width: new_w,
            new_height: new_h,
            pad_width,
            pad_height,
            num_tokens,
        }
    }

    fn background_pixel(config: &TransparentBgConfig, x: usize, y: usize) -> Rgb<u8> {
        match config.pattern.as_str() {
            "white" => Rgb([255; 3]),
            "gray" => Rgb([128; 3]),
            "chessboard" => {
                let size = config.chessboard_square_size.max(1);
                let alternate =
                    (x / size + y / size).is_multiple_of(2) != config.chessboard_square_on_top_left;
                if alternate {
                    Rgb([config.chessboard_gray_value; 3])
                } else {
                    Rgb([config.chessboard_white_value; 3])
                }
            }
            _ => Rgb([0; 3]),
        }
    }

    fn composite_transparent_background(
        image: &DynamicImage,
        config: Option<&TransparentBgConfig>,
    ) -> DynamicImage {
        let Some(config) = config else {
            return DynamicImage::ImageRgb8(image.to_rgb8());
        };
        if !image.color().has_alpha() {
            return DynamicImage::ImageRgb8(image.to_rgb8());
        }
        let rgba = image.to_rgba8();
        let mut rgb = RgbImage::new(rgba.width(), rgba.height());
        for (x, y, pixel) in rgba.enumerate_pixels() {
            let background = Self::background_pixel(config, x as usize, y as usize);
            let alpha = pixel[3] as u16;
            let inverse_alpha = 255 - alpha;
            rgb.put_pixel(
                x,
                y,
                Rgb(std::array::from_fn(|c| {
                    ((alpha * pixel[c] as u16 + inverse_alpha * background[c] as u16) / 255) as u8
                })),
            );
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Fused resize + zero-pad + normalize into a single [C, H_padded, W_padded] tensor.
    ///
    /// Avoids intermediate allocations by:
    /// 1. Allocating the final padded canvas directly
    /// 2. Pre-filling with normalized black (bias value)
    /// 3. Deinterleaving + normalizing the image region in one pass
    fn resize_pad_and_normalize(
        &self,
        image: &DynamicImage,
        cfg: &ResizeConfig,
        mean: &[f64; 3],
        std: &[f64; 3],
    ) -> Array3<f32> {
        let canvas_h = cfg.new_height + cfg.pad_height;
        let canvas_w = cfg.new_width + cfg.pad_width;

        let source = if self.transparent_bg_fill_stage == TransparentBgFillStage::AfterResize {
            image.clone()
        } else if self.transparent_bg_config.is_some() {
            Self::composite_transparent_background(image, self.transparent_bg_config.as_ref())
        } else {
            image.clone()
        };

        // Resize using SIMD-accelerated BICUBIC (fast_image_resize)
        let resized = transforms::resize(
            &source,
            cfg.new_width as u32,
            cfg.new_height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let resized = if self.transparent_bg_fill_stage == TransparentBgFillStage::AfterResize {
            Self::composite_transparent_background(&resized, self.transparent_bg_config.as_ref())
        } else {
            resized
        };

        let (img_w, img_h, raw) = transforms::rgb_bytes(&resized);
        let canvas_pixels = canvas_h * canvas_w;

        // Precompute fused scale/bias: pixel/255 → normalized
        // output[c][i] = raw[i*3+c] / 255.0 * (1/std[c]) + (-mean[c]/std[c])
        let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
        let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32));

        // Pooled: this per-image CHW buffer (tens of MB) is recycled by the
        // caller after patch extraction, keeping its pages mapped and hot.
        let mut data = scratch::take_f32(3 * canvas_pixels);
        let (r_plane, rest) = data.split_at_mut(canvas_pixels);
        let (g_plane, b_plane) = rest.split_at_mut(canvas_pixels);

        // Pre-fill with normalized black: (0/255 - mean) / std = bias
        r_plane.fill(bias[0]);
        g_plane.fill(bias[1]);
        b_plane.fill(bias[2]);

        // Overwrite image region row-by-row using vectorized deinterleave
        let rw = img_w.min(canvas_w);
        let rh = img_h.min(canvas_h);
        for y in 0..rh {
            let src_row = &raw[y * img_w * 3..y * img_w * 3 + rw * 3];
            let dst_offset = y * canvas_w;
            transforms::deinterleave_rgb_to_planes(
                src_row,
                &mut r_plane[dst_offset..dst_offset + rw],
                &mut g_plane[dst_offset..dst_offset + rw],
                &mut b_plane[dst_offset..dst_offset + rw],
                scale,
                bias,
            );
        }

        #[expect(
            clippy::expect_used,
            reason = "data has exactly 3*canvas_h*canvas_w elements by construction"
        )]
        Array3::from_shape_vec((3, canvas_h, canvas_w), data)
            .expect("shape matches pre-allocated buffer")
    }

    /// Extract [C, patch_size, patch_size] patches from a contiguous [C, H, W] tensor.
    ///
    /// Uses row-based `copy_from_slice` instead of per-element indexing so the
    /// compiler can auto-vectorize the inner copy.
    /// Append this image's patches directly into `out` (no per-image intermediate
    /// Vec): `out` is the pooled batch buffer pre-sized for the whole request.
    fn extract_patches_into(tensor: &Array3<f32>, patch_size: usize, out: &mut Vec<f32>) {
        let channels = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];

        let grid_h = height / patch_size;
        let grid_w = width / patch_size;

        // Get contiguous slice for direct row addressing
        let flat = tensor.as_standard_layout();
        #[expect(
            clippy::expect_used,
            reason = "as_standard_layout guarantees contiguous C-order memory"
        )]
        let data = flat
            .as_slice()
            .expect("as_standard_layout guarantees contiguous memory");

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let h_start = gh * patch_size;
                let w_start = gw * patch_size;
                for c in 0..channels {
                    let plane_offset = c * height * width;
                    for ph in 0..patch_size {
                        let row_start = plane_offset + (h_start + ph) * width + w_start;
                        out.extend_from_slice(&data[row_start..row_start + patch_size]);
                    }
                }
            }
        }
    }
}

impl KimiMoonViTProcessor {
    pub(crate) fn preprocess_images(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();
        let mean = self.mean;
        let std = self.std;

        // Pre-size the pooled batch buffer exactly (patch_features per patch =
        // 3 * patch_size^2; this is the data plane's hottest allocation).
        let patch_features = 3 * self.patch_size * self.patch_size;
        let mut estimated_total = 0usize;
        for image in images {
            let (w, h) = image.dimensions();
            let cfg = self.compute_resize_config(w as usize, h as usize);
            let grid_h = (cfg.new_height + cfg.pad_height) / self.patch_size;
            let grid_w = (cfg.new_width + cfg.pad_width) / self.patch_size;
            estimated_total += grid_h * grid_w * patch_features;
        }
        let mut all_patches: Vec<f32> = scratch::take_f32_cap(estimated_total);
        let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut feature_token_counts = Vec::with_capacity(images.len());

        for image in images {
            let (w, h) = image.dimensions();
            let cfg = self.compute_resize_config(w as usize, h as usize);

            // Fused resize + pad + normalize in one pass (avoids 2 extra allocations)
            let tensor = self.resize_pad_and_normalize(image, &cfg, &mean, &std);

            let padded_h = cfg.new_height + cfg.pad_height;
            let padded_w = cfg.new_width + cfg.pad_width;
            let grid_h = padded_h / self.patch_size;
            let grid_w = padded_w / self.patch_size;
            let grid_t = 1usize;

            grid_thw_data.push(grid_t as i64);
            grid_thw_data.push(grid_h as i64);
            grid_thw_data.push(grid_w as i64);

            let num_patches = grid_h * grid_w;
            feature_token_counts.push(cfg.num_tokens);

            // Patchify directly into the pooled batch buffer, then recycle the
            // CHW tensor's storage (standard layout, offset 0) for the next image.
            Self::extract_patches_into(&tensor, self.patch_size, &mut all_patches);
            let (storage, _offset) = tensor.into_raw_vec_and_offset();
            scratch::give_f32(storage);
            patches_per_image.push(num_patches as i64);
        }

        let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
        let encoder_input = ndarray::Array4::from_shape_vec(
            (total_patches, 3, self.patch_size, self.patch_size),
            all_patches,
        )
        .map_err(|e| {
            TransformError::ShapeError(format!(
                "Failed to create encoder_input [{total_patches}, 3, {}, {}]: {e}",
                self.patch_size, self.patch_size
            ))
        })?;

        let result =
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "grid_thws",
                    ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                );

        Ok(result)
    }

    pub(crate) fn calculate_num_tokens(&self, width: u32, height: u32) -> usize {
        self.compute_resize_config(width as usize, height as usize)
            .num_tokens
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::{
        preprocessor_config::PatchSize,
        processor::VisionPreProcessor,
        processors::kimi_k25::{
            KimiK25Processor, DEFAULT_PATCH_LIMIT_ON_ONE_SIDE, KIMI_K25_MEAN, KIMI_K25_STD,
        },
    };

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_defaults() {
        let p = KimiK25Processor::new();
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
        assert_eq!(p.factor(), 28);
    }

    #[test]
    fn test_mean_std() {
        let p = KimiK25Processor::new();
        assert_eq!(p.default_mean(), KIMI_K25_MEAN);
        assert_eq!(p.default_std(), KIMI_K25_STD);
    }

    #[test]
    fn test_model_name() {
        assert_eq!(KimiK25Processor::new().model_name(), "kimi-k2.5");
    }

    #[test]
    fn test_resize_config_no_upscale() {
        let p = KimiK25Processor::new();
        // Small image should NOT be upscaled (scale capped at 1.0)
        let cfg = p.compute_resize_config(100, 100);
        assert!(cfg.new_width <= 100);
        assert!(cfg.new_height <= 100);
        // Padded dimensions must be factor-aligned
        assert_eq!((cfg.new_height + cfg.pad_height) % 28, 0);
        assert_eq!((cfg.new_width + cfg.pad_width) % 28, 0);
    }

    #[test]
    fn test_resize_config_large_image_downscaled() {
        let p = KimiK25Processor::new();
        // Large image should be downscaled
        let cfg = p.compute_resize_config(4000, 3000);
        // Resized dimensions should be smaller than original
        assert!(cfg.new_width < 4000);
        assert!(cfg.new_height < 3000);
        // Per-side patch limit must be respected (HF assertion)
        let padded_h = cfg.new_height + cfg.pad_height;
        let padded_w = cfg.new_width + cfg.pad_width;
        assert!(padded_h / 14 <= DEFAULT_PATCH_LIMIT_ON_ONE_SIDE * 2);
        assert!(padded_w / 14 <= DEFAULT_PATCH_LIMIT_ON_ONE_SIDE * 2);
    }

    #[test]
    fn test_resize_config_matches_hf_reference() {
        let p = KimiK25Processor::new();
        // 600x400 image: scale=1.0 (small enough), resize to 600x400,
        // pad to (600+4=) → let's compute:
        // factor=28, 400 % 28 = 400 - 14*28 = 400-392 = 8, pad_h = 28-8 = 20
        // 600 % 28 = 600 - 21*28 = 600-588 = 12, pad_w = 28-12 = 16
        let cfg = p.compute_resize_config(600, 400);
        assert_eq!(cfg.new_width, 600);
        assert_eq!(cfg.new_height, 400);
        assert_eq!(cfg.pad_height, 20);
        assert_eq!(cfg.pad_width, 16);
        // Padded: 420 x 616, grid: 30 x 44, tokens: (30*44)/(2*2) = 330
        assert_eq!(cfg.num_tokens, 330);
    }

    #[test]
    fn test_preprocess_4d_output() {
        let config = PreProcessorConfig {
            do_normalize: Some(true),
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = KimiK25Processor::from_preprocessor_config(&config)
            .preprocess(&[image])
            .unwrap();

        // 4D output: [total_patches, 3, 14, 14]
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.encoder_input.shape()[1], 3);
        assert_eq!(result.encoder_input.shape()[2], 14);
        assert_eq!(result.encoder_input.shape()[3], 14);

        assert!(result.model_specific.contains_key("grid_thws"));
        assert!(result.model_specific.contains_key("patches_per_image"));
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let p = KimiK25Processor::new();
        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = p.preprocess(&images).unwrap();

        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.encoder_input.shape()[1], 3);

        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("grid_thws")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected grid_thws to be IntTensor");
        }

        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("patches_per_image")
        {
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.encoder_input.shape()[0]);
        }
    }

    #[test]
    fn test_calculate_num_tokens() {
        let p = KimiK25Processor::new();
        let tokens = p.calculate_num_tokens(600, 400);
        assert_eq!(tokens, 330);
    }

    #[test]
    fn test_from_preprocessor_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            ..Default::default()
        };
        let p = KimiK25Processor::from_preprocessor_config(&config);
        assert_eq!(p.patch_size(), 14);
        assert_eq!(p.merge_size(), 2);
    }

    #[test]
    fn test_zero_padding_applied() {
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };

        // 100x100 white image — after normalization: (255/255 - 0.5) / 0.5 = 1.0
        // Padded region: (0/255 - 0.5) / 0.5 = -1.0
        let image = create_test_image(100, 100, Rgb([255, 255, 255]));
        let result = KimiK25Processor::from_preprocessor_config(&config)
            .preprocess(&[image])
            .unwrap();

        let flat = result.encoder_input_flat();
        // Padded region should be normalized black (-1.0)
        let has_neg_ones = flat.iter().any(|&v| (v - (-1.0)).abs() < 1e-6);
        assert!(
            has_neg_ones,
            "Expected normalized-black padding (-1.0) in output"
        );

        // Image region should be normalized white (1.0)
        let has_ones = flat.iter().any(|&v| (v - 1.0).abs() < 1e-6);
        assert!(
            has_ones,
            "Expected normalized-white image values (1.0) in output"
        );
    }

    #[test]
    fn test_preprocess_tiny_image() {
        // 1x1 image should not panic — padded to 28x28
        let config = PreProcessorConfig {
            image_mean: Some(KIMI_K25_MEAN.to_vec()),
            image_std: Some(KIMI_K25_STD.to_vec()),
            ..Default::default()
        };
        let image = create_test_image(1, 1, Rgb([128, 128, 128]));
        let result = KimiK25Processor::from_preprocessor_config(&config)
            .preprocess(&[image])
            .unwrap();
        assert_eq!(result.encoder_input.ndim(), 4);
        assert!(result.encoder_input.shape()[0] > 0);
        assert!(result.feature_token_counts[0] > 0);
    }

    #[test]
    fn test_preprocess_empty_batch_returns_error() {
        let p = KimiK25Processor::new();
        let result = p.preprocess(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_preprocessor_config_reads_limits() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            extra: [
                ("in_patch_limit".to_string(), serde_json::json!(8192)),
                (
                    "patch_limit_on_one_side".to_string(),
                    serde_json::json!(256),
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let p = KimiK25Processor::from_preprocessor_config(&config);
        assert_eq!(p.in_patch_limit(), 8192);
        assert_eq!(p.patch_limit_on_one_side(), 256);
    }
}
