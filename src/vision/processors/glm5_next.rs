//! GLM-5.3-Flash image processor.

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{
    QwenImageResizeMode, QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode,
};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
pub const DEFAULT_MIN_IMAGE_TOKENS: usize = 16;
pub const DEFAULT_MAX_IMAGE_TOKENS: usize = 8000;

fn pixel_budget(
    min_image_tokens: usize,
    max_image_tokens: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> (usize, usize) {
    let factor = temporal_patch_size * (patch_size * merge_size).pow(2);
    (min_image_tokens * factor, max_image_tokens * factor)
}

#[derive(Debug, Clone)]
pub struct Glm5NextProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Glm5NextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Glm5NextProcessor {
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
            DEFAULT_MIN_IMAGE_TOKENS,
            DEFAULT_MAX_IMAGE_TOKENS,
        )
    }

    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_image_tokens: usize,
        max_image_tokens: usize,
    ) -> Self {
        let (min_pixels, max_pixels) = pixel_budget(
            min_image_tokens,
            max_image_tokens,
            patch_size,
            merge_size,
            temporal_patch_size,
        );
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
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "glm5_next",
            })
            .with_image_resize_mode(QwenImageResizeMode::GlmPad),
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let patch_size = config.get_patch_size(DEFAULT_PATCH_SIZE);
        let merge_size = config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
        let temporal_patch_size = config
            .temporal_patch_size
            .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);
        let min_image_tokens = config
            .get_extra::<usize>("min_image_tokens")
            .unwrap_or(DEFAULT_MIN_IMAGE_TOKENS);
        let max_image_tokens = config
            .get_extra::<usize>("max_image_tokens")
            .unwrap_or(DEFAULT_MAX_IMAGE_TOKENS);
        Self::with_config(
            patch_size,
            merge_size,
            temporal_patch_size,
            min_image_tokens,
            max_image_tokens,
        )
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.has_structural_overrides()
            || config.extra.contains_key("min_image_tokens")
            || config.extra.contains_key("max_image_tokens")
        {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }

    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }
}

impl Deref for Glm5NextProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for Glm5NextProcessor {
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

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = self.with_preprocessor_config(config);
        processor.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        let processor = self.with_preprocessor_config(config);
        processor.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbImage};

    use super::*;
    use crate::vision::processor::ModelSpecificValue;

    #[test]
    fn glm_defaults_match_checkpoint() {
        let processor = Glm5NextProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.temporal_patch_size(), 2);
        assert_eq!(processor.min_pixels(), 16 * 2 * 28 * 28);
        assert_eq!(processor.max_pixels(), 8000 * 2 * 28 * 28);
    }

    #[test]
    fn glm_448_image_has_256_tokens() {
        let processor = Glm5NextProcessor::new();
        let config = PreProcessorConfig::default();
        assert_eq!(processor.calculate_num_tokens(448, 448, &config), 256);
    }

    #[test]
    fn glm_preprocess_emits_grid_and_patch_counts() {
        let processor = Glm5NextProcessor::new();
        let image = DynamicImage::ImageRgb8(RgbImage::new(448, 448));
        let out = processor
            .preprocess(&[image], &PreProcessorConfig::default())
            .unwrap();
        assert_eq!(out.feature_token_counts, vec![256]);
        match out.model_specific.get("image_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape }) => {
                assert_eq!(shape, &vec![1, 3]);
                assert_eq!(data, &vec![1, 32, 32]);
            }
            other => panic!("unexpected image_grid_thw: {other:?}"),
        }
    }
}
