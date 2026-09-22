//! Kimi-K2.5 MoonViT vision processor.

use image::DynamicImage;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

use super::kimi_base::{KimiMoonViTProcessor, MoonViTConfig, TransparentBgFillStage};

pub const KIMI_K25_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K25_STD: [f64; 3] = [0.5, 0.5, 0.5];
pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
pub const DEFAULT_IN_PATCH_LIMIT: usize = 16384;
pub const DEFAULT_PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

#[derive(Debug, Clone)]
pub struct KimiK25Processor {
    inner: KimiMoonViTProcessor,
}

impl Default for KimiK25Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK25Processor {
    pub fn new() -> Self {
        Self {
            inner: KimiMoonViTProcessor::new(MoonViTConfig {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                in_patch_limit: DEFAULT_IN_PATCH_LIMIT,
                patch_limit_on_one_side: DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
                fixed_output_tokens: None,
                transparent_bg_config: None,
                transparent_bg_fill_stage: TransparentBgFillStage::BeforeResize,
            }),
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: KimiMoonViTProcessor::new(MoonViTConfig {
                patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                in_patch_limit: config
                    .get_extra("in_patch_limit")
                    .unwrap_or(DEFAULT_IN_PATCH_LIMIT),
                patch_limit_on_one_side: config
                    .get_extra("patch_limit_on_one_side")
                    .unwrap_or(DEFAULT_PATCH_LIMIT_ON_ONE_SIDE),
                fixed_output_tokens: None,
                transparent_bg_config: None,
                transparent_bg_fill_stage: TransparentBgFillStage::BeforeResize,
            })
            .with_preprocessor_config(config),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }
    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    #[cfg(test)]
    pub(crate) fn factor(&self) -> usize {
        self.inner.factor()
    }
    #[cfg(test)]
    pub(crate) fn compute_resize_config(
        &self,
        width: usize,
        height: usize,
    ) -> super::kimi_base::ResizeConfig {
        self.inner.compute_resize_config(width, height)
    }
    #[cfg(test)]
    pub(crate) fn in_patch_limit(&self) -> usize {
        self.inner.in_patch_limit()
    }
    #[cfg(test)]
    pub(crate) fn patch_limit_on_one_side(&self) -> usize {
        self.inner.patch_limit_on_one_side()
    }
}

impl VisionPreProcessor for KimiK25Processor {
    fn default_mean(&self) -> [f64; 3] {
        KIMI_K25_MEAN
    }
    fn default_std(&self) -> [f64; 3] {
        KIMI_K25_STD
    }
    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess_images(images)
    }
    fn calculate_num_tokens(&self, width: u32, height: u32) -> usize {
        self.inner.calculate_num_tokens(width, height)
    }
    fn model_name(&self) -> &'static str {
        "kimi-k2.5"
    }
    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}
