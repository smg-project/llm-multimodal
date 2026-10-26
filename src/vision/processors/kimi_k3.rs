//! Kimi-K3 wrapper around the shared MoonViT processor.

use image::DynamicImage;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

use super::kimi_base::{
    KimiMoonViTProcessor, MoonViTConfig, TransparentBgConfig, TransparentBgFillStage,
};

pub const KIMI_K3_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const KIMI_K3_STD: [f64; 3] = [0.5, 0.5, 0.5];
const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_MERGE_SIZE: usize = 2;
const DEFAULT_IN_PATCH_LIMIT: usize = 65536;
const DEFAULT_PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

#[derive(Debug, Clone)]
pub struct KimiK3Processor {
    inner: KimiMoonViTProcessor,
}

impl Default for KimiK3Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiK3Processor {
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
        let fill_stage = Self::extra::<String>(config, "transparent_bg_fill_stage").map_or(
            TransparentBgFillStage::BeforeResize,
            |stage| {
                if stage == "after_resize" {
                    TransparentBgFillStage::AfterResize
                } else {
                    TransparentBgFillStage::BeforeResize
                }
            },
        );
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
                fixed_output_tokens: Self::extra(config, "fixed_output_tokens"),
                transparent_bg_config: Self::extra::<TransparentBgConfig>(
                    config,
                    "transparent_bg_config",
                ),
                transparent_bg_fill_stage: fill_stage,
            })
            .with_preprocessor_config(config),
        }
    }

    fn extra<T: serde::de::DeserializeOwned>(config: &PreProcessorConfig, key: &str) -> Option<T> {
        config.get_extra(key).or_else(|| {
            config
                .extra
                .get("media_proc_cfg")
                .and_then(|media_cfg| media_cfg.get(key))
                .and_then(|value| serde_json::from_value(value.clone()).ok())
        })
    }
}

impl VisionPreProcessor for KimiK3Processor {
    fn default_mean(&self) -> [f64; 3] {
        KIMI_K3_MEAN
    }
    fn default_std(&self) -> [f64; 3] {
        KIMI_K3_STD
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
        "kimi-k3"
    }
    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgba, RgbaImage};

    #[test]
    fn composites_alpha_with_checkpoint_chessboard() {
        let config = PreProcessorConfig::from_json(r#"{"media_proc_cfg":{"patch_size":14,"merge_kernel_size":2,"transparent_bg_config":{"pattern":"chessboard","chessboard_square_size":8,"chessboard_square_on_top_left":true,"chessboard_white_value":255,"chessboard_gray_value":180},"transparent_bg_fill_stage":"after_resize","image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}}"#).unwrap();
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([0, 0, 0, 0])));
        let output = KimiK3Processor::from_preprocessor_config(&config)
            .preprocess(&[image])
            .unwrap();
        assert!((output.encoder_input_flat()[0] - 1.0).abs() < 1e-6);
    }
}
