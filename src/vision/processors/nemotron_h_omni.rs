//! Nemotron-H Omni image preprocessing.

use image::{imageops::FilterType, DynamicImage};
use ndarray::Axis;
use serde_json::Value;

use crate::encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs};
use crate::vision::{
    transforms, PreProcessorConfig, TransformError, VisionPreProcessor, VisionPreprocessingContext,
};

#[derive(Debug, Clone)]
struct ProcessorConfig {
    patch_size: usize,
    reduction_factor: usize,
    min_num_patches: usize,
    max_num_patches: usize,
    max_model_len: usize,
    mean: [f64; 3],
    std: [f64; 3],
}

#[derive(Debug, Clone, Default)]
pub struct NemotronHOmniProcessor;

impl NemotronHOmniProcessor {
    pub fn new() -> Self {
        Self
    }

    fn processor_config(
        config: &PreProcessorConfig,
        context: &VisionPreprocessingContext,
    ) -> Result<ProcessorConfig, TransformError> {
        let model_config = &context.model_config;
        let patch_size = json_usize(model_config, &["patch_size"])
            .or_else(|| {
                config
                    .patch_size
                    .as_ref()
                    .and_then(|patch| patch.height.or(patch.width))
                    .map(|value| value as usize)
            })
            .ok_or_else(|| missing_config_field("patch_size"))?;
        let downsample_ratio = json_f64(model_config, &["downsample_ratio"])
            .or_else(|| config.extra.get("downsample_ratio").and_then(Value::as_f64))
            .ok_or_else(|| missing_config_field("downsample_ratio"))?;
        let reduction_factor = (1.0 / downsample_ratio).round() as usize;
        if patch_size == 0
            || downsample_ratio <= 0.0
            || (downsample_ratio * reduction_factor as f64 - 1.0).abs() > f64::EPSILON
            || reduction_factor != 2
        {
            return Err(TransformError::ShapeError(
                "Nemotron-H Omni requires a positive patch size and a 0.5 downsample ratio"
                    .to_string(),
            ));
        }

        let min_num_patches =
            json_usize(model_config, &["vision_config", "args", "min_num_patches"])
                .or_else(|| config_usize(config, "min_num_patches"))
                .ok_or_else(|| missing_config_field("vision_config.args.min_num_patches"))?;
        let max_num_patches =
            json_usize(model_config, &["vision_config", "args", "max_num_patches"])
                .or_else(|| config_usize(config, "max_num_patches"))
                .ok_or_else(|| missing_config_field("vision_config.args.max_num_patches"))?;
        let max_model_len = context
            .max_model_len
            .or_else(|| json_usize(model_config, &["max_position_embeddings"]))
            .or_else(|| json_usize(model_config, &["text_config", "max_position_embeddings"]))
            .or_else(|| config_usize(config, "max_model_len"))
            .ok_or_else(|| missing_config_field("max_model_len"))?;
        let mean = json_array3(model_config, &["norm_mean"])
            .or_else(|| slice_to_array3(config.image_mean.as_deref()))
            .ok_or_else(|| missing_config_field("norm_mean"))?;
        let std = json_array3(model_config, &["norm_std"])
            .or_else(|| slice_to_array3(config.image_std.as_deref()))
            .ok_or_else(|| missing_config_field("norm_std"))?;
        if min_num_patches == 0 || max_num_patches < min_num_patches {
            return Err(TransformError::ShapeError(
                "Nemotron-H Omni patch limits must be positive and ordered".to_string(),
            ));
        }
        if std.contains(&0.0) {
            return Err(TransformError::ShapeError(
                "Nemotron-H Omni normalization standard deviations must be nonzero".to_string(),
            ));
        }

        Ok(ProcessorConfig {
            patch_size,
            reduction_factor,
            min_num_patches,
            max_num_patches,
            max_model_len,
            mean,
            std,
        })
    }

    fn target_patch_grid(
        width: u32,
        height: u32,
        config: &ProcessorConfig,
        text_prompt_length: usize,
    ) -> (usize, usize) {
        let closest_h = (f64::from(height) / config.patch_size as f64 + 0.5)
            .round_ties_even()
            .max(1.0) as usize;
        let closest_w = (f64::from(width) / config.patch_size as f64 + 0.5)
            .round_ties_even()
            .max(1.0) as usize;
        let source_patches = closest_h.saturating_mul(closest_w).max(1);
        let post_shuffle_budget = config
            .max_model_len
            .saturating_sub(text_prompt_length)
            .saturating_sub(4);
        let patch_budget = post_shuffle_budget
            .saturating_mul(
                config
                    .reduction_factor
                    .saturating_mul(config.reduction_factor),
            )
            .max(config.min_num_patches)
            .min(config.max_num_patches);
        let scale = (patch_budget as f64 / source_patches as f64)
            .sqrt()
            .min(1.0);
        let mut target_h = ((closest_h as f64 * scale).floor() as usize).max(1);
        let mut target_w = ((closest_w as f64 * scale).floor() as usize).max(1);

        if patch_budget > config.min_num_patches
            && target_h.saturating_mul(target_w) < config.min_num_patches
        {
            let upscale =
                (config.min_num_patches as f64 / target_h.saturating_mul(target_w) as f64).sqrt();
            target_h = (target_h as f64 * upscale).ceil() as usize;
            target_w = (target_w as f64 * upscale).ceil() as usize;
        }

        round_grid_dimension(
            &mut target_h,
            target_w,
            config.reduction_factor,
            patch_budget,
        );
        round_grid_dimension(
            &mut target_w,
            target_h,
            config.reduction_factor,
            patch_budget,
        );
        (target_w, target_h)
    }

    fn preprocess_one(
        &self,
        image: &DynamicImage,
        config: &ProcessorConfig,
        text_prompt_length: usize,
    ) -> PreprocessedEncoderInputs {
        let (patch_w, patch_h) =
            Self::target_patch_grid(image.width(), image.height(), config, text_prompt_length);
        let target_w = patch_w * config.patch_size;
        let target_h = patch_h * config.patch_size;
        let resized = transforms::resize(
            image,
            target_w as u32,
            target_h as u32,
            FilterType::CatmullRom,
        );
        let encoder_input =
            transforms::to_tensor_and_normalize(&resized, &config.mean, &config.std)
                .insert_axis(Axis(0));
        let feature_count = patch_w * patch_h / (config.reduction_factor * config.reduction_factor);

        PreprocessedEncoderInputs::new(
            encoder_input,
            vec![feature_count],
            vec![(image.width(), image.height())],
        )
        .with_extra(
            "imgs_sizes",
            ModelSpecificValue::TupleVec(vec![(target_h as u32, target_w as u32)]),
        )
        .with_extra(
            "num_tokens_per_image",
            ModelSpecificValue::IntVec(vec![feature_count as i64]),
        )
    }
}

impl VisionPreProcessor for NemotronHOmniProcessor {
    fn default_mean(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_with_context(images, config, &VisionPreprocessingContext::default())
    }

    fn preprocess_with_context(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
        context: &VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let [image] = images else {
            return Err(if images.is_empty() {
                TransformError::EmptyBatch
            } else {
                TransformError::ShapeError(
                    "Nemotron-H Omni currently supports one image per request".to_string(),
                )
            });
        };
        let config = Self::processor_config(config, context)?;
        Ok(self.preprocess_one(image, &config, context.text_prompt_length))
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let context = VisionPreprocessingContext::default();
        Self::processor_config(config, &context)
            .map(|config| {
                let (patch_w, patch_h) = Self::target_patch_grid(width, height, &config, 0);
                patch_w * patch_h / (config.reduction_factor * config.reduction_factor)
            })
            .unwrap_or(0)
    }

    fn model_name(&self) -> &'static str {
        "nemotron_h_omni"
    }
}

fn missing_config_field(field: &str) -> TransformError {
    TransformError::ShapeError(format!(
        "Nemotron-H Omni requires `{field}` from checkpoint or runtime configuration"
    ))
}

fn json_value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn json_usize(value: &Value, path: &[&str]) -> Option<usize> {
    json_value_at(value, path)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
}

fn json_f64(value: &Value, path: &[&str]) -> Option<f64> {
    json_value_at(value, path)?.as_f64()
}

fn json_array3(value: &Value, path: &[&str]) -> Option<[f64; 3]> {
    let values = json_value_at(value, path)?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    Some([
        values[0].as_f64()?,
        values[1].as_f64()?,
        values[2].as_f64()?,
    ])
}

fn config_usize(config: &PreProcessorConfig, key: &str) -> Option<usize> {
    config
        .extra
        .get(key)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
}

fn slice_to_array3(values: Option<&[f64]>) -> Option<[f64; 3]> {
    let [a, b, c] = values? else {
        return None;
    };
    Some([*a, *b, *c])
}

fn round_grid_dimension(dimension: &mut usize, other: usize, divisor: usize, patch_budget: usize) {
    let remainder = *dimension % divisor;
    if remainder == 0 {
        return;
    }
    let increment = divisor - remainder;
    if dimension.saturating_add(increment).saturating_mul(other) <= patch_budget {
        *dimension += increment;
    } else {
        *dimension = dimension.saturating_sub(remainder).max(divisor);
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use serde_json::json;

    use super::*;
    use crate::encoder_inputs::ModelSpecificValue;

    fn context(max_model_len: usize, text_prompt_length: usize) -> VisionPreprocessingContext {
        VisionPreprocessingContext {
            model_config: json!({
                "model_type": "nemotron_h_omni",
                "patch_size": 16,
                "downsample_ratio": 0.5,
                "norm_mean": [0.48145466, 0.4578275, 0.40821073],
                "norm_std": [0.26862954, 0.26130258, 0.27577711],
                "vision_config": {
                    "args": {
                        "min_num_patches": 1024,
                        "max_num_patches": 13312
                    }
                }
            }),
            max_model_len: Some(max_model_len),
            text_prompt_length,
        }
    }

    #[test]
    fn matches_python_patch_grid_rounding() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(48, 32, Rgb([255, 0, 0])));
        let output = NemotronHOmniProcessor::new()
            .preprocess_with_context(&[image], &PreProcessorConfig::default(), &context(16384, 0))
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 3, 384, 736]);
        assert_eq!(output.feature_token_counts, vec![276]);
        assert!(matches!(
            output.model_specific.get("imgs_sizes"),
            Some(ModelSpecificValue::TupleVec(values)) if values == &vec![(384, 736)]
        ));
        assert!(matches!(
            output.model_specific.get("num_tokens_per_image"),
            Some(ModelSpecificValue::IntVec(values)) if values == &vec![276]
        ));
    }

    #[test]
    fn prompt_length_limits_dynamic_resolution_budget() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4096, 4096, Rgb([1, 2, 3])));
        let output = NemotronHOmniProcessor::new()
            .preprocess_with_context(&[image], &PreProcessorConfig::default(), &context(512, 100))
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 3, 640, 640]);
        assert_eq!(output.feature_token_counts, vec![400]);
    }

    #[test]
    fn rejects_a_ragged_multi_image_batch() {
        let images = [
            DynamicImage::ImageRgb8(RgbImage::new(32, 32)),
            DynamicImage::ImageRgb8(RgbImage::new(48, 32)),
        ];
        let error = NemotronHOmniProcessor::new()
            .preprocess_with_context(&images, &PreProcessorConfig::default(), &context(16384, 0))
            .unwrap_err();

        assert!(error.to_string().contains("one image"));
    }

    #[test]
    fn requires_checkpoint_configuration() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
        let error = NemotronHOmniProcessor::new()
            .preprocess_with_context(
                &[image],
                &PreProcessorConfig::default(),
                &VisionPreprocessingContext::default(),
            )
            .unwrap_err();

        assert!(error.to_string().contains("patch_size"));
    }

    #[test]
    fn requires_runtime_max_model_len() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
        let mut context = context(16384, 0);
        context.max_model_len = None;
        let error = NemotronHOmniProcessor::new()
            .preprocess_with_context(&[image], &PreProcessorConfig::default(), &context)
            .unwrap_err();

        assert!(error.to_string().contains("max_model_len"));
    }
}
