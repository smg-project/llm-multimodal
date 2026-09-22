//! Nemotron-H Omni image preprocessing.

use image::{imageops::FilterType, DynamicImage};
use ndarray::Axis;
use serde_json::Value;

use crate::encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs};
use crate::registry::{ModelRegistryError, RegistryResult};
use crate::vision::{
    transforms, PreProcessorConfig, TransformError, VisionPreProcessor, VisionPreprocessingContext,
};

#[derive(Debug, Clone)]
pub struct NemotronHOmniProcessor {
    patch_size: usize,
    reduction_factor: usize,
    min_num_patches: usize,
    max_num_patches: usize,
    mean: [f64; 3],
    std: [f64; 3],
}

impl NemotronHOmniProcessor {
    /// Resolve and validate checkpoint parameters once for this model instance.
    pub fn from_configs(model_config: &Value, config: &PreProcessorConfig) -> RegistryResult<Self> {
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
            return Err(ModelRegistryError::InvalidConfig {
                message:
                    "Nemotron-H Omni requires a positive patch size and a 0.5 downsample ratio"
                        .to_string(),
            });
        }

        let min_num_patches =
            json_usize(model_config, &["vision_config", "args", "min_num_patches"])
                .or_else(|| config_usize(config, "min_num_patches"))
                .ok_or_else(|| missing_config_field("vision_config.args.min_num_patches"))?;
        let max_num_patches =
            json_usize(model_config, &["vision_config", "args", "max_num_patches"])
                .or_else(|| config_usize(config, "max_num_patches"))
                .ok_or_else(|| missing_config_field("vision_config.args.max_num_patches"))?;
        let mean = json_array3(model_config, &["norm_mean"])
            .or_else(|| slice_to_array3(config.image_mean.as_deref()))
            .ok_or_else(|| missing_config_field("norm_mean"))?;
        let std = json_array3(model_config, &["norm_std"])
            .or_else(|| slice_to_array3(config.image_std.as_deref()))
            .ok_or_else(|| missing_config_field("norm_std"))?;
        if min_num_patches == 0 || max_num_patches < min_num_patches {
            return Err(ModelRegistryError::InvalidConfig {
                message: "Nemotron-H Omni patch limits must be positive and ordered".to_string(),
            });
        }
        if std.contains(&0.0) {
            return Err(ModelRegistryError::InvalidConfig {
                message: "Nemotron-H Omni normalization standard deviations must be nonzero"
                    .to_string(),
            });
        }

        Ok(Self {
            patch_size,
            reduction_factor,
            min_num_patches,
            max_num_patches,
            mean,
            std,
        })
    }

    fn target_patch_grid(&self, width: u32, height: u32, token_budget: usize) -> (usize, usize) {
        let closest_h = (f64::from(height) / self.patch_size as f64 + 0.5)
            .round_ties_even()
            .max(1.0) as usize;
        let closest_w = (f64::from(width) / self.patch_size as f64 + 0.5)
            .round_ties_even()
            .max(1.0) as usize;
        let source_patches = closest_h.saturating_mul(closest_w).max(1);
        let post_shuffle_budget = token_budget.saturating_sub(4);
        let patch_budget = post_shuffle_budget
            .saturating_mul(self.reduction_factor.saturating_mul(self.reduction_factor))
            .max(self.min_num_patches)
            .min(self.max_num_patches);
        let scale = (patch_budget as f64 / source_patches as f64)
            .sqrt()
            .min(1.0);
        let mut target_h = ((closest_h as f64 * scale).floor() as usize).max(1);
        let mut target_w = ((closest_w as f64 * scale).floor() as usize).max(1);

        if patch_budget > self.min_num_patches
            && target_h.saturating_mul(target_w) < self.min_num_patches
        {
            let upscale =
                (self.min_num_patches as f64 / target_h.saturating_mul(target_w) as f64).sqrt();
            target_h = (target_h as f64 * upscale).ceil() as usize;
            target_w = (target_w as f64 * upscale).ceil() as usize;
        }

        round_grid_dimension(&mut target_h, target_w, self.reduction_factor, patch_budget);
        round_grid_dimension(&mut target_w, target_h, self.reduction_factor, patch_budget);
        (target_w, target_h)
    }

    fn preprocess_one(
        &self,
        image: &DynamicImage,
        token_budget: usize,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let (patch_w, patch_h) =
            self.target_patch_grid(image.width(), image.height(), token_budget);
        let feature_count = patch_w * patch_h / (self.reduction_factor * self.reduction_factor);
        if token_budget
            .checked_sub(4)
            .is_none_or(|budget| feature_count > budget)
        {
            return Err(TransformError::ShapeError(format!(
                "Nemotron-H Omni image requires {feature_count} feature tokens plus 4 structural tokens, exceeding token_budget {token_budget}"
            )));
        }
        let target_w = patch_w * self.patch_size;
        let target_h = patch_h * self.patch_size;
        let resized = transforms::resize(
            image,
            target_w as u32,
            target_h as u32,
            FilterType::CatmullRom,
        );
        let encoder_input = transforms::to_tensor_and_normalize(&resized, &self.mean, &self.std)
            .insert_axis(Axis(0));

        Ok(PreprocessedEncoderInputs::new(
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
        ))
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
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_with_context(images, &VisionPreprocessingContext::default())
    }

    fn preprocess_with_context(
        &self,
        images: &[DynamicImage],
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
        let token_budget = context.token_budget.ok_or_else(|| {
            TransformError::ShapeError(
                "Nemotron-H Omni requires a request token_budget".to_string(),
            )
        })?;
        self.preprocess_one(image, token_budget)
    }

    /// Estimate using the checkpoint patch limit. Preprocessing reports the actual
    /// feature count after applying the request's token budget.
    fn calculate_num_tokens(&self, width: u32, height: u32) -> usize {
        let (patch_w, patch_h) = self.target_patch_grid(width, height, usize::MAX);
        patch_w * patch_h / (self.reduction_factor * self.reduction_factor)
    }

    fn model_name(&self) -> &'static str {
        "nemotron_h_omni"
    }
}

fn missing_config_field(field: &str) -> ModelRegistryError {
    ModelRegistryError::MissingConfigField {
        field: field.to_string(),
    }
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

    fn model_config() -> Value {
        json!({
            "model_type": "nemotron_h_omni",
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_config": {
                "args": { "min_num_patches": 1024, "max_num_patches": 13312 }
            }
        })
    }

    fn processor() -> NemotronHOmniProcessor {
        NemotronHOmniProcessor::from_configs(&model_config(), &PreProcessorConfig::default())
            .unwrap()
    }

    fn context(token_budget: usize) -> VisionPreprocessingContext {
        VisionPreprocessingContext {
            token_budget: Some(token_budget),
        }
    }

    #[test]
    fn matches_python_patch_grid_rounding() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(48, 32, Rgb([255, 0, 0])));
        let output = processor()
            .preprocess_with_context(&[image], &context(16384))
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
    fn token_budget_limits_dynamic_resolution() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4096, 4096, Rgb([1, 2, 3])));
        let output = processor()
            .preprocess_with_context(&[image], &context(512 - 100))
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 3, 640, 640]);
        assert_eq!(output.feature_token_counts, vec![400]);
    }

    #[test]
    fn request_budgets_do_not_change_processor_configuration() {
        let processor = processor();
        let images = [DynamicImage::new_rgb8(1024, 1024)];
        let unrestricted = processor.calculate_num_tokens(1024, 1024);
        for budget in [16384, 412, 16384] {
            let output = processor
                .preprocess_with_context(&images, &context(budget))
                .unwrap();
            let count = output.feature_token_counts[0];
            assert_eq!(count, if budget == 412 { 400 } else { unrestricted });
            assert!(count + 4 <= budget);
        }
    }

    #[test]
    fn rejects_a_ragged_multi_image_batch() {
        let images = [
            DynamicImage::new_rgb8(32, 32),
            DynamicImage::new_rgb8(48, 32),
        ];
        let error = processor()
            .preprocess_with_context(&images, &context(16384))
            .unwrap_err();
        assert!(error.to_string().contains("one image"));
    }

    #[test]
    fn requires_checkpoint_configuration_at_construction() {
        assert!(matches!(
            NemotronHOmniProcessor::from_configs(&json!({}), &PreProcessorConfig::default()),
            Err(ModelRegistryError::MissingConfigField { field }) if field == "patch_size"
        ));
    }

    #[test]
    fn validates_checkpoint_parameters_at_construction() {
        for (field, value) in [
            ("patch_size", json!(0)),
            ("downsample_ratio", json!(0.25)),
            ("norm_std", json!([1.0, 0.0, 1.0])),
            (
                "vision_config",
                json!({"args": {"min_num_patches": 0, "max_num_patches": 1024}}),
            ),
            (
                "vision_config",
                json!({"args": {"min_num_patches": 1024, "max_num_patches": 512}}),
            ),
        ] {
            let mut config = model_config();
            config[field] = value;
            assert!(
                matches!(
                    NemotronHOmniProcessor::from_configs(&config, &PreProcessorConfig::default()),
                    Err(ModelRegistryError::InvalidConfig { .. })
                ),
                "{field}"
            );
        }
    }

    #[test]
    fn preprocessor_fallbacks_are_owned_by_the_instance() {
        let mut config = PreProcessorConfig::from_value(json!({
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "min_num_patches": 1024,
            "max_num_patches": 13312,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0]
        }))
        .unwrap();
        let first = NemotronHOmniProcessor::from_configs(&json!({}), &config).unwrap();
        config.image_mean = Some(vec![1.0; 3]);
        let second = NemotronHOmniProcessor::from_configs(&json!({}), &config).unwrap();
        drop(config);

        let images = [DynamicImage::ImageRgb8(RgbImage::from_pixel(
            48,
            32,
            Rgb([255; 3]),
        ))];
        let first_output = first
            .preprocess_with_context(&images, &context(16384))
            .unwrap();
        let second_output = second
            .preprocess_with_context(&images, &context(16384))
            .unwrap();
        assert_eq!(first_output.encoder_input.shape(), &[1, 3, 384, 736]);
        assert!(first_output.encoder_input.iter().all(|&v| v == 1.0));
        assert!(second_output.encoder_input.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn requires_request_token_budget() {
        let image = DynamicImage::new_rgb8(32, 32);
        let error = processor().preprocess(&[image]).unwrap_err();
        assert!(error.to_string().contains("token_budget"));
    }

    #[test]
    fn rejects_insufficient_budget_including_structural_tokens() {
        let processor = processor();
        let images = [DynamicImage::new_rgb8(512, 512)];
        for budget in [0, 3, 4, 259] {
            let error = processor
                .preprocess_with_context(&images, &context(budget))
                .unwrap_err();
            assert!(error.to_string().contains("exceeding token_budget"));
        }
        let output = processor
            .preprocess_with_context(&images, &context(260))
            .unwrap();
        assert_eq!(output.feature_token_counts, vec![256]);
    }
}
