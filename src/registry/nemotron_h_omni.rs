//! Nemotron-H Omni image model contract.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{NemotronHOmniProcessor, PreProcessorConfig, VisionPreProcessor},
};

const IMAGE_PLACEHOLDER: &str = "<image>";
const IMAGE_START: &str = "<img>";
const IMAGE_END: &str = "</img>";

pub(super) struct NemotronHOmniVisionSpec;

impl ModelProcessorSpec for NemotronHOmniVisionSpec {
    fn vision_processor(
        &self,
        metadata: &ModelMetadata,
        config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(NemotronHOmniProcessor::from_configs(
                metadata.config,
                config,
            )?)),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn name(&self) -> &'static str {
        "nemotron_h_omni"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == "nemotron_h_omni")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["img_context_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "img_context_token_id".to_string(),
            })
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 1)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token = self.placeholder_token_id(metadata)?;
        let image_start = metadata.token_id(IMAGE_START)?;
        let image_end = metadata.token_id(IMAGE_END)?;

        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(image_start);
                tokens.extend(std::iter::repeat_n(image_token, count));
                tokens.push(image_end);
                PromptReplacement::sequence(Modality::Image, IMAGE_PLACEHOLDER, tokens)
            })
            .collect())
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([
                    ("imgs_sizes".to_string(), FieldLayout::Batched),
                    ("num_tokens_per_image".to_string(), FieldLayout::Batched),
                ]),
            ),
            _ => EncoderFieldLayouts::default(),
        }
    }

    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        (modality == Modality::Image).then(|| "pixel_values_flat".to_string())
    }

    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        match modality {
            Modality::Image => vec!["imgs_sizes".to_string(), "num_tokens_per_image".to_string()],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;
    use ndarray::Array4;
    use serde_json::json;

    use super::*;
    use crate::registry::test_helpers::TestTokenizer;
    use crate::PreprocessingContext;

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026",
            tokenizer,
            config,
        }
    }

    #[test]
    fn registry_resolves_nemotron_h_omni() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({
            "model_type": "nemotron_h_omni",
            "img_context_token_id": 18,
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "norm_mean": [0.0, 0.0, 0.0],
            "norm_std": [1.0, 1.0, 1.0],
            "vision_config": {
                "args": {"min_num_patches": 1024, "max_num_patches": 13312}
            }
        });
        let metadata = metadata(&tokenizer, &config);

        assert_eq!(
            crate::ModelRegistry::new()
                .lookup(&metadata)
                .unwrap()
                .name(),
            "nemotron_h_omni"
        );
        let processor = NemotronHOmniVisionSpec
            .vision_processor(&metadata, &PreProcessorConfig::default(), Modality::Image)
            .unwrap();
        let output = processor
            .preprocess_with_context(
                &[DynamicImage::new_rgb8(48, 32)],
                &PreprocessingContext {
                    token_budget: Some(16384),
                },
            )
            .unwrap();
        assert_eq!(output.feature_token_counts, vec![276]);
    }

    #[test]
    fn factory_distinguishes_unsupported_modality_and_invalid_config() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "nemotron_h_omni"});
        let metadata = metadata(&tokenizer, &config);
        assert!(matches!(
            NemotronHOmniVisionSpec.vision_processor(
                &metadata,
                &PreProcessorConfig::default(),
                Modality::Video,
            ),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "nemotron_h_omni",
                modality: Modality::Video,
            })
        ));
        assert!(matches!(
            NemotronHOmniVisionSpec.vision_processor(
                &metadata, &PreProcessorConfig::default(), Modality::Image,
            ),
            Err(ModelRegistryError::MissingConfigField { field }) if field == "patch_size"
        ));
    }

    #[test]
    fn declares_dynamic_image_fields() {
        let layouts = NemotronHOmniVisionSpec.encoder_field_layouts_for(Modality::Image);

        assert_eq!(layouts.encoder_input, FieldLayout::Batched);
        assert_eq!(
            layouts.model_specific,
            HashMap::from([
                ("imgs_sizes".to_string(), FieldLayout::Batched),
                ("num_tokens_per_image".to_string(), FieldLayout::Batched),
            ])
        );
        assert_eq!(
            NemotronHOmniVisionSpec.keep_on_cpu_keys_for(Modality::Image),
            vec!["imgs_sizes", "num_tokens_per_image"]
        );
    }

    #[test]
    fn prompt_replacement_wraps_image_embeddings() {
        let tokenizer =
            TestTokenizer::new(&[(IMAGE_PLACEHOLDER, 18), (IMAGE_START, 19), (IMAGE_END, 20)]);
        let config = json!({
            "model_type": "nemotron_h_omni",
            "img_context_token_id": 18
        });
        let inputs = PreprocessedEncoderInputs::new(
            Array4::<f32>::zeros((1, 3, 384, 736)),
            vec![276],
            vec![(48, 32)],
        );

        let replacements = NemotronHOmniVisionSpec
            .prompt_replacements(&metadata(&tokenizer, &config), &inputs)
            .unwrap();

        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens.len(), 278);
        assert_eq!(replacements[0].tokens[0], 19);
        assert!(replacements[0].tokens[1..277].iter().all(|id| *id == 18));
        assert_eq!(replacements[0].tokens[277], 20);
    }
}
