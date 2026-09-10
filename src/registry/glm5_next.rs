use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

pub(super) struct Glm5NextVisionSpec;

impl Glm5NextVisionSpec {
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|value| value as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for Glm5NextVisionSpec {
    fn name(&self) -> &'static str {
        "glm5_next"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == "glm5_next")
            || metadata.model_id.to_ascii_lowercase().contains("glm-5.3")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 64)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = Self::image_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                PromptReplacement::sequence(
                    Modality::Image,
                    &placeholder,
                    vec![image_token_id; num_tokens],
                )
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn glm5_next_matches_model_type_and_expands_image_tokens() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 154854)]);
        let config = json!({
            "model_type": "glm5_next",
            "image_token_id": 154854
        });
        let metadata = ModelMetadata {
            model_id: "custom-glm",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("glm5_next spec");
        assert_eq!(spec.name(), "glm5_next");
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[256]),
            )
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 256);
        assert!(replacements[0].tokens.iter().all(|&token| token == 154854));
    }
}
