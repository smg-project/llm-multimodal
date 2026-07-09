use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{ModelMetadata, ModelProcessorSpec, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::processor::PreprocessedEncoderInputs,
};

const IMAGE_MARKER_TOKEN: &str = "<|content_image|>";
const IMAGE_MARKER_ID: TokenId = 200005;
const IMAGE_TOKEN_ID: TokenId = 200007;

pub(super) struct InklingVisionSpec;

impl ModelProcessorSpec for InklingVisionSpec {
    fn name(&self) -> &'static str {
        "inkling"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|mt| mt == "inkling_mm_model")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_MARKER_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, _metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Ok(IMAGE_TOKEN_ID)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, usize::MAX)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        _metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                let mut tokens = vec![IMAGE_TOKEN_ID; num_tokens + 1];
                tokens[0] = IMAGE_MARKER_ID;
                PromptReplacement::sequence(Modality::Image, IMAGE_MARKER_TOKEN, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::flat("num_patches")),
            (
                "vision_patches_bthwc".to_string(),
                FieldLayout::flat("num_patches"),
            ),
            ("num_patches".to_string(), FieldLayout::Batched),
        ])
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
    fn inkling_matches_model_type() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({"model_type": "inkling_mm_model"});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };

        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");

        assert_eq!(spec.name(), "inkling");
    }

    #[test]
    fn inkling_does_not_match_model_id_without_model_type() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({"model_type": "unknown"});
        let metadata = ModelMetadata {
            model_id: "example/model-share",
            tokenizer: &tokenizer,
            config: &config,
        };

        let registry = ModelRegistry::new();
        assert!(registry.lookup(&metadata).is_none());
    }

    #[test]
    fn inkling_prompt_replacement_keeps_marker_and_flags_placeholders() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({"model_type": "inkling_mm_model"});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");

        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(80, 40)], &[3]),
            )
            .unwrap();

        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens, vec![200005, 200007, 200007, 200007]);
    }
}
