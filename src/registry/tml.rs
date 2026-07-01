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

pub(super) struct TmlVisionSpec;

impl ModelProcessorSpec for TmlVisionSpec {
    fn name(&self) -> &'static str {
        "tml"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("tml")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "tml" || mt == "tml_mm_model")
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
                let mut tokens = Vec::with_capacity(num_tokens + 1);
                tokens.push(IMAGE_MARKER_ID);
                tokens.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(num_tokens));
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
    fn tml_matches_model_type() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({"model_type": "tml"});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };

        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("tml spec");

        assert_eq!(spec.name(), "tml");
    }

    #[test]
    fn tml_matches_mm_model_type() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({"model_type": "tml_mm_model"});
        let metadata = ModelMetadata {
            model_id: "thinkingmachineslabinc/tml-model-share",
            tokenizer: &tokenizer,
            config: &config,
        };

        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("tml spec");

        assert_eq!(spec.name(), "tml");
    }

    #[test]
    fn tml_prompt_replacement_keeps_marker_and_flags_placeholders() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({"model_type": "tml"});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("tml spec");

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
