use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, InklingAudioProcessor},
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{processor::PreprocessedEncoderInputs, PreProcessorConfig},
};

const IMAGE_MARKER_TOKEN: &str = "<|content_image|>";
const IMAGE_MARKER_ID: TokenId = 200005;
const IMAGE_TOKEN_ID: TokenId = 200054;
const AUDIO_MARKER_TOKEN: &str = "<|content_audio_input|>";
const AUDIO_TOKEN_ID: TokenId = 200053;

pub(super) struct InklingSpec;

impl InklingSpec {
    fn audio_enabled(metadata: &ModelMetadata) -> bool {
        metadata
            .config
            .get("audio_config")
            .and_then(|config| config.get("decoder_dmodel"))
            .is_some_and(|value| !value.is_null())
    }

    fn unsupported(&self, modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: self.name(),
            modality,
        }
    }
}

impl ModelProcessorSpec for InklingSpec {
    fn name(&self) -> &'static str {
        "inkling"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|mt| mt == "inkling_mm_model")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Image)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Image)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(IMAGE_MARKER_TOKEN.to_string()),
            Modality::Audio => Ok(AUDIO_MARKER_TOKEN.to_string()),
            Modality::Video | Modality::ImageEmbeds => Err(self.unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => Ok(IMAGE_TOKEN_ID),
            Modality::Audio => Ok(AUDIO_TOKEN_ID),
            Modality::Video | Modality::ImageEmbeds => Err(self.unsupported(modality)),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::from([(Modality::Image, usize::MAX)]);
        if Self::audio_enabled(metadata) {
            limits.insert(Modality::Audio, usize::MAX);
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        Some(Box::new(InklingAudioProcessor::from_configs(
            model_config,
            preprocessor_config,
        )))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let (marker_token, marker_id, embed_token_id) = match modality {
            Modality::Image => (IMAGE_MARKER_TOKEN, IMAGE_MARKER_ID, IMAGE_TOKEN_ID),
            Modality::Audio => (
                AUDIO_MARKER_TOKEN,
                metadata.token_id(AUDIO_MARKER_TOKEN)?,
                AUDIO_TOKEN_ID,
            ),
            Modality::Video | Modality::ImageEmbeds => return Err(self.unsupported(modality)),
        };

        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                let mut tokens = vec![embed_token_id; num_tokens + 1];
                tokens[0] = marker_id;
                PromptReplacement::sequence(modality, marker_token, tokens)
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

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image => EncoderFieldLayouts::from_legacy_fields(self.field_layouts()),
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::flat("num_audio_tokens"),
                HashMap::from([("num_audio_tokens".to_string(), FieldLayout::Batched)]),
            ),
            Modality::Video | Modality::ImageEmbeds => EncoderFieldLayouts::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::{json, Value};

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelProcessorSpec, ModelRegistry},
        types::{FieldLayout, ImageSize, Modality},
    };

    fn inkling_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (super::IMAGE_MARKER_TOKEN, 200005),
            (super::AUDIO_MARKER_TOKEN, 200020),
        ])
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "test-model",
            tokenizer,
            config,
        }
    }

    #[test]
    fn inkling_matches_model_type() {
        let tokenizer = inkling_tokenizer();
        let config = json!({"model_type": "inkling_mm_model"});
        let metadata = metadata(&tokenizer, &config);

        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");

        assert_eq!(spec.name(), "inkling");
    }

    #[test]
    fn inkling_does_not_match_model_id_without_model_type() {
        let tokenizer = inkling_tokenizer();
        let config = json!({"model_type": "unknown"});
        let metadata = ModelMetadata {
            model_id: "example/model-share",
            ..metadata(&tokenizer, &config)
        };

        let registry = ModelRegistry::new();
        assert!(registry.lookup(&metadata).is_none());
    }

    #[test]
    fn inkling_prompt_replacement_keeps_marker_and_flags_placeholders() {
        let tokenizer = inkling_tokenizer();
        let config = json!({"model_type": "inkling_mm_model"});
        let metadata = metadata(&tokenizer, &config);
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("inkling spec");

        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(80, 40)], &[3]),
            )
            .unwrap();

        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens, vec![200005, 200054, 200054, 200054]);
    }

    #[test]
    fn inkling_audio_capability_follows_decoder_config() {
        let tokenizer = inkling_tokenizer();
        let enabled = json!({
            "model_type": "inkling_mm_model",
            "audio_config": {"decoder_dmodel": 1024}
        });
        let disabled = json!({
            "model_type": "inkling_mm_model",
            "audio_config": {"decoder_dmodel": null}
        });
        let registry = ModelRegistry::new();

        let enabled_metadata = metadata(&tokenizer, &enabled);
        let enabled_spec = registry.lookup(&enabled_metadata).unwrap();
        assert_eq!(
            enabled_spec.modality_limits(&enabled_metadata).unwrap(),
            HashMap::from([(Modality::Image, usize::MAX), (Modality::Audio, usize::MAX),])
        );

        let disabled_metadata = metadata(&tokenizer, &disabled);
        let disabled_spec = registry.lookup(&disabled_metadata).unwrap();
        assert_eq!(
            disabled_spec.modality_limits(&disabled_metadata).unwrap(),
            HashMap::from([(Modality::Image, usize::MAX)])
        );
    }

    #[test]
    fn inkling_audio_replacement_keeps_marker_and_appends_frames() {
        let tokenizer = inkling_tokenizer();
        let config = json!({
            "model_type": "inkling_mm_model",
            "audio_config": {"decoder_dmodel": 1024}
        });
        let metadata = metadata(&tokenizer, &config);
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();

        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Audio)
                .unwrap(),
            super::AUDIO_MARKER_TOKEN
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Audio)
                .unwrap(),
            200053
        );
        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(80, 3)], &[3]),
                Modality::Audio,
            )
            .unwrap();
        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens, vec![200020, 200053, 200053, 200053]);
    }

    #[test]
    fn inkling_audio_replacement_requires_marker_in_tokenizer() {
        let tokenizer = TestTokenizer::new(&[(super::IMAGE_MARKER_TOKEN, 200005)]);
        let config = json!({
            "model_type": "inkling_mm_model",
            "audio_config": {"decoder_dmodel": 1024}
        });
        let metadata = metadata(&tokenizer, &config);
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();
        let error = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(80, 1)], &[1]),
                Modality::Audio,
            )
            .unwrap_err();

        assert_eq!(
            error,
            crate::registry::ModelRegistryError::TokenNotFound {
                token: super::AUDIO_MARKER_TOKEN.to_string(),
            }
        );
    }

    #[test]
    fn inkling_audio_layout_is_flat_by_token_count() {
        let layouts = super::InklingSpec.encoder_field_layouts_for(Modality::Audio);
        assert_eq!(layouts.encoder_input, FieldLayout::flat("num_audio_tokens"));
        assert_eq!(
            layouts.model_specific.get("num_audio_tokens"),
            Some(&FieldLayout::Batched)
        );
    }
}
