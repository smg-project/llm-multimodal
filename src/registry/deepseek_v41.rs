//! DeepSeek-V4.1 vision model contract.
//!
//! Mirrors vLLM's `models/deepseek_v4_1/common/mm_preprocess.py`: each
//! `<｜deepseek_image｜>` placeholder expands to a span where every position
//! carries `image_token_id` (the roles ride in the per-image `types` tensor),
//! plus a compressor-alignment pad prepended at splice time (see
//! [`crate::AlignmentPad`]). The pad positions borrow the reserved in-vocab
//! token `<|place_holder_mm_span_0436|>` so they stay distinguishable from
//! real span positions; v4.1 uses ratio-2 compressors (v4.0 used 4).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::processors::deepseek_v41::COMPRESS_PAD_TO,
};

/// The placeholder text inlined at each image's position; the multimodal
/// pipeline later expands it into the image span. Public so frontends can
/// render it without re-resolving the spec (mirrors the Python encoding's
/// hard-coded `IMAGE_PLACEHOLDER`).
pub const DEEPSEEK_V41_IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub(super) struct DeepseekV41VisionSpec;

impl DeepseekV41VisionSpec {
    /// Reserved in-vocab token borrowed by compressor-alignment pads.
    const IMAGE_PAD_TOKEN_NAME: &'static str = "<|place_holder_mm_span_0436|>";

    /// TODO: vLLM leaves the image count unlimited for this model
    /// (`get_supported_mm_limits` returns `{"image": None}`); keep a generous
    /// cap until the frontend can express "unlimited".
    const MAX_IMAGES_PER_PROMPT: usize = 128;

    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn image_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(Self::IMAGE_PAD_TOKEN_NAME)
    }
}

impl ModelProcessorSpec for DeepseekV41VisionSpec {
    fn name(&self) -> &'static str {
        "deepseek_v41"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == "deepseek_v41")
            || metadata
                .model_id
                .to_ascii_lowercase()
                .contains("deepseek-v4.1")
            || metadata
                .model_id
                .to_ascii_lowercase()
                .contains("deepseek_v41")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(DEEPSEEK_V41_IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(
            Modality::Image,
            Self::MAX_IMAGES_PER_PROMPT,
        )]))
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
        let image_pad_id = Self::image_pad_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                // Every span position carries image_token_id; the roles live
                // in `types`. All of them are embed positions (delimiters get
                // the learned vectors from embed_multimodal, not the embed
                // table). The alignment pad is prepended at splice time.
                PromptReplacement::repeated(Modality::Image, &placeholder, image_token_id, count)
                    .with_alignment_pad(image_pad_id, COMPRESS_PAD_TO)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("vit_grid".to_string(), FieldLayout::Batched),
            ("llm_grid".to_string(), FieldLayout::Batched),
            ("types".to_string(), FieldLayout::flat("types_per_image")),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            ("types_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    /// The model's forward pops `patches` (not the HF-conventional
    /// `pixel_values`); see `DeepseekV4VLImagePixelInputs`.
    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        match modality {
            Modality::Image => Some("patches".to_string()),
            _ => None,
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "vit_grid".to_string(),
            "llm_grid".to_string(),
            "types".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::registry::test_helpers::TestTokenizer;

    const IMAGE_TOKEN_ID: u32 = 129264;
    const IMAGE_PAD_TOKEN_ID: u32 = 129265;

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "/models/local-checkpoint",
            tokenizer,
            config,
        }
    }
    fn spec_inputs(span_lens: &[usize]) -> PreprocessedEncoderInputs {
        PreprocessedEncoderInputs::new(
            Array2::<f32>::zeros((4, 588)),
            span_lens.to_vec(),
            vec![(100, 100); span_lens.len()],
        )
    }

    #[test]
    fn matches_deepseek_v41_model_type() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        assert!(DeepseekV41VisionSpec.matches(&metadata(&tokenizer, &config)));
        assert!(crate::VisionProcessorRegistry::with_defaults()
            .find("/models/local-checkpoint", Some("deepseek_v41"))
            .is_some());

        // A text-only DeepSeek model with a neutral model id must not match.
        let other = json!({"model_type": "deepseek_v3"});
        let other_metadata = ModelMetadata {
            model_id: "deepseek-ai/DeepSeek-V3",
            tokenizer: &tokenizer,
            config: &other,
        };
        assert!(!DeepseekV41VisionSpec.matches(&other_metadata));
    }

    #[test]
    fn prompt_replacements_expand_spans_with_alignment_pad() {
        let tokenizer = TestTokenizer::new(&[(
            DeepseekV41VisionSpec::IMAGE_PAD_TOKEN_NAME,
            IMAGE_PAD_TOKEN_ID,
        )]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        let metadata = metadata(&tokenizer, &config);

        let replacements = DeepseekV41VisionSpec
            .prompt_replacements(&metadata, &spec_inputs(&[5, 8]))
            .unwrap();

        assert_eq!(replacements.len(), 2);
        for (replacement, &count) in replacements.iter().zip([5usize, 8].iter()) {
            assert_eq!(replacement.modality, Modality::Image);
            assert_eq!(replacement.tokens, vec![IMAGE_TOKEN_ID as TokenId; count]);
            let pad = replacement.alignment_pad.expect("alignment pad");
            assert_eq!(pad.token_id, IMAGE_PAD_TOKEN_ID as TokenId);
            assert_eq!(pad.period, COMPRESS_PAD_TO);
        }
    }

    #[test]
    fn prompt_replacements_require_pad_token_in_vocab() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        let metadata = metadata(&tokenizer, &config);

        let result = DeepseekV41VisionSpec.prompt_replacements(&metadata, &spec_inputs(&[5]));
        assert!(matches!(
            result,
            Err(ModelRegistryError::TokenNotFound { .. })
        ));
    }
}
