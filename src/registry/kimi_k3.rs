use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

pub(super) struct KimiK3VisionSpec;

impl KimiK3VisionSpec {
    const PROMPT_MARKER: &str = "<|media_pad|>";

    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["media_placeholder_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "media_placeholder_token_id".to_string(),
            })
    }

    fn encode_text(metadata: &ModelMetadata, text: &str) -> Vec<TokenId> {
        metadata
            .tokenizer
            .encode_text(text)
            .unwrap_or_default()
            .into_iter()
            .map(|token| token as TokenId)
            .collect()
    }
}

impl ModelProcessorSpec for KimiK3VisionSpec {
    fn metadata_only_codec(&self, modality: Modality) -> Option<super::MetadataOnlyCodec> {
        (modality == Modality::Image).then_some(super::MetadataOnlyCodec {
            fields: &[
                super::MetadataField::BatchedTensor("grid_thws"),
                super::MetadataField::ImageSize,
            ],
            parse: super::metadata_only::kimi_k3_image,
        })
    }

    fn name(&self) -> &'static str {
        "kimi_k3"
    }
    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == "kimi_k3")
            || metadata.model_id.to_ascii_lowercase().contains("kimi-k3")
    }
    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::PROMPT_MARKER.to_string())
    }
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::pad_token_id(metadata)
    }
    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }
    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }
    fn prompt_replacements_from_metadata(
        &self,
        metadata: &ModelMetadata,
        preprocessed: super::EncoderMetadata<'_>,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality != Modality::Image {
            return Err(super::ModelRegistryError::UnsupportedMetadataOnly {
                spec: self.name(),
                modality,
            });
        }
        let pad = Self::pad_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .item_sizes
            .iter()
            .zip(preprocessed.feature_token_counts)
            .map(|(&(width, height), &count)| {
                let mut tokens = Self::encode_text(
                    metadata,
                    &format!("<|media_begin|>image {width}x{height}<|media_content|>"),
                );
                tokens.extend(std::iter::repeat_n(pad, count));
                tokens.extend(Self::encode_text(metadata, "<|media_end|>"));
                PromptReplacement::sequence(Modality::Image, &placeholder, tokens)
            })
            .collect())
    }
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("grid_thws".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["grid_thws".to_string()]
    }
}
