use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{image_processor::PreprocessedImages, video_processor::PreprocessedVideos},
};

pub(super) struct QwenVLVisionSpec;

impl QwenVLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vision_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vision_token_id".to_string(),
            })
    }

    /// Resolve `<|video_pad|>` token ID from the tokenizer.
    ///
    /// Qwen2-VL uses a dedicated `<|video_pad|>` token for video pad tokens,
    /// which is NOT exposed in `config.json` (only `video_token_id` for the
    /// `<<video>>` placeholder is). Fall back to the image pad token if the
    /// video pad token is absent.
    fn video_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .token_id("<|video_pad|>")
            .or_else(|_| Self::pad_token_id(metadata))
    }
}

impl ModelProcessorSpec for QwenVLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("qwen") && id.contains("vl")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "qwen2_vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image_pad|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        // Must match pad_token_id (vision_token_id) — this is the repeated token
        // in the expanded sequence. image_token_id is a distinct token in Qwen2-VL.
        Self::pad_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10), (Modality::Video, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // The chat template already wraps each image with <|vision_start|> ... <|vision_end|>,
        // so we only expand the single <image> placeholder to N pad tokens.
        Ok(preprocessed
            .num_img_tokens
            .iter()
            .map(|&num_tokens| {
                let tokens = vec![pad_token_id; num_tokens];
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Image fields: pixel_values is patchified [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        // Video fields: same structure with video_grid_thw / patches_per_video.
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            (
                "pixel_values_videos".to_string(),
                FieldLayout::flat("patches_per_video"),
            ),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }

    fn video_placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|video_pad|>".to_string())
    }

    fn video_placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::video_pad_token_id(metadata)
    }

    fn video_prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedVideos,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::video_pad_token_id(metadata)?;
        let placeholder_token = self.video_placeholder_token(metadata)?;
        Ok(preprocessed
            .num_video_tokens
            .iter()
            .map(|&num_tokens| {
                let tokens = vec![pad_token_id; num_tokens];
                PromptReplacement::sequence(Modality::Video, &placeholder_token, tokens)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::{ImageSize, Modality},
        vision::video_processor::PreprocessedVideos,
    };

    #[test]
    fn qwen_vision_uses_config_token_ids() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");
        // 448/14 = 32 grid, merge_size=2 => (32*32)/4 = 256 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[256]),
            )
            .unwrap();
        // Only pad tokens — vision_start/vision_end are already in the chat template
        assert_eq!(replacements[0].tokens.len(), 256);
        assert_eq!(replacements[0].tokens[0], 151654); // pad (vision_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151654);
    }

    #[test]
    fn qwen_vl_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match qwen alias");
        assert_eq!(spec.name(), "qwen_vl");
    }

    #[test]
    fn qwen_vl_video_prompt_replacements() {
        let tokenizer = TestTokenizer::new(&[
            ("<|image_pad|>", 151654),
            ("<|video_pad|>", 151657),
        ]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");

        // 2 videos: 128 and 256 tokens respectively
        let preprocessed = PreprocessedVideos::new(
            ndarray::Array2::zeros((384, 3 * 2 * 14 * 14)),
            vec![128, 256],
            vec![(448, 448, 8), (448, 448, 16)],
        );

        let replacements = spec
            .video_prompt_replacements(&metadata, &preprocessed)
            .unwrap();

        assert_eq!(replacements.len(), 2);

        // First video
        assert_eq!(replacements[0].modality, Modality::Video);
        assert_eq!(replacements[0].placeholder_token, "<|video_pad|>");
        assert_eq!(replacements[0].tokens.len(), 128);
        assert_eq!(replacements[0].tokens[0], 151657);
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151657);

        // Second video
        assert_eq!(replacements[1].tokens.len(), 256);
        assert_eq!(replacements[1].tokens[0], 151657);
    }

    #[test]
    fn qwen_vl_video_falls_back_to_image_pad() {
        // When <|video_pad|> is not in the tokenizer, fall back to vision_token_id.
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151654)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");

        let preprocessed = PreprocessedVideos::new(
            ndarray::Array2::zeros((64, 3 * 2 * 14 * 14)),
            vec![64],
            vec![(448, 448, 4)],
        );

        let replacements = spec
            .video_prompt_replacements(&metadata, &preprocessed)
            .unwrap();

        assert_eq!(replacements[0].tokens[0], 151654); // falls back to image pad
    }

    #[test]
    fn qwen_vl_modality_limits_includes_video() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151654)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_token_id": 151654,
            "image_token_id": 151655
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");
        let limits = spec.modality_limits(&metadata).unwrap();
        assert_eq!(limits.get(&Modality::Image), Some(&10));
        assert_eq!(limits.get(&Modality::Video), Some(&10));
    }
}
