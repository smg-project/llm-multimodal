use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    registry::{ModelMetadata, ModelProcessorSpec, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::image_processor::PreprocessedImages,
};

pub(super) struct MiniMaxM3VisionSpec;

impl MiniMaxM3VisionSpec {
    const IMAGE_TOKEN: &'static str = "]<]image[>[";
    const VISION_START_TOKEN: &'static str = "]<]start of image[>[";
    const VISION_END_TOKEN: &'static str = "]<]end of image[>[";
}

impl ModelProcessorSpec for MiniMaxM3VisionSpec {
    fn name(&self) -> &'static str {
        "minimax_m3_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("minimax") && id.contains("m3")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "minimax_m3_vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(Self::IMAGE_TOKEN)
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
        preprocessed: &PreprocessedImages,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = metadata.token_id(Self::IMAGE_TOKEN)?;
        let start_token_id = metadata.token_id(Self::VISION_START_TOKEN)?;
        let end_token_id = metadata.token_id(Self::VISION_END_TOKEN)?;

        Ok(preprocessed
            .num_img_tokens
            .iter()
            .map(|&n| {
                // Mirrors MiniMaxM3VLMultiModalProcessor._get_prompt_updates:
                // full = [start_token_id] + [image_token_id] * N + [end_token_id]
                // The image_token_id positions are marked as embed tokens.
                let mut tokens = Vec::with_capacity(n + 2);
                tokens.push(start_token_id);
                tokens.extend(std::iter::repeat_n(image_token_id, n));
                tokens.push(end_token_id);
                PromptReplacement::sequence(Modality::Image, Self::IMAGE_TOKEN, tokens)
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
    fn minimax_m3_matches_model_type() {
        let tokenizer = TestTokenizer::new(&[
            ("]<]image[>[", 200025),
            ("]<]start of image[>[", 200029),
            ("]<]end of image[>[", 200030),
        ]);
        let config = json!({ "model_type": "minimax_m3_vl" });
        let metadata = ModelMetadata {
            model_id: "some-custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("minimax_m3_vl spec");
        assert_eq!(spec.name(), "minimax_m3_vl");
    }

    #[test]
    fn minimax_m3_matches_model_id() {
        let tokenizer = TestTokenizer::new(&[
            ("]<]image[>[", 200025),
            ("]<]start of image[>[", 200029),
            ("]<]end of image[>[", 200030),
        ]);
        let config = json!({ "model_type": "minimax_m3_vl" });
        let metadata = ModelMetadata {
            model_id: "MiniMaxAI/Minimax-M3-preview",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("minimax_m3_vl spec");
        assert_eq!(spec.name(), "minimax_m3_vl");
    }

    #[test]
    fn minimax_m3_prompt_replacements_wrap_with_start_end() {
        let tokenizer = TestTokenizer::new(&[
            ("]<]image[>[", 200025),
            ("]<]start of image[>[", 200029),
            ("]<]end of image[>[", 200030),
        ]);
        let config = json!({ "model_type": "minimax_m3_vl" });
        let metadata = ModelMetadata {
            model_id: "MiniMaxAI/Minimax-M3-preview",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("minimax_m3_vl spec");

        // 672x672 image → 576 tokens (48x48 grid, merge_size=2 → 48*48/4=576)
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(672, 672)], &[576]),
            )
            .unwrap();

        assert_eq!(replacements.len(), 1);
        let tokens = &replacements[0].tokens;
        assert_eq!(tokens.len(), 578); // 1 start + 576 image + 1 end
        assert_eq!(tokens[0], 200029); // start
        assert!(tokens[1..577].iter().all(|&t| t == 200025)); // image tokens
        assert_eq!(tokens[577], 200030); // end
    }
}
