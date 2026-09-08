//! Validated metadata-only inputs, independent of HTTP and transfer backends.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::{ModelMetadata, ModelRegistryError, RegistryResult};
use crate::{ModelSpecificValue, PreProcessorConfig, PromptReplacement};

/// Borrowed prompt inputs shared by preprocessed media and metadata references.
#[derive(Clone, Copy)]
pub struct EncoderMetadata<'a> {
    pub feature_token_counts: &'a [usize],
    pub item_sizes: &'a [(u32, u32)],
    pub model_specific: &'a HashMap<String, ModelSpecificValue>,
}

/// Batched auxiliary inputs without pixels or embedding tensors.
#[derive(Debug, Default)]
pub struct PreparedEncoderMetadata {
    pub feature_token_counts: Vec<usize>,
    pub item_sizes: Vec<(u32, u32)>,
    pub model_specific: HashMap<String, ModelSpecificValue>,
}

impl PreparedEncoderMetadata {
    pub fn as_metadata(&self) -> EncoderMetadata<'_> {
        EncoderMetadata {
            feature_token_counts: &self.feature_token_counts,
            item_sizes: &self.item_sizes,
            model_specific: &self.model_specific,
        }
    }
}

#[derive(Debug)]
pub struct PreparedMetadataOnly {
    pub metadata: PreparedEncoderMetadata,
    pub replacements: Vec<PromptReplacement>,
}

pub type MetadataOnlyParser = fn(
    &ModelMetadata<'_>,
    &PreProcessorConfig,
    &[Value],
) -> RegistryResult<PreparedEncoderMetadata>;

/// Where a published field comes from in the actual preprocessor output.
#[derive(Clone, Copy)]
pub enum MetadataField {
    BatchedTensor(&'static str),
    /// Published as `num_image_tokens: [n]`.
    FeatureTokenCount,
    /// Published as `image_sizes: [width, height]` (Kimi-K3 prompt metadata).
    ImageSize,
}

/// One model's publication and consumption contract.
#[derive(Clone, Copy)]
pub struct MetadataOnlyCodec {
    pub fields: &'static [MetadataField],
    pub parse: MetadataOnlyParser,
}

impl MetadataOnlyCodec {
    pub(super) fn export(&self, metadata: EncoderMetadata<'_>) -> RegistryResult<Vec<Value>> {
        let batch = metadata.feature_token_counts.len();
        let mut payloads = Vec::with_capacity(batch);
        for index in 0..batch {
            let mut payload = serde_json::Map::new();
            for field in self.fields {
                let (name, value) = match field {
                    MetadataField::BatchedTensor(name) => {
                        let tensor = metadata
                            .model_specific
                            .get(*name)
                            .ok_or_else(|| invalid(format!("missing metadata field {name}")))?;
                        let value = match tensor {
                            ModelSpecificValue::IntTensor { data, shape } => {
                                export_row(data, shape, batch, index)?
                            }
                            ModelSpecificValue::UintTensor { data, shape } => {
                                export_row(data, shape, batch, index)?
                            }
                            _ => {
                                return Err(invalid(format!(
                                    "metadata field {name} must be an integer tensor"
                                )))
                            }
                        };
                        (*name, value)
                    }
                    MetadataField::FeatureTokenCount => {
                        let count = metadata.feature_token_counts[index];
                        if count == 0 {
                            return Err(invalid("feature token count must be positive"));
                        }
                        ("num_image_tokens", serde_json::json!([count]))
                    }
                    MetadataField::ImageSize => {
                        if metadata.item_sizes.len() != batch {
                            return Err(invalid("original image sizes do not match media items"));
                        }
                        let (width, height) = metadata.item_sizes[index];
                        positive_size([width, height])?;
                        ("image_sizes", serde_json::json!([width, height]))
                    }
                };
                payload.insert(name.to_owned(), value);
            }
            payloads.push(Value::Object(payload));
        }
        Ok(payloads)
    }
}

fn export_row<T: serde::Serialize>(
    data: &[T],
    shape: &[usize],
    batch: usize,
    index: usize,
) -> RegistryResult<Value> {
    if shape.len() != 2
        || shape[0] != batch
        || shape[1] == 0
        || shape[0].checked_mul(shape[1]) != Some(data.len())
    {
        return Err(invalid(
            "metadata tensor must have shape [num_items, num_values]",
        ));
    }
    let start = index * shape[1];
    serde_json::to_value(&data[start..start + shape[1]])
        .map_err(|error| invalid(format!("invalid metadata row: {error}")))
}

pub(super) fn invalid(message: impl Into<String>) -> ModelRegistryError {
    ModelRegistryError::InvalidMetadata {
        message: message.into(),
    }
}

fn parse<T: serde::de::DeserializeOwned>(payload: &Value) -> RegistryResult<T> {
    serde_json::from_value(payload.clone())
        .map_err(|error| invalid(format!("invalid metadata: {error}")))
}

fn grid_tokens([t, h, w]: [u32; 3], merge: usize) -> RegistryResult<usize> {
    let (h, w) = (h as usize, w as usize);
    if t != 1 || merge == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
        return Err(invalid(
            "image grid must be positive, t=1 and divisible by merge size",
        ));
    }
    (h / merge)
        .checked_mul(w / merge)
        .ok_or_else(|| invalid("image token count overflow"))
}

fn positive_size([a, b]: [u32; 2]) -> RegistryResult<(u32, u32)> {
    if a == 0 || b == 0 {
        return Err(invalid("image dimensions must be positive"));
    }
    Ok((a, b))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageGrid {
    image_grid_thw: [u32; 3],
}

pub(super) fn qwen_image(
    model: &ModelMetadata<'_>,
    processor: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    let merge = processor.merge_size.unwrap_or(
        model
            .config
            .pointer("/vision_config/spatial_merge_size")
            .and_then(Value::as_u64)
            .unwrap_or(2) as usize,
    );
    parse_image_grid(payloads, merge)
}

pub(super) fn image_grid(
    _: &ModelMetadata<'_>,
    processor: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    parse_image_grid(payloads, processor.merge_size.unwrap_or(2))
}

fn parse_image_grid(payloads: &[Value], merge: usize) -> RegistryResult<PreparedEncoderMetadata> {
    let mut result = PreparedEncoderMetadata::default();
    let mut grids = Vec::new();
    for payload in payloads {
        let item: ImageGrid = parse(payload)?;
        result
            .feature_token_counts
            .push(grid_tokens(item.image_grid_thw, merge)?);
        grids.extend(item.image_grid_thw.map(i64::from));
    }
    result.model_specific.insert(
        "image_grid_thw".into(),
        ModelSpecificValue::int_2d(grids, payloads.len(), 3),
    );
    Ok(result)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageTokenCount {
    num_image_tokens: [usize; 1],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageTokenCountWithSizes {
    num_image_tokens: [usize; 1],
    image_sizes: [u32; 2],
}

pub(super) fn image_token_count(
    _: &ModelMetadata<'_>,
    _: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    let mut result = PreparedEncoderMetadata::default();
    for payload in payloads {
        let item: ImageTokenCount = parse(payload)?;
        result.feature_token_counts.push(item.num_image_tokens[0]);
    }
    Ok(result)
}

pub(super) fn image_token_count_with_sizes(
    _: &ModelMetadata<'_>,
    _: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    let mut result = PreparedEncoderMetadata::default();
    let mut sizes = Vec::new();
    for payload in payloads {
        let item: ImageTokenCountWithSizes = parse(payload)?;
        let (height, width) = positive_size(item.image_sizes)?;
        result.item_sizes.push((width, height));
        result.feature_token_counts.push(item.num_image_tokens[0]);
        sizes.extend(item.image_sizes.map(i64::from));
    }
    result.model_specific.insert(
        "image_sizes".into(),
        ModelSpecificValue::int_2d(sizes, payloads.len(), 2),
    );
    Ok(result)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Llama4Image {
    aspect_ratios: [u32; 2],
}

pub(super) fn llama4_image(
    model: &ModelMetadata<'_>,
    _: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    let per_tile = super::llama4::Llama4Spec::tokens_per_tile(model);
    if per_tile == 0 {
        return Err(invalid("invalid Llama4 tokens per tile"));
    }
    let mut result = PreparedEncoderMetadata::default();
    let mut ratios = Vec::new();
    for payload in payloads {
        let item: Llama4Image = parse(payload)?;
        let (h, w) = positive_size(item.aspect_ratios)?;
        let grid_tiles = (h as usize)
            .checked_mul(w as usize)
            .ok_or_else(|| invalid("tile count overflow"))?;
        let tiles = if grid_tiles == 1 {
            1
        } else {
            grid_tiles
                .checked_add(1)
                .ok_or_else(|| invalid("tile count overflow"))?
        };
        result.feature_token_counts.push(
            tiles
                .checked_mul(per_tile)
                .ok_or_else(|| invalid("image token count overflow"))?,
        );
        ratios.extend(item.aspect_ratios.map(i64::from));
    }
    result.model_specific.insert(
        "aspect_ratios".into(),
        ModelSpecificValue::int_2d(ratios, payloads.len(), 2),
    );
    Ok(result)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KimiImage {
    grid_thws: [u32; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KimiK3Image {
    grid_thws: [u32; 3],
    /// Original width, height, used verbatim by K3's prompt format.
    image_sizes: [u32; 2],
}

pub(super) fn kimi_image(
    _: &ModelMetadata<'_>,
    processor: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    let mut result = PreparedEncoderMetadata::default();
    let mut grids = Vec::new();
    for payload in payloads {
        let item: KimiImage = parse(payload)?;
        result.feature_token_counts.push(grid_tokens(
            item.grid_thws,
            processor.merge_size.unwrap_or(2),
        )?);
        grids.extend(item.grid_thws.map(i64::from));
    }
    result.model_specific.insert(
        "grid_thws".into(),
        ModelSpecificValue::int_2d(grids, payloads.len(), 3),
    );
    Ok(result)
}

pub(super) fn kimi_k3_image(
    _: &ModelMetadata<'_>,
    processor: &PreProcessorConfig,
    payloads: &[Value],
) -> RegistryResult<PreparedEncoderMetadata> {
    let fixed = processor.extra.get("fixed_output_tokens").or_else(|| {
        processor
            .extra
            .get("media_proc_cfg")
            .and_then(|v| v.get("fixed_output_tokens"))
    });
    let fixed = fixed
        .map(|v| {
            serde_json::from_value::<usize>(v.clone())
                .map_err(|_| invalid("invalid fixed_output_tokens"))
        })
        .transpose()?;
    let mut result = PreparedEncoderMetadata::default();
    let mut grids = Vec::new();
    for payload in payloads {
        let item: KimiK3Image = parse(payload)?;
        let count = grid_tokens(item.grid_thws, processor.merge_size.unwrap_or(2))?;
        result.feature_token_counts.push(fixed.unwrap_or(count));
        result.item_sizes.push(positive_size(item.image_sizes)?);
        grids.extend(item.grid_thws.map(i64::from));
    }
    result.model_specific.insert(
        "grid_thws".into(),
        ModelSpecificValue::int_2d(grids, payloads.len(), 3),
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;
    use serde_json::json;

    use super::*;
    use crate::registry::test_helpers::TestTokenizer;
    use crate::vision::processors::{
        InklingImageProcessor, KimiK25Processor, KimiK3Processor, Llama4VisionProcessor,
        LlavaNextProcessor, LlavaProcessor, MiniMaxM3Processor, Phi3VisionProcessor,
        Qwen2VLProcessor, Qwen3OmniVisionProcessor, Qwen3VLProcessor,
    };
    use crate::{Modality, ModelRegistry, Tokenizer, VisionPreProcessor};

    struct TokenResolver(TestTokenizer);
    impl Tokenizer for TokenResolver {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.0.token_to_id(token)
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0.id_to_token(id)
        }
        fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
            Some(text.bytes().map(u32::from).collect())
        }
    }

    fn tokenizer() -> TokenResolver {
        TokenResolver(TestTokenizer::new(&[
            ("<|image_pad|>", 151655),
            ("<image>", 32000),
            ("<|media_pad|>", 163605),
            ("<|image_start|>", 200088),
            ("<|image_end|>", 200089),
            ("<|image|>", 200090),
            ("<|patch|>", 200092),
            ("<|tile_x_separator|>", 200093),
            ("<|tile_y_separator|>", 200094),
            ("]<]image[>[", 200025),
            ("]<]start of image[>[", 200029),
            ("]<]end of image[>[", 200030),
            ("<|content_image|>", 200005),
        ]))
    }

    fn config(model: &str) -> Value {
        json!({"model_type": model, "image_token_id": 151655,
            "image_token_index": if model == "llama4" {200092} else {32000},
            "media_placeholder_token_id": 163605,
            "vision_config": {"image_size": 336, "patch_size": 14, "spatial_merge_size": 2}})
    }

    fn rows(value: &ModelSpecificValue, width: usize) -> Vec<Value> {
        let data = match value {
            ModelSpecificValue::IntTensor { data, .. } => data.clone(),
            ModelSpecificValue::UintTensor { data, .. } => {
                data.iter().copied().map(i64::from).collect()
            }
            _ => panic!("expected integer tensor"),
        };
        data.chunks_exact(width).map(|row| json!(row)).collect()
    }

    #[test]
    fn published_metadata_round_trips_real_preprocessing_for_supported_images() {
        let cases: Vec<(&str, Box<dyn VisionPreProcessor>)> = vec![
            ("qwen2_vl", Box::new(Qwen2VLProcessor::new())),
            ("qwen2_5_vl", Box::new(Qwen2VLProcessor::new())),
            ("qwen3_vl", Box::new(Qwen3VLProcessor::new())),
            ("qwen3_5", Box::new(Qwen3VLProcessor::new())),
            ("qwen3_5_moe", Box::new(Qwen3VLProcessor::new())),
            ("qwen4_exp", Box::new(Qwen3VLProcessor::new())),
            ("llava", Box::new(LlavaProcessor::new())),
            ("llava_next", Box::new(LlavaNextProcessor::new())),
            ("llama4", Box::new(Llama4VisionProcessor::new())),
            ("kimi_k25", Box::new(KimiK25Processor::new())),
            ("kimi_k3", Box::new(KimiK3Processor::new())),
            ("minimax_m3_vl", Box::new(MiniMaxM3Processor::new())),
            ("phi3_v", Box::new(Phi3VisionProcessor::new())),
            ("qwen3_omni_moe", Box::new(Qwen3OmniVisionProcessor::new())),
            (
                "qwen3_omni_moe_thinker",
                Box::new(Qwen3OmniVisionProcessor::new()),
            ),
            ("inkling_mm_model", Box::new(InklingImageProcessor::new())),
        ];
        let images = [
            DynamicImage::new_rgb8(80, 40),
            DynamicImage::new_rgb8(40, 80),
        ];
        let tokenizer = tokenizer();
        let registry = ModelRegistry::new();
        for (name, processor) in cases {
            let config = config(name);
            let model = ModelMetadata {
                model_id: name,
                config: &config,
                tokenizer: &tokenizer,
            };
            let spec = registry.lookup(&model).unwrap();
            let processor_config = PreProcessorConfig::default();
            let raw = processor.preprocess(&images, &processor_config).unwrap();
            let payloads = spec
                .export_metadata(raw.as_metadata(), Modality::Image)
                .unwrap();
            let prepared = spec
                .prepare_metadata_only(&model, &processor_config, Modality::Image, &payloads, 32768)
                .unwrap();
            let expected = spec
                .prompt_replacements_for(&model, &raw, Modality::Image)
                .unwrap();
            assert_eq!(prepared.replacements.len(), images.len(), "{name}");
            let embed_id = spec.placeholder_token_id(&model).unwrap();
            for ((actual, expected), count) in prepared
                .replacements
                .iter()
                .zip(expected)
                .zip(&prepared.metadata.feature_token_counts)
            {
                assert_eq!(actual.tokens, expected.tokens, "{name}");
                assert_eq!(
                    actual.structural_prefix, expected.structural_prefix,
                    "{name}"
                );
                assert_eq!(
                    *count,
                    actual.tokens.iter().filter(|&&t| t == embed_id).count(),
                    "{name}"
                );
            }
            for (key, value) in &prepared.metadata.model_specific {
                let width = if key == "image_sizes" || key == "aspect_ratios" {
                    2
                } else {
                    3
                };
                assert_eq!(
                    rows(value, width),
                    rows(&raw.model_specific[key], width),
                    "{name}: {key}"
                );
            }
            assert!(!prepared
                .metadata
                .model_specific
                .contains_key("pixel_values"));
            if !prepared.metadata.item_sizes.is_empty() {
                assert_eq!(prepared.metadata.item_sizes, raw.item_sizes, "{name}");
            }
        }
    }

    #[test]
    fn image_grid_round_trip_honors_processor_merge_and_omni_nested_token() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        let processor_config = PreProcessorConfig {
            merge_size: Some(3),
            ..Default::default()
        };
        let cases: Vec<(&str, Box<dyn VisionPreProcessor>)> = vec![
            ("minimax_m3_vl", Box::new(MiniMaxM3Processor::new())),
            ("qwen3_omni_moe", Box::new(Qwen3OmniVisionProcessor::new())),
        ];
        for (name, processor) in cases {
            let mut config = config(name);
            // Omni must use the nested token, not this deliberately different outer ID.
            config["image_token_id"] = json!(999);
            config["thinker_config"] = json!({"image_token_id": 151655});
            let model = ModelMetadata {
                model_id: name,
                config: &config,
                tokenizer: &tokenizer,
            };
            let spec = registry.lookup(&model).unwrap();
            let raw = processor
                .preprocess(&[DynamicImage::new_rgb8(96, 48)], &processor_config)
                .unwrap();
            let payloads = spec
                .export_metadata(raw.as_metadata(), Modality::Image)
                .unwrap();
            let prepared = spec
                .prepare_metadata_only(&model, &processor_config, Modality::Image, &payloads, 32768)
                .unwrap();
            assert_eq!(
                prepared.metadata.feature_token_counts, raw.feature_token_counts,
                "{name}"
            );
            assert_eq!(
                prepared.replacements[0].tokens,
                spec.prompt_replacements(&model, &raw).unwrap()[0].tokens,
                "{name}"
            );
            if name == "qwen3_omni_moe" {
                assert!(prepared.replacements[0]
                    .tokens
                    .iter()
                    .all(|&id| id == 151655));
            }
        }
    }

    #[test]
    fn metadata_only_budget_includes_minimax_and_inkling_markers() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        for (name, payload, tokens) in [
            (
                "minimax_m3_vl",
                json!({"image_grid_thw": [1, 2, 2]}),
                vec![200029, 200025, 200030],
            ),
            (
                "inkling_mm_model",
                json!({"num_image_tokens": [1]}),
                vec![200005, 200054],
            ),
        ] {
            let config = config(name);
            let model = ModelMetadata {
                model_id: name,
                config: &config,
                tokenizer: &tokenizer,
            };
            let spec = registry.lookup(&model).unwrap();
            let processor = PreProcessorConfig::default();
            assert!(
                spec.prepare_metadata_only(
                    &model,
                    &processor,
                    Modality::Image,
                    std::slice::from_ref(&payload),
                    tokens.len() - 1,
                )
                .is_err(),
                "{name}"
            );
            let prepared = spec
                .prepare_metadata_only(
                    &model,
                    &processor,
                    Modality::Image,
                    &[payload],
                    tokens.len(),
                )
                .unwrap();
            assert_eq!(prepared.metadata.feature_token_counts, [1]);
            assert_eq!(prepared.replacements[0].tokens, tokens);
        }
    }

    #[test]
    fn additional_image_contracts_reject_invalid_metadata_and_other_modalities() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        for (name, payload) in [
            ("minimax_m3_vl", json!({"image_grid_thw": [2, 2, 2]})),
            ("qwen3_omni_moe", json!({"image_grid_thw": [1, 3, 2]})),
            (
                "phi3_v",
                json!({"num_image_tokens": [1], "image_sizes": [0, 336]}),
            ),
            ("phi3_v", json!({"num_image_tokens": [1]})),
            ("inkling_mm_model", json!({"num_image_tokens": [0]})),
            ("inkling_mm_model", json!({"num_image_tokens": [true]})),
            (
                "inkling_mm_model",
                json!({"num_image_tokens": [1], "pixel_values": [0]}),
            ),
        ] {
            let config = config(name);
            let model = ModelMetadata {
                model_id: name,
                config: &config,
                tokenizer: &tokenizer,
            };
            let spec = registry.lookup(&model).unwrap();
            let processor = PreProcessorConfig::default();
            assert!(
                matches!(
                    spec.prepare_metadata_only(
                        &model,
                        &processor,
                        Modality::Image,
                        &[payload],
                        128,
                    ),
                    Err(ModelRegistryError::InvalidMetadata { .. })
                ),
                "{name}"
            );
            for modality in [Modality::Audio, Modality::Video] {
                assert!(
                    matches!(
                        spec.prepare_metadata_only(&model, &processor, modality, &[], 128,),
                        Err(ModelRegistryError::UnsupportedMetadataOnly { .. })
                    ),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn metadata_only_rejects_malformed_fields_and_excessive_counts() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        let config = config("qwen3_vl");
        let model = ModelMetadata {
            model_id: "qwen3_vl",
            config: &config,
            tokenizer: &tokenizer,
        };
        let spec = registry.lookup(&model).unwrap();
        for payload in [
            json!({}),
            json!({"image_grid_thw": [true, 2, 2]}),
            json!({"image_grid_thw": [1, -2, 2]}),
            json!({"image_grid_thw": [1, 3, 2]}),
            json!({"image_grid_thw": [1, 2, 2], "image_embeds": [1]}),
            json!({"image_grid_thw": [1, 10000, 10000]}),
        ] {
            assert!(spec
                .prepare_metadata_only(
                    &model,
                    &PreProcessorConfig::default(),
                    Modality::Image,
                    &[payload],
                    16
                )
                .is_err());
        }
        let payload = json!({"image_grid_thw": [1, 8, 8]});
        assert!(spec
            .prepare_metadata_only(
                &model,
                &PreProcessorConfig::default(),
                Modality::Image,
                &[payload.clone(), payload.clone()],
                16
            )
            .is_err());
        assert!(spec
            .prepare_metadata_only(
                &model,
                &PreProcessorConfig::default(),
                Modality::Video,
                &[payload],
                16
            )
            .is_err());
    }

    #[test]
    fn kimi_k3_requires_original_dimensions_and_honors_fixed_output_tokens() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        let config = config("kimi_k3");
        let model = ModelMetadata {
            model_id: "kimi_k3",
            config: &config,
            tokenizer: &tokenizer,
        };
        let spec = registry.lookup(&model).unwrap();
        let processor =
            PreProcessorConfig::from_json(r#"{"media_proc_cfg":{"fixed_output_tokens":8}}"#)
                .unwrap();
        assert!(spec
            .prepare_metadata_only(
                &model,
                &processor,
                Modality::Image,
                &[json!({"grid_thws": [1, 2, 2]})],
                128
            )
            .is_err());
        let prepared = spec
            .prepare_metadata_only(
                &model,
                &processor,
                Modality::Image,
                &[json!({"grid_thws": [1, 2, 2], "image_sizes": [100, 50]})],
                128,
            )
            .unwrap();
        assert_eq!(prepared.metadata.feature_token_counts, [8]);
        let prefix: Vec<_> = b"<|media_begin|>image 100x50<|media_content|>"
            .iter()
            .map(|&b| i32::from(b))
            .collect();
        assert!(prepared.replacements[0].tokens.starts_with(&prefix));
    }

    #[test]
    fn publication_excludes_pixels_and_preserves_item_order() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        let config = config("kimi_k3");
        let model = ModelMetadata {
            model_id: "kimi_k3",
            config: &config,
            tokenizer: &tokenizer,
        };
        let spec = registry.lookup(&model).unwrap();
        let metadata = PreparedEncoderMetadata {
            feature_token_counts: vec![2, 3],
            item_sizes: vec![(56, 28), (28, 84)],
            model_specific: [
                (
                    "grid_thws".into(),
                    ModelSpecificValue::int_2d(vec![1, 2, 4, 1, 6, 2], 2, 3),
                ),
                (
                    "pixel_values".into(),
                    ModelSpecificValue::int_2d(vec![99, 99], 2, 1),
                ),
                (
                    "image_embeds".into(),
                    ModelSpecificValue::int_2d(vec![88, 88], 2, 1),
                ),
            ]
            .into(),
        };
        assert_eq!(
            spec.export_metadata(metadata.as_metadata(), Modality::Image)
                .unwrap(),
            [
                json!({"grid_thws": [1, 2, 4], "image_sizes": [56, 28]}),
                json!({"grid_thws": [1, 6, 2], "image_sizes": [28, 84]}),
            ]
        );
        assert!(spec
            .export_metadata(metadata.as_metadata(), Modality::Video)
            .is_err());
    }

    #[test]
    fn publication_rejects_missing_fields_and_inconsistent_batch_shapes() {
        let registry = ModelRegistry::new();
        let tokenizer = tokenizer();
        let config = config("kimi_k3");
        let model = ModelMetadata {
            model_id: "kimi_k3",
            config: &config,
            tokenizer: &tokenizer,
        };
        let spec = registry.lookup(&model).unwrap();
        let mut metadata = PreparedEncoderMetadata {
            feature_token_counts: vec![1],
            ..Default::default()
        };
        assert!(spec
            .export_metadata(metadata.as_metadata(), Modality::Image)
            .is_err());
        metadata.model_specific.insert(
            "grid_thws".into(),
            ModelSpecificValue::int_2d(vec![1, 2, 2], 1, 3),
        );
        // Original dimensions cannot be reconstructed from the processed grid.
        assert!(spec
            .export_metadata(metadata.as_metadata(), Modality::Image)
            .is_err());
        metadata.item_sizes = vec![(28, 28)];
        metadata.model_specific.insert(
            "grid_thws".into(),
            ModelSpecificValue::int_2d(vec![1, 2, 2], 2, 3),
        );
        assert!(spec
            .export_metadata(metadata.as_metadata(), Modality::Image)
            .is_err());
    }
}
