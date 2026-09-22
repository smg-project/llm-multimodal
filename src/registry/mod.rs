mod deepseek_v41;
mod inkling;
mod kimi_k25;
mod kimi_k3;
mod llama4;
mod llava;
mod minimax_m3;
mod phi3_v;
mod qwen3_asr;
mod qwen3_omni;
mod qwen3_vl;
mod qwen_vl;
mod traits;

use deepseek_v41::DeepseekV41VisionSpec;
use inkling::InklingSpec;
use kimi_k25::KimiK25VisionSpec;
use kimi_k3::KimiK3VisionSpec;
use llama4::Llama4Spec;
use llava::{LlavaNextSpec, LlavaSpec};
use minimax_m3::MiniMaxM3VisionSpec;
use once_cell::sync::Lazy;
use phi3_v::Phi3VisionSpec;
use qwen3_asr::Qwen3AsrSpec;
use qwen3_omni::Qwen3OmniSpec;
use qwen3_vl::Qwen3VLVisionSpec;
use qwen_vl::QwenVLVisionSpec;
// Re-export public API from traits.
pub use deepseek_v41::DEEPSEEK_V41_IMAGE_PLACEHOLDER;
pub use traits::{
    ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult, Tokenizer,
};

pub struct ModelRegistry {
    specs: Vec<LazySpec>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            specs: vec![
                LazySpec::new(|| Box::new(DeepseekV41VisionSpec)),
                LazySpec::new(|| Box::new(KimiK3VisionSpec)),
                LazySpec::new(|| Box::new(KimiK25VisionSpec)),
                LazySpec::new(|| Box::new(Llama4Spec)),
                // LlavaNext must be registered before Llava so "llava_next" model_type matches first.
                LazySpec::new(|| Box::new(LlavaNextSpec)),
                LazySpec::new(|| Box::new(LlavaSpec)),
                LazySpec::new(|| Box::new(MiniMaxM3VisionSpec)),
                LazySpec::new(|| Box::new(Qwen3AsrSpec)),
                LazySpec::new(|| Box::new(Qwen3OmniSpec)),
                // Qwen3-VL must be registered before QwenVL so "qwen3" matches first.
                LazySpec::new(|| Box::new(Qwen3VLVisionSpec)),
                LazySpec::new(|| Box::new(QwenVLVisionSpec)),
                LazySpec::new(|| Box::new(Phi3VisionSpec)),
                LazySpec::new(|| Box::new(InklingSpec)),
            ],
        }
    }

    pub fn lookup<'a>(&'a self, metadata: &ModelMetadata) -> Option<&'a dyn ModelProcessorSpec> {
        for spec in &self.specs {
            let spec_ref = spec.get();
            if spec_ref.matches(metadata) {
                return Some(spec_ref);
            }
        }
        None
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

struct LazySpec {
    inner: Lazy<Box<dyn ModelProcessorSpec>>,
}

impl LazySpec {
    fn new(factory: fn() -> Box<dyn ModelProcessorSpec>) -> Self {
        Self {
            inner: Lazy::new(factory),
        }
    }

    fn get(&self) -> &dyn ModelProcessorSpec {
        self.inner.as_ref()
    }
}

#[cfg(test)]
pub(super) mod test_helpers {
    use std::collections::HashMap;

    use crate::{
        encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
        registry::Tokenizer,
        types::ImageSize,
    };

    pub struct TestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TestTokenizer {
        pub fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs
                .iter()
                .map(|(token, id)| ((*token).to_string(), *id))
                .collect();
            Self { vocab }
        }
    }

    impl Tokenizer for TestTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab
                .iter()
                .find(|(_, &v)| v == id)
                .map(|(k, _)| k.clone())
        }

        fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
            Some(Vec::new())
        }
    }

    pub fn test_preprocessed_with_tokens(
        item_sizes: &[ImageSize],
        feature_token_counts: &[usize],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            feature_token_counts: feature_token_counts.to_vec(),
            item_sizes: sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Build `PreprocessedEncoderInputs` with explicit aspect_ratios (for Llama4 tests).
    pub fn test_preprocessed_with_aspects(
        item_sizes: &[ImageSize],
        aspect_ratios: &[(i64, i64)],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        let flat: Vec<i64> = aspect_ratios
            .iter()
            .flat_map(|&(h, w)| vec![h, w])
            .collect();
        let batch = aspect_ratios.len();
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::IntTensor {
                data: flat,
                shape: vec![batch, 2],
            },
        );
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]),
            feature_token_counts: vec![0; sizes.len()],
            item_sizes: sizes,
            model_specific,
        }
    }
}

#[cfg(test)]
mod processor_tests {
    use image::{DynamicImage, Rgb, RgbImage};
    use serde_json::json;

    use super::{test_helpers::TestTokenizer, ModelMetadata, ModelRegistry};
    use crate::{Modality, ModelSpecificValue, PreProcessorConfig};

    fn image() -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(128, 128, Rgb([255; 3])))
    }

    #[test]
    fn vision_instances_retain_independent_checkpoint_parameters() {
        let registry = ModelRegistry::new();
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({"model_type": "qwen3_vl"});
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &model_config,
        };
        let spec = registry.lookup(&metadata).unwrap();
        let mut config = PreProcessorConfig::from_json(
            r#"{"min_pixels":1024,"max_pixels":1024,"image_mean":[0,0,0],"image_std":[1,1,1]}"#,
        )
        .unwrap();
        let small = spec
            .vision_processor(&metadata, &config, Modality::Image)
            .unwrap();
        config.min_pixels = Some(4096);
        config.max_pixels = Some(4096);
        config.image_mean = Some(vec![1.0; 3]);
        let large = spec
            .vision_processor(&metadata, &config, Modality::Image)
            .unwrap();
        drop(config);

        // Interleaved requests keep each instance's geometry and normalization.
        for _ in 0..2 {
            let small_output = small.preprocess(&[image()]).unwrap();
            let large_output = large.preprocess(&[image()]).unwrap();
            assert_eq!(small_output.feature_token_counts, vec![1]);
            assert_eq!(large_output.feature_token_counts, vec![4]);
            assert!(small_output.encoder_input.iter().all(|&v| v == 1.0));
            assert!(large_output.encoder_input.iter().all(|&v| v == 0.0));
            assert_eq!(small.calculate_num_tokens(128, 128), 1);
            assert_eq!(large.calculate_num_tokens(128, 128), 4);
        }
    }

    #[test]
    fn vision_factory_resolves_image_and_video_configs_independently() {
        let registry = ModelRegistry::new();
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({"model_type": "qwen3_vl"});
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &model_config,
        };
        let spec = registry.lookup(&metadata).unwrap();
        let image_config =
            PreProcessorConfig::from_json(r#"{"min_pixels":1024,"max_pixels":1024}"#).unwrap();
        let video_config =
            PreProcessorConfig::from_json(r#"{"min_pixels":8192,"max_pixels":8192,"fps":8.0}"#)
                .unwrap();
        let images = spec
            .vision_processor(&metadata, &image_config, Modality::Image)
            .unwrap();
        let videos = spec
            .vision_processor(&metadata, &video_config, Modality::Video)
            .unwrap();
        assert_eq!(
            images.preprocess(&[image()]).unwrap().feature_token_counts,
            vec![1]
        );
        let video = videos.preprocess_video(&[image(), image()]).unwrap();
        assert_eq!(video.feature_token_counts, vec![4]);
        assert!(matches!(
            video.model_specific.get("video_second_per_grid"),
            Some(ModelSpecificValue::Tensor { data, shape })
                if data == &[0.25] && shape == &[1]
        ));
        assert!(spec
            .vision_processor(&metadata, &image_config, Modality::Audio)
            .is_none());
    }

    #[test]
    fn vision_factory_uses_model_metadata_for_llava_geometry() {
        let registry = ModelRegistry::new();
        let tokenizer = TestTokenizer::new(&[]);
        let model_config = json!({
            "model_type": "llava",
            "vision_config": {"image_size": 28, "patch_size": 14},
            "image_aspect_ratio": "pad",
        });
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &model_config,
        };
        let spec = registry.lookup(&metadata).unwrap();
        let config = PreProcessorConfig::default();
        let processor = spec
            .vision_processor(&metadata, &config, Modality::Image)
            .unwrap();
        let output = processor.preprocess(&[image()]).unwrap();
        assert_eq!(output.encoder_input.shape(), &[1, 3, 28, 28]);
        assert_eq!(output.feature_token_counts, vec![4]);
        assert_eq!(processor.get_processed_size(), Some((28, 28)));
        assert!(spec
            .vision_processor(&metadata, &config, Modality::Video)
            .is_none());
    }
}
