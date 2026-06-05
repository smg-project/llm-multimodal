//! Video processor trait and output types.
//!
//! This module defines the interface for model-specific image processors
//! and the common output format for preprocessed images.

use std::{borrow::Cow, collections::HashMap};

use ndarray::{Array4, ArrayD};

use super::{preprocessor_config::PreProcessorConfig, transforms::TransformError};
use crate::types::FieldLayout;

/// Helper to extract a dimension from pixel_values given an ndim-dependent axis index.
/// Returns `Err` if the ndim is not 4 or 5.
fn dim_for_ndim(
    ndim: usize,
    axis_4d: usize,
    axis_5d: usize,
    shape: &[usize],
) -> Result<usize, TransformError> {
    match ndim {
        4 => Ok(shape[axis_4d]),
        5 => Ok(shape[axis_5d]),
        _ => Err(TransformError::InvalidShape {
            expected: format!("4D or 5D pixel_values tensor, got {ndim}D"),
            actual: shape.to_vec(),
        }),
    }
}

/// Model-specific output values that vary by architecture.
///
/// Different vision models require different auxiliary outputs beyond pixel_values.
/// This enum captures the common types of such outputs.
#[derive(Debug, Clone)]
pub enum ModelSpecificValue {
    /// A tensor with shape information (data as flat vec, shape as dims)
    Tensor { data: Vec<f32>, shape: Vec<usize> },

    /// A tensor of integers (e.g., aspect_ratio_ids)
    IntTensor { data: Vec<i64>, shape: Vec<usize> },

    /// A tensor of unsigned integers (e.g., image_grid_thw)
    UintTensor { data: Vec<u32>, shape: Vec<usize> },

    /// Simple integer value
    Int(i64),

    /// Simple float value
    Float(f64),

    /// List of integers
    IntVec(Vec<i64>),

    /// List of unsigned integers
    UintVec(Vec<u32>),

    /// List of floats
    FloatVec(Vec<f32>),

    /// List of tuples (e.g., image sizes)
    TupleVec(Vec<(u32, u32)>),

    /// Boolean flag
    Bool(bool),
}

impl ModelSpecificValue {
    /// Create a 1D uint tensor from a vector.
    pub fn uint_1d(data: Vec<u32>) -> Self {
        let len = data.len();
        Self::UintTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D uint tensor.
    pub fn uint_2d(data: Vec<u32>, rows: usize, cols: usize) -> Self {
        Self::UintTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Create a 1D int tensor from a vector.
    pub fn int_1d(data: Vec<i64>) -> Self {
        let len = data.len();
        Self::IntTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D int tensor.
    pub fn int_2d(data: Vec<i64>, rows: usize, cols: usize) -> Self {
        Self::IntTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Get the first dimension of a tensor variant, if applicable.
    pub fn first_dim(&self) -> Option<usize> {
        match self {
            Self::Tensor { shape, .. }
            | Self::IntTensor { shape, .. }
            | Self::UintTensor { shape, .. } => shape.first().copied(),
            _ => None,
        }
    }
}

/// Preprocessed images ready for model consumption.
///
/// This struct contains all the outputs needed by the SGLang scheduler
/// to construct `MultimodalInputs` for the model.
#[derive(Debug, Clone)]
pub struct PreprocessedImages {
    /// Pixel values as a dynamic-dimensional float32 tensor.
    ///
    /// This is the primary input to the vision encoder.
    /// Shape varies by model:
    /// - Standard: [B, C, H, W] (4D)
    /// - Phi3-Vision: [B, num_crops+1, C, H, W] (5D)
    pub pixel_values: ArrayD<f32>,

    /// Number of video tokens per video in the batch.
    ///
    /// Used to expand placeholder tokens in the text input.
    /// For example, LLaVA with 336x336 and patch_size=14 produces 576 tokens.
    pub num_video_tokens: Vec<usize>,

    /// Original video sizes as (width, height, frames) before preprocessing.
    ///
    /// Some models need this for proper attention masking or position encoding.
    pub video_sizes: Vec<(u32, u32, u32)>,

    /// Model-specific auxiliary outputs.
    ///
    /// Examples:
    /// - Qwen-VL: `video_grid_thw` for rotary position encoding
    pub model_specific: HashMap<String, ModelSpecificValue>,
}

impl PreprocessedVideos {
    /// Create a new PreprocessedVideos with required fields (4D pixel values).
    pub fn new(
        pixel_values: Array4<f32>,
        num_video_tokens: Vec<usize>,
        video_sizes: Vec<(u32, u32, u32)>,
    ) -> Self {
        Self {
            pixel_values: pixel_values.into_dyn(),
            num_video_tokens,
            video_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Create a new PreprocessedVideos with dynamic-dimensional pixel values.
    ///
    /// Use this for models like Phi3-Vision that have 5D tensors.
    pub fn new_dynamic(
        pixel_values: ArrayD<f32>,
        num_video_tokens: Vec<usize>,
        video_sizes: Vec<(u32, u32, u32)>,
    ) -> Self {
        Self {
            pixel_values,
            num_video_tokens,
            video_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Add a model-specific value.
    pub fn with_extra(mut self, key: impl Into<String>, value: ModelSpecificValue) -> Self {
        self.model_specific.insert(key.into(), value);
        self
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.pixel_values.shape()[0]
    }

    /// Get the number of dimensions of pixel_values.
    pub fn ndim(&self) -> usize {
        self.pixel_values.ndim()
    }

    /// Get total number of video tokens across all videos.
    pub fn total_tokens(&self) -> usize {
        self.num_video_tokens.iter().sum()
    }

    /// Get pixel values as a flat f32 slice without copying if possible.
    pub fn pixel_values_flat(&self) -> Cow<'_, [f32]> {
        match self.pixel_values.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(self.pixel_values.iter().copied().collect()),
        }
    }

    /// Get the shape of pixel values as a vector.
    pub fn pixel_values_shape(&self) -> Vec<usize> {
        self.pixel_values.shape().to_vec()
    }

    /// Number of videos in this batch.
    pub fn num_videos(&self) -> usize {
        self.video_sizes.len()
    }

    /// Extract batched tensor keys from explicit field layout declarations.
    pub fn batched_keys(layouts: &HashMap<String, FieldLayout>) -> Vec<String> {
        layouts
            .iter()
            .filter(|(_, l)| matches!(l, FieldLayout::Batched))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Extract flat-slicing tensor keys from explicit field layout declarations.
    ///
    /// Returns a map of tensor name → sizes tensor name.
    pub fn flat_keys(layouts: &HashMap<String, FieldLayout>) -> HashMap<String, String> {
        layouts
            .iter()
            .filter_map(|(k, l)| match l {
                FieldLayout::Flat { sizes_key } => Some((k.clone(), sizes_key.clone())),
                FieldLayout::Batched => None,
            })
            .collect()
    }
}

/// Trait for model-specific video preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline.
pub trait VideoPreProcessor: Send + Sync {
    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of videos.
    ///
    /// # Arguments
    /// * `videos` - Input videos to preprocess
    /// * `config` - Preprocessor configuration from HuggingFace
    ///
    /// # Returns
    /// Preprocessed videos ready for the model, or an error.
    fn preprocess(
        &self,
        videos: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedVideos, TransformError>;

    /// Calculate the number of video tokens for a given video size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the video has been fully processed.
    ///
    /// # Arguments
    /// * `width` - Video width after preprocessing
    /// * `height` - Video height after preprocessing
    /// * `num_frames` - Number of frames in the video after preprocessing
    /// * `config` - Preprocessor configuration
    fn calculate_num_tokens(&self, width: u32, height: u32, num_frames: u32, config: &PreProcessorConfig) -> usize;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;
}

/// Registry of available video processors.
pub struct VideoProcessorRegistry {
    processors: HashMap<String, Box<dyn VideoPreProcessor>>,
}

impl VideoProcessorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
        }
    }

    /// Register a processor for a model pattern.
    pub fn register(&mut self, pattern: impl Into<String>, processor: Box<dyn VideoPreProcessor>) {
        self.processors.insert(pattern.into(), processor);
    }

    /// Find a processor for the given model ID, falling back to model_type.
    ///
    /// Matches by substring containment (case-insensitive).
    pub fn find(&self, model_id: &str, model_type: Option<&str>) -> Option<&dyn VideoPreProcessor> {
        self.find_in_candidate(model_id)
            .or_else(|| model_type.and_then(|mt| self.find_in_candidate(mt)))
    }

    fn find_in_candidate(&self, candidate: &str) -> Option<&dyn VideoPreProcessor> {
        let candidate = candidate.to_lowercase();
        for (pattern, processor) in &self.processors {
            if candidate.contains(&pattern.to_lowercase()) {
                return Some(processor.as_ref());
            }
        }
        None
    }

    /// Get list of supported model patterns.
    pub fn supported_patterns(&self) -> Vec<&str> {
        self.processors.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for VideoProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoProcessorRegistry {
    /// Create a registry with all built-in processors registered.
    ///
    /// Currently registers:
    /// - `llava-next` -> LlavaNextProcessor
    /// - `llava-1.5` / `llava-v1.5` -> LlavaProcessor
    /// - `qwen2-vl` -> Qwen2VLProcessor
    /// - `qwen2.5-vl` -> Qwen2VLProcessor (same preprocessing as Qwen2-VL)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // LLaVA-NeXT (v1.6+, anyres multi-crop)
        registry.register(
            "llava-next",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );
        registry.register(
            "llava_next",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );
        registry.register(
            "llava-v1.6",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );

        // Standard LLaVA (v1.5, single-patch).
        // Use specific patterns so they don't accidentally match LLaVA-NeXT
        // model IDs like "llava-v1.6-*".
        registry.register(
            "llava-1.5",
            Box::new(super::processors::LlavaProcessor::new()),
        );
        registry.register(
            "llava-v1.5",
            Box::new(super::processors::LlavaProcessor::new()),
        );

        // Register Qwen3-VL first (more specific pattern - must match before qwen2)
        registry.register(
            "qwen3-vl",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );
        registry.register(
            "qwen3_vl",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );

        // Register Qwen2-VL (matches Qwen/Qwen2-VL-*, etc.)
        registry.register(
            "qwen2-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Register Qwen2.5-VL (uses identical preprocessing to Qwen2-VL)
        registry.register(
            "qwen2.5-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_5-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_5_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Register Phi3-Vision
        registry.register(
            "phi-3-vision",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );
        registry.register(
            "phi3-vision",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );
        registry.register(
            "phi3_v",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );

        // Register LLaMA 4 Vision
        registry.register(
            "llama-4",
            Box::new(super::processors::Llama4VisionProcessor::new()),
        );
        registry.register(
            "llama4",
            Box::new(super::processors::Llama4VisionProcessor::new()),
        );

        // Register Kimi-K2.5 Vision
        registry.register(
            "kimi-k2",
            Box::new(super::processors::KimiK25Processor::new()),
        );
        registry.register(
            "kimi_k2",
            Box::new(super::processors::KimiK25Processor::new()),
        );

        registry
    }
}
