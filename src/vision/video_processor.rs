//! Video processor trait and output types.
//!
//! This module defines the interface for model-specific video preprocessors
//! and the common output format for preprocessed videos.
//!
//! A "video" is represented as an ordered slice of decoded frames
//! (`&[DynamicImage]`), and a batch of videos as `&[Vec<DynamicImage>]`.
//! Preprocessors group frames along the temporal axis (see
//! [`VideoPreProcessor::preprocess`]) and produce a patchified tensor plus the
//! per-video grid dimensions needed for position encoding.

use std::{borrow::Cow, collections::HashMap};

use image::DynamicImage;
use ndarray::{Array2, ArrayD};

use super::{
    image_processor::ModelSpecificValue, preprocessor_config::PreProcessorConfig,
    transforms::TransformError,
};
use crate::types::FieldLayout;

/// Preprocessed videos ready for model consumption.
///
/// This struct contains all the outputs needed by the SGLang scheduler
/// to construct `MultimodalInputs` for the model.
#[derive(Debug, Clone)]
pub struct PreprocessedVideos {
    /// Pixel values as a dynamic-dimensional float32 tensor.
    ///
    /// This is the primary input to the vision encoder. For Qwen-VL family
    /// models it is patchified to 2D `[total_patches, patch_features]`, where
    /// patches from every video in the batch are concatenated along axis 0.
    pub pixel_values: ArrayD<f32>,

    /// Number of video tokens per video in the batch.
    ///
    /// Used to expand placeholder tokens in the text input.
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
    /// Create a new PreprocessedVideos from a patchified 2D pixel-values tensor.
    pub fn new(
        pixel_values: Array2<f32>,
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
/// Each vision model that supports video input (Qwen2-VL, Qwen2.5-VL, etc.)
/// implements this trait to provide the correct preprocessing pipeline.
pub trait VideoPreProcessor: Send + Sync {
    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of videos.
    ///
    /// # Arguments
    /// * `videos` - Batch of videos; each video is an ordered slice of frames.
    /// * `config` - Preprocessor configuration from HuggingFace
    ///
    /// # Returns
    /// Preprocessed videos ready for the model, or an error.
    fn preprocess(
        &self,
        videos: &[Vec<DynamicImage>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedVideos, TransformError>;

    /// Calculate the number of video tokens for a given video size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the video has been fully processed.
    ///
    /// # Arguments
    /// * `width` - Frame width before preprocessing
    /// * `height` - Frame height before preprocessing
    /// * `num_frames` - Number of sampled frames in the video
    /// * `config` - Preprocessor configuration
    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        num_frames: u32,
        config: &PreProcessorConfig,
    ) -> usize;

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
    /// Create a registry with all built-in video processors registered.
    ///
    /// Currently registers:
    /// - `qwen2-vl` / `qwen2_vl` -> Qwen2VLProcessor
    /// - `qwen2.5-vl` / `qwen2_5-vl` / `qwen2_5_vl` -> Qwen2VLProcessor
    ///   (Qwen2.5-VL uses identical video preprocessing to Qwen2-VL)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Qwen2-VL (matches Qwen/Qwen2-VL-*, etc.)
        registry.register(
            "qwen2-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Qwen2.5-VL (identical preprocessing to Qwen2-VL)
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

        registry
    }
}
