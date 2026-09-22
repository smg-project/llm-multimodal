//! Pure Rust vision processing module for multimodal models.
//!
//! This module provides vision preprocessing pipelines that match HuggingFace
//! processor outputs without requiring Python dependencies.
//!
//! # Architecture
//!
//! The vision module is structured as follows:
//!
//! - `transforms`: Core image transformations (resize, normalize, crop, etc.)
//! - `preprocessor_config`: HuggingFace config parsing
//! - `processor`: Vision processor trait
//! - `processors`: Model-specific implementations (LLaVA, Qwen-VL, etc.)
//!
//! Modality-neutral encoder outputs live in [`crate::encoder_inputs`], while
//! shared errors live in [`crate::error`].
//!
//! # Usage
//!
//! ```rust,ignore
//! use smg::multimodal::vision::{
//!     PreProcessorConfig,
//!     processors::LlavaProcessor,
//!     VisionPreProcessor,
//! };
//!
//! // Load config from HuggingFace
//! let config = PreProcessorConfig::from_json(config_json)?;
//!
//! // Create processor and preprocess images
//! let processor = LlavaProcessor::from_configs(&model_config, &config);
//! let result = processor.preprocess(&images)?;
//! ```

pub(crate) mod execution;
pub mod preprocessor_config;
pub mod processor;
pub mod processors;
pub(crate) mod scratch;
pub mod transforms;

// Re-export commonly used types, including compatibility paths for shared
// preprocessing outputs.
pub use preprocessor_config::PreProcessorConfig;
pub use processor::{
    ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor, VisionPreprocessingContext,
};
pub use processors::{
    DeepseekV41Processor, InklingImageProcessor, KimiK3Processor, Llama4VisionProcessor,
    LlavaNextProcessor, LlavaProcessor, MiniMaxM3Processor, NemotronHOmniProcessor,
    Phi3VisionProcessor, Phi4VisionProcessor, PixtralProcessor, Qwen2VLProcessor,
    Qwen3OmniVisionProcessor, Qwen3VLProcessor,
};
pub use transforms::TransformError;
