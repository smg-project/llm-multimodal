pub mod audio;
pub mod encoder_inputs;
pub mod error;
pub mod hasher;
#[cfg(feature = "hf-hub")]
pub mod hub;
pub mod jpeg_turbo;
pub mod media;
#[cfg(feature = "opencv-video")]
mod opencv_buffer;
pub mod registry;
pub mod tracker;
pub mod types;
pub mod vision;

pub use audio::AudioPreProcessor;
pub use encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs};
pub use error::{MediaConnectorError, MultiModalError, MultiModalResult, TransformError};
pub use media::{
    ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource, VideoFetchConfig,
};
pub use registry::{
    DEEPSEEK_V41_IMAGE_PLACEHOLDER, ModelMetadata, ModelProcessorSpec, ModelRegistry, Tokenizer,
};
pub use tracker::{AsyncMultiModalTracker, TrackerOutput};
pub use types::{
    AlignmentPad, AudioClip, AudioSource, EncoderFieldLayouts, FieldLayout, ImageDetail,
    ImageFrame, ImageSize, ImageSource, MediaContentPart, Modality, MultiModalData,
    MultiModalUUIDs, PlaceholderRange, PromptReplacement, RgbFrameRef, TokenId, TrackedMedia,
    VideoClip, VideoSource,
};
// Re-export vision processing components
pub use vision::{
    DeepseekV41Processor, InklingImageProcessor, KimiK3Processor, LlavaNextProcessor,
    LlavaProcessor, PreProcessorConfig, VisionPreProcessor, VisionProcessorRegistry,
};
