pub mod error;
pub mod hasher;
pub mod media;
pub mod registry;
pub mod tracker;
pub mod types;
pub mod vision;

#[cfg(feature = "video")]
pub mod video;

pub use error::{MediaConnectorError, MultiModalError, MultiModalResult};
pub use media::{ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource};
pub use registry::{ModelMetadata, ModelProcessorSpec, ModelRegistry, TokenResolver};
pub use tracker::{AsyncMultiModalTracker, TrackerOutput};
pub use types::{
    FieldLayout, ImageDetail, ImageFrame, ImageSize, ImageSource, MediaContentPart, Modality,
    MultiModalData, MultiModalUUIDs, PlaceholderRange, PromptReplacement, TokenId, TrackedMedia,
};
// Re-export vision processing components
pub use vision::{
    ImagePreProcessor, ImageProcessorRegistry, LlavaNextProcessor, LlavaProcessor,
    ModelSpecificValue, PreProcessorConfig, PreprocessedImages, TransformError,
};

#[cfg(feature = "video")]
pub use media::VideoFetchConfig;
#[cfg(feature = "video")]
pub use types::{VideoFrame, VideoSource};
#[cfg(feature = "video")]
pub use video::{
    AllFramesSampler, FpsSampler, FrameSampler, UniformSampler, VideoDecodeError, VideoMetadata,
};
