//! Model-owned vision processor trait.
//!
//! Shared encoder output types live in [`crate::encoder_inputs`] and are re-exported
//! here for compatibility.

use image::DynamicImage;

use super::transforms::TransformError;
pub use crate::encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs};
use crate::{types::RgbFrameRef, PreprocessingContext};

/// Helper to extract a dimension from encoder_input given an ndim-dependent axis index.
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
            expected: format!("4D or 5D encoder_input tensor, got {ndim}D"),
            actual: shape.to_vec(),
        }),
    }
}

impl PreprocessedEncoderInputs {
    /// Get the number of channels.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[1].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[2].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn channels(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 1, 2, self.encoder_input.shape())
    }

    /// Get the height of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[2].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[3].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn height(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 2, 3, self.encoder_input.shape())
    }

    /// Get the width of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[3].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[4].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn width(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 3, 4, self.encoder_input.shape())
    }
}

/// Trait for model-specific vision preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline. Each instance owns its resolved
/// model parameters; request methods receive media and optional request context.
pub trait VisionPreProcessor: Send + Sync {
    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of images.
    ///
    /// # Arguments
    /// * `images` - Input images to preprocess
    ///
    /// # Returns
    /// Preprocessed encoder inputs ready for the model, or an error.
    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError>;

    /// Preprocess a batch with request-specific inputs.
    ///
    /// Processors that do not use the request context retain their usual path.
    fn preprocess_with_context(
        &self,
        images: &[DynamicImage],
        _context: &PreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess(images)
    }

    /// Preprocess one decoded video clip represented as sampled frames.
    ///
    /// Implementations that support video should emit the same primary
    /// `encoder_input` tensor shape used by the image path, plus video-specific
    /// model metadata such as `video_grid_thw`.
    fn preprocess_video(
        &self,
        _frames: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support video preprocessing",
            self.model_name()
        )))
    }

    /// Preprocess one decoded video clip represented as borrowed RGB frame
    /// buffers. Implementations can override this to avoid materializing
    /// `DynamicImage` objects after media decode.
    fn preprocess_video_rgb(
        &self,
        _frames: &[RgbFrameRef<'_>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support RGB video preprocessing",
            self.model_name()
        )))
    }

    /// Calculate the number of vision tokens for a given image size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the image has been fully processed.
    ///
    /// # Arguments
    /// * `width` - Image width after preprocessing
    /// * `height` - Image height after preprocessing
    fn calculate_num_tokens(&self, width: u32, height: u32) -> usize;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;

    /// Get the expected image size after preprocessing.
    ///
    /// Some models have fixed sizes, others are dynamic.
    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}
