//! Model-specific audio preprocessing implementations.

mod inkling;
mod parakeet;
mod qwen3_audio;

pub use inkling::{InklingAudioEncoderParams, InklingAudioProcessor};
pub use parakeet::{ParakeetAudioParams, ParakeetAudioProcessor, ParakeetConfigError};
pub use qwen3_audio::{Qwen3AudioParams, Qwen3AudioProcessor};
