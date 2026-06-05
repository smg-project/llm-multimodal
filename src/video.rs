//! Video decoding and frame sampling.
//!
//! Provides OpenCV-based video decoding with pluggable frame sampling strategies.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use image::{DynamicImage, RgbImage};
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::VideoCaptureTrait;
use opencv::videoio::{self, VideoCapture};
use tempfile::NamedTempFile;

/// Metadata extracted from a video container.
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Estimated total number of frames.
    pub total_frames: usize,
    /// Frames per second.
    pub fps: f64,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Codec fourcc (e.g., "H264", "XVID").
    pub codec: String,
}

/// Errors from video decoding.
#[derive(Debug, thiserror::Error)]
pub enum VideoDecodeError {
    #[error("video decode error: {0}")]
    OpenCV(#[from] opencv::Error),
    #[error("no video stream found")]
    NoVideoStream,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("frame conversion failed: {0}")]
    FrameConversion(String),
}

/// A frame sampling strategy that decides which frames to extract from a video.
///
/// Implementations receive video metadata and return the 0-based indices of
/// frames to decode. Different vision processors can provide their own sampler
/// to control temporal resolution.
pub trait FrameSampler: Send + Sync {
    /// Return the 0-based frame indices to extract.
    fn sample_indices(&self, meta: &VideoMetadata) -> Vec<usize>;
}

// ---------------------------------------------------------------------------
// Built-in samplers
// ---------------------------------------------------------------------------

/// Sample `num_frames` uniformly distributed frames across the video.
#[derive(Debug, Clone)]
pub struct UniformSampler {
    pub num_frames: usize,
}

impl FrameSampler for UniformSampler {
    fn sample_indices(&self, meta: &VideoMetadata) -> Vec<usize> {
        if meta.total_frames == 0 || self.num_frames == 0 {
            return Vec::new();
        }
        let n = self.num_frames.min(meta.total_frames);
        (0..n).map(|i| i * meta.total_frames / n).collect()
    }
}

/// Sample frames at a target FPS.
#[derive(Debug, Clone)]
pub struct FpsSampler {
    pub target_fps: f64,
}

impl FrameSampler for FpsSampler {
    fn sample_indices(&self, meta: &VideoMetadata) -> Vec<usize> {
        if meta.fps <= 0.0 || meta.total_frames == 0 {
            return Vec::new();
        }
        let step = (meta.fps / self.target_fps).max(1.0) as usize;
        (0..meta.total_frames).step_by(step).collect()
    }
}

/// Decode all frames in the video.
///
/// **Warning**: This can consume large amounts of memory for long videos.
/// A 30-second 1080p video at 30fps requires ~2.7 GB of decoded pixel data.
#[derive(Debug, Clone, Default)]
pub struct AllFramesSampler;

impl FrameSampler for AllFramesSampler {
    fn sample_indices(&self, meta: &VideoMetadata) -> Vec<usize> {
        (0..meta.total_frames).collect()
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decode video from raw bytes using a pluggable sampling strategy.
///
/// Writes `bytes` to a temporary file, opens it with OpenCV, probes metadata,
/// applies the sampler, and decodes only the requested frames.
///
/// This function is synchronous and intended to be called inside
/// `tokio::task::spawn_blocking`.
pub fn decode_video_bytes(
    bytes: &[u8],
    sampler: &dyn FrameSampler,
) -> Result<(VideoMetadata, Vec<DynamicImage>), VideoDecodeError> {
    let mut temp = NamedTempFile::new()?;
    temp.write_all(bytes)?;
    temp.flush()?;
    decode_video_path_inner(temp.path(), sampler)
}

/// Decode video from a file path using a pluggable sampling strategy.
///
/// This function is synchronous and intended to be called inside
/// `tokio::task::spawn_blocking`.
pub fn decode_video_path(
    path: &Path,
    sampler: &dyn FrameSampler,
) -> Result<(VideoMetadata, Vec<DynamicImage>), VideoDecodeError> {
    decode_video_path_inner(path, sampler)
}

fn decode_video_path_inner(
    path: &Path,
    sampler: &dyn FrameSampler,
) -> Result<(VideoMetadata, Vec<DynamicImage>), VideoDecodeError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| VideoDecodeError::FrameConversion("non-UTF-8 path".into()))?;

    let mut cap = VideoCapture::from_file(path_str, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(VideoDecodeError::NoVideoStream);
    }

    // Extract metadata
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;
    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
    let duration_secs = if fps > 0.0 {
        total_frames as f64 / fps
    } else {
        0.0
    };
    let fourcc_int = cap.get(videoio::CAP_PROP_FOURCC)? as u32;
    let codec: String = fourcc_int
        .to_le_bytes()
        .iter()
        .filter(|&&b| b != 0)
        .map(|&b| b as char)
        .collect();

    let metadata = VideoMetadata {
        duration_secs,
        total_frames: total_frames.max(1),
        fps,
        width,
        height,
        codec,
    };

    // Ask sampler which frames to extract
    let indices = sampler.sample_indices(&metadata);
    if indices.is_empty() {
        return Ok((metadata, Vec::new()));
    }

    let desired: HashSet<usize> = indices.iter().copied().collect();
    let max_idx = *indices.iter().max().unwrap();

    // Read frames sequentially, keep only desired ones
    let mut frames = Vec::with_capacity(desired.len());
    let mut mat = Mat::default();
    let mut frame_count: usize = 0;

    loop {
        if !cap.read(&mut mat)? {
            break;
        }
        if mat.empty()? {
            break;
        }

        if desired.contains(&frame_count) {
            frames.push((frame_count, mat_to_dynamic_image(&mat)?));
        }

        frame_count += 1;
        if frame_count > max_idx || frames.len() == desired.len() {
            break;
        }
    }

    // Reorder to match sampler's requested order
    let mut result = Vec::with_capacity(indices.len());
    for idx in &indices {
        if let Some(pos) = frames.iter().position(|(i, _)| i == idx) {
            result.push(frames[pos].1.clone());
        }
    }

    Ok((metadata, result))
}

/// Convert an OpenCV BGR `Mat` to a `DynamicImage` (RGB).
fn mat_to_dynamic_image(mat: &Mat) -> Result<DynamicImage, VideoDecodeError> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let w = rgb.cols() as u32;
    let h = rgb.rows() as u32;
    let data = rgb.data_bytes()?.to_vec();

    RgbImage::from_raw(w, h, data)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| {
            VideoDecodeError::FrameConversion(format!("failed to create {}x{} image", w, h))
        })
}
