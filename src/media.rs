use std::{collections::HashSet, path::PathBuf, sync::Arc, time::Duration};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::Bytes;
use reqwest::Client;
use tokio::{fs, task};
use url::Url;

use super::{
    error::MediaConnectorError,
    registry::{ModelMetadata, ModelRegistry, RegistryResult},
    types::{ImageDetail, ImageFrame, ImageSource, VideoFrame, VideoSource},
    video::{FrameSampler, UniformSampler},
    vision::PreProcessorConfig,
};

#[derive(Clone)]
pub struct MediaConnectorConfig {
    pub allowed_domains: Option<Vec<String>>,
    pub allowed_local_media_path: Option<PathBuf>,
    pub fetch_timeout: Duration,
}

impl Default for MediaConnectorConfig {
    fn default() -> Self {
        Self {
            allowed_domains: None,
            allowed_local_media_path: None,
            fetch_timeout: Duration::from_secs(10),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageFetchConfig {
    pub detail: ImageDetail,
}

impl Default for ImageFetchConfig {
    fn default() -> Self {
        Self {
            detail: ImageDetail::Auto,
        }
    }
}

#[derive(Clone)]
pub struct VideoFetchConfig {
    pub sampler: Arc<dyn FrameSampler>,
}

impl Default for VideoFetchConfig {
    fn default() -> Self {
        Self::new(UniformSampler { num_frames: 8 })
    }
}

impl VideoFetchConfig {
    pub fn new(sampler: impl FrameSampler + 'static) -> Self {
        Self {
            sampler: Arc::new(sampler),
        }
    }

    pub fn from_model(
        metadata: &ModelMetadata,
        preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Self> {
        let registry = ModelRegistry::new();
        let Some(spec) = registry.lookup(metadata) else {
            return Ok(Self::default());
        };

        let Some(sampler) = spec.build_video_sampler(metadata, preprocessor_config)? else {
            return Ok(Self::default());
        };

        Ok(Self {
            sampler: Arc::from(sampler),
        })
    }
}

#[derive(Debug, Clone)]
pub enum MediaSource {
    Url(String),
    DataUrl(String),
    InlineBytes(Vec<u8>),
    File(PathBuf),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::test_helpers::TestTokenizer;
    use crate::video::VideoMetadata;

    fn sample_video_metadata() -> VideoMetadata {
        VideoMetadata {
            duration_secs: 4.0,
            total_frames: 24,
            fps: 6.0,
            width: 640,
            height: 480,
            codec: "TEST".into(),
        }
    }

    #[test]
    fn model_specific_video_sampler_uses_qwen2_vl_sampler() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 1), ("<|video_pad|>", 2)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "image_token_id": 151655
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let preprocessor_config = PreProcessorConfig {
            do_sample_frames: Some(true),
            num_frames: Some(5),
            temporal_patch_size: Some(2),
            ..Default::default()
        };

        let fetch_config = VideoFetchConfig::from_model(&metadata, &preprocessor_config).unwrap();
        let indices = fetch_config
            .sampler
            .sample_indices(&sample_video_metadata())
            .unwrap();

        assert_eq!(indices, vec![0, 6, 12, 18]);
    }

    #[test]
    fn model_without_hook_uses_default_uniform_sampler() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "custom"});
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };

        let fetch_config =
            VideoFetchConfig::from_model(&metadata, &PreProcessorConfig::default()).unwrap();
        let indices = fetch_config
            .sampler
            .sample_indices(&sample_video_metadata())
            .unwrap();

        assert_eq!(indices, vec![0, 3, 6, 9, 12, 15, 18, 21]);
    }
}

#[derive(Clone)]
pub struct MediaConnector {
    client: Client,
    allowed_domains: Option<HashSet<String>>,
    allowed_local_media_path: Option<PathBuf>,
    fetch_timeout: Duration,
}

impl MediaConnector {
    pub fn new(client: Client, config: MediaConnectorConfig) -> Result<Self, MediaConnectorError> {
        let allowed_domains = config.allowed_domains.map(|domains| {
            domains
                .into_iter()
                .map(|d| d.to_ascii_lowercase())
                .collect::<HashSet<_>>()
        });

        let allowed_local_media_path = if let Some(path) = config.allowed_local_media_path {
            Some(std::fs::canonicalize(path)?)
        } else {
            None
        };

        Ok(Self {
            client,
            allowed_domains,
            allowed_local_media_path,
            fetch_timeout: config.fetch_timeout,
        })
    }

    pub async fn fetch_image(
        &self,
        source: MediaSource,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        match source {
            MediaSource::Url(url) => self.fetch_http_image(url, cfg).await,
            MediaSource::DataUrl(data_url) => self.fetch_data_url_image(data_url, cfg).await,
            MediaSource::InlineBytes(bytes) => {
                self.decode_image(bytes.into(), cfg.detail, ImageSource::InlineBytes)
                    .await
            }
            MediaSource::File(path) => self.fetch_file_image(path, cfg).await,
        }
    }

    // -----------------------------------------------------------------------
    // Video fetching
    // -----------------------------------------------------------------------

    pub async fn fetch_video(
        &self,
        source: MediaSource,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoFrame>, MediaConnectorError> {
        match source {
            MediaSource::File(path) => self.fetch_file_video(path, cfg).await,
            other => {
                // For non-file sources, resolve to bytes first
                let (bytes, video_source) = self.resolve_bytes(other).await?;
                self.decode_video_bytes(bytes, video_source, cfg).await
            }
        }
    }

    async fn resolve_bytes(
        &self,
        source: MediaSource,
    ) -> Result<(Bytes, VideoSource), MediaConnectorError> {
        match source {
            MediaSource::Url(url) => {
                let bytes = self.fetch_http_bytes(&url).await?;
                Ok((bytes, VideoSource::Url { url }))
            }
            MediaSource::DataUrl(data_url) => {
                let bytes = self.decode_data_url_bytes(&data_url)?;
                Ok((bytes, VideoSource::DataUrl))
            }
            MediaSource::InlineBytes(bytes) => Ok((bytes.into(), VideoSource::InlineBytes)),
            MediaSource::File(path) => {
                let bytes = self.fetch_file_bytes(&path).await?;
                Ok((bytes, VideoSource::File { path }))
            }
        }
    }

    async fn fetch_file_video(
        &self,
        path: PathBuf,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoFrame>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let container_hash = {
            let raw = fs::read(&canonical).await?;
            crate::hasher::hash_bytes(&raw)
        };

        let sampler = Arc::clone(&cfg.sampler);
        let canonical_clone = canonical.clone();
        let decode_result = task::spawn_blocking(move || {
            crate::video::decode_video_path(&canonical_clone, sampler.as_ref())
        })
        .await
        .map_err(MediaConnectorError::Blocking)??;

        let frame_hashes: Vec<String> = decode_result
            .1
            .iter()
            .map(|img| {
                let rgb = img.to_rgb8();
                crate::hasher::hash_bytes(rgb.as_raw())
            })
            .collect();

        Ok(Arc::new(VideoFrame::new(
            decode_result.1,
            container_hash,
            frame_hashes,
            decode_result.0,
            VideoSource::File { path: canonical },
        )))
    }

    async fn decode_video_bytes(
        &self,
        bytes: Bytes,
        source: VideoSource,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoFrame>, MediaConnectorError> {
        let container_hash = crate::hasher::hash_bytes(&bytes);

        let sampler = Arc::clone(&cfg.sampler);
        let decode_result = task::spawn_blocking(move || {
            crate::video::decode_video_bytes(&bytes, sampler.as_ref())
        })
        .await
        .map_err(MediaConnectorError::Blocking)??;

        let frame_hashes: Vec<String> = decode_result
            .1
            .iter()
            .map(|img| {
                let rgb = img.to_rgb8();
                crate::hasher::hash_bytes(rgb.as_raw())
            })
            .collect();

        Ok(Arc::new(VideoFrame::new(
            decode_result.1,
            container_hash,
            frame_hashes,
            decode_result.0,
            source,
        )))
    }

    // -----------------------------------------------------------------------
    // Shared byte-fetching helpers
    // -----------------------------------------------------------------------

    async fn fetch_http_bytes(&self, url: &str) -> Result<Bytes, MediaConnectorError> {
        let parsed =
            Url::parse(url).map_err(|_| MediaConnectorError::InvalidUrl(url.to_string()))?;
        self.ensure_domain_allowed(&parsed)?;

        let mut req = self.client.get(parsed.as_str());
        if self.fetch_timeout > Duration::ZERO {
            req = req.timeout(self.fetch_timeout);
        }

        let resp = req.send().await.map_err(|err| {
            if err.is_timeout() {
                MediaConnectorError::Timeout(self.fetch_timeout)
            } else {
                MediaConnectorError::Http(err)
            }
        })?;

        let resp = resp.error_for_status()?;
        resp.bytes().await.map_err(MediaConnectorError::Http)
    }

    fn decode_data_url_bytes(&self, data_url: &str) -> Result<Bytes, MediaConnectorError> {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;

        if !metadata.ends_with(";base64") {
            return Err(MediaConnectorError::DataUrl(
                "only base64 encoded data URLs are supported".into(),
            ));
        }

        let data = data.trim();
        let decoded = BASE64_STANDARD.decode(data)?;
        Ok(decoded.into())
    }

    async fn fetch_file_bytes(&self, path: &PathBuf) -> Result<Bytes, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let bytes = fs::read(&canonical).await?;
        Ok(bytes.into())
    }

    // -----------------------------------------------------------------------
    // Image fetching (refactored to use shared helpers)
    // -----------------------------------------------------------------------

    async fn fetch_http_image(
        &self,
        url: String,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let bytes = self.fetch_http_bytes(&url).await?;
        self.decode_image(bytes, cfg.detail, ImageSource::Url { url })
            .await
    }

    async fn fetch_data_url_image(
        &self,
        data_url: String,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let bytes = self.decode_data_url_bytes(&data_url)?;
        self.decode_image(bytes, cfg.detail, ImageSource::DataUrl)
            .await
    }

    async fn fetch_file_image(
        &self,
        path: PathBuf,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let bytes = fs::read(&canonical).await?;
        self.decode_image(
            bytes.into(),
            cfg.detail,
            ImageSource::File { path: canonical },
        )
        .await
    }

    fn ensure_domain_allowed(&self, url: &Url) -> Result<(), MediaConnectorError> {
        if let Some(allowed) = &self.allowed_domains {
            let host = url
                .host_str()
                .map(|h| h.to_ascii_lowercase())
                .ok_or_else(|| MediaConnectorError::InvalidUrl(url.to_string()))?;
            if !allowed.contains(&host) {
                return Err(MediaConnectorError::DisallowedDomain(host));
            }
        }
        Ok(())
    }

    async fn decode_image(
        &self,
        bytes: Bytes,
        detail: ImageDetail,
        source: ImageSource,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let hash = crate::hasher::hash_bytes(&bytes);

        let cursor = std::io::Cursor::new(bytes.clone());
        let reader = image::ImageReader::new(cursor).with_guessed_format()?;

        let image = task::spawn_blocking(move || reader.decode())
            .await
            .map_err(MediaConnectorError::Blocking)??;

        Ok(Arc::new(ImageFrame::new(
            image, bytes, detail, source, hash,
        )))
    }
}
