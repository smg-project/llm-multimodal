use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig},
    registry::ModelMetadata,
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
    vision::PreProcessorConfig,
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    video_fetch_config: VideoFetchConfig,
    pending: HashMap<Modality, Vec<PendingTask>>,
    uuids: MultiModalUUIDs,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self::with_video_fetch_config(media_connector, VideoFetchConfig::default())
    }

    pub fn with_video_fetch_config(
        media_connector: Arc<MediaConnector>,
        video_fetch_config: VideoFetchConfig,
    ) -> Self {
        Self {
            media_connector,
            video_fetch_config,
            pending: HashMap::new(),
            uuids: HashMap::new(),
        }
    }

    pub fn for_model(
        media_connector: Arc<MediaConnector>,
        metadata: &ModelMetadata,
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Self, super::registry::ModelRegistryError> {
        let video_fetch_config = VideoFetchConfig::from_model(metadata, preprocessor_config)?;
        Ok(Self::with_video_fetch_config(
            media_connector,
            video_fetch_config,
        ))
    }

    pub fn push_part(&mut self, part: MediaContentPart) -> MultiModalResult<()> {
        match part {
            MediaContentPart::Text { .. } => {}
            MediaContentPart::ImageUrl { url, detail, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_image(source, detail.unwrap_or_default(), uuid);
            }
            MediaContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => {
                self.enqueue_image(
                    MediaSource::InlineBytes(data),
                    detail.unwrap_or_default(),
                    uuid,
                );
            }
            MediaContentPart::ImageEmbeds { .. } => {
                return Err(MultiModalError::UnsupportedContent("image_embeds"));
            }
            MediaContentPart::VideoUrl { url, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                let fetch_cfg = self.video_fetch_config.clone();
                self.enqueue_video(source, fetch_cfg, uuid);
            }
            MediaContentPart::VideoData {
                data,
                mime_type: _,
                uuid,
            } => {
                let fetch_cfg = self.video_fetch_config.clone();
                self.enqueue_video(MediaSource::InlineBytes(data), fetch_cfg, uuid);
            }
        }
        Ok(())
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
    }

    fn enqueue_image(&mut self, source: MediaSource, detail: ImageDetail, uuid: Option<String>) {
        let modality = Modality::Image;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(source, ImageFetchConfig { detail })
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_video(
        &mut self,
        source: MediaSource,
        fetch_cfg: VideoFetchConfig,
        uuid: Option<String>,
    ) {
        let modality = Modality::Video;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize()"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector.fetch_video(source, fetch_cfg).await?;
            Ok(TrackedMedia::Video(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }
}
