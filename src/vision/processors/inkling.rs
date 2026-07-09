//! Inkling image processor.
//!
//! Matches the vendored vLLM `InklingImageProcessor` image path. Each image is
//! split into normalized NHWC patches with one extra patch column, then each
//! patch is duplicated along the temporal axis to produce
//! `vision_patches_bthwc` shaped `[num_patches, 2, patch, patch, 3]`.

use image::{DynamicImage, GenericImageView};
use ndarray::ArrayD;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::{self, TransformError},
};

pub const INKLING_IMAGE_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const INKLING_IMAGE_STD: [f64; 3] = [0.26862954, 0.2613026, 0.2757771];
pub const DEFAULT_PATCH_SIZE: usize = 40;

const PAD_RAW_VALUE: f32 = -1.0 / 255.0;

#[derive(Debug, Clone)]
pub struct InklingImageProcessor {
    patch_size: usize,
}

impl Default for InklingImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingImageProcessor {
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
        }
    }

    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.patch_size.is_some() {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }

    fn grid(&self, width: usize, height: usize) -> Result<(usize, usize), TransformError> {
        if self.patch_size == 0 {
            return Err(TransformError::ShapeError(
                "patch_size must be greater than zero".to_string(),
            ));
        }
        let grid_h = height.div_ceil(self.patch_size);
        let grid_w = width / self.patch_size + 1;
        Ok((grid_h, grid_w))
    }

    fn normalized_padding(mean: &[f64; 3], std: &[f64; 3]) -> [f32; 3] {
        std::array::from_fn(|c| (PAD_RAW_VALUE - mean[c] as f32) / std[c] as f32)
    }

    fn fill_image_patches(
        &self,
        image: &DynamicImage,
        mean: &[f64; 3],
        std: &[f64; 3],
        all_patches: &mut Vec<f32>,
    ) -> Result<usize, TransformError> {
        let (width, height, raw) = transforms::rgb_bytes(image);
        let (grid_h, grid_w) = self.grid(width, height)?;
        let num_patches = grid_h * grid_w;
        let patch_area = self.patch_size * self.patch_size;
        let pad = Self::normalized_padding(mean, std);
        let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
        let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / std[c] as f32);

        all_patches.reserve(num_patches * 2 * patch_area * 3);
        for patch_idx in 0..num_patches {
            let patch_y = patch_idx / grid_w;
            let patch_x = patch_idx - patch_y * grid_w;
            let y_base = patch_y * self.patch_size;
            let x_base = patch_x * self.patch_size;
            let start = all_patches.len();

            for _temporal in 0..2 {
                for y in 0..self.patch_size {
                    let src_y = y_base + y;
                    for x in 0..self.patch_size {
                        let src_x = x_base + x;
                        if src_y < height && src_x < width {
                            let src = (src_y * width + src_x) * 3;
                            all_patches.push(raw[src] as f32 * scale[0] + bias[0]);
                            all_patches.push(raw[src + 1] as f32 * scale[1] + bias[1]);
                            all_patches.push(raw[src + 2] as f32 * scale[2] + bias[2]);
                        } else {
                            all_patches.extend_from_slice(&pad);
                        }
                    }
                }
            }

            debug_assert_eq!(all_patches.len() - start, 2 * patch_area * 3);
        }

        Ok(num_patches)
    }
}

impl VisionPreProcessor for InklingImageProcessor {
    fn default_mean(&self) -> [f64; 3] {
        INKLING_IMAGE_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        INKLING_IMAGE_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let processor = self.with_preprocessor_config(config);
        let mean = config.image_mean_3().unwrap_or(INKLING_IMAGE_MEAN);
        let std = config.image_std_3().unwrap_or(INKLING_IMAGE_STD);
        let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();

        let mut patches = Vec::new();
        let mut num_patches = Vec::with_capacity(images.len());
        for image in images {
            let count = processor.fill_image_patches(image, &mean, &std, &mut patches)?;
            num_patches.push(count);
        }

        let total_patches: usize = num_patches.iter().sum();
        let patch = processor.patch_size;
        let tensor_shape = vec![total_patches, 2, patch, patch, 3];
        let vision_patches_bthwc = patches.clone();
        let encoder_input =
            ArrayD::from_shape_vec(tensor_shape.clone(), patches).map_err(
                |error| {
                    TransformError::ShapeError(format!(
                        "Failed to create vision_patches_bthwc [{total_patches}, 2, {patch}, {patch}, 3]: {error}"
                    ))
                },
            )?;
        let num_patches_i64 = num_patches
            .iter()
            .map(|&count| count as i64)
            .collect::<Vec<_>>();

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, num_patches, item_sizes)
                .with_extra(
                    "vision_patches_bthwc",
                    ModelSpecificValue::Tensor {
                        data: vision_patches_bthwc,
                        shape: tensor_shape,
                    },
                )
                .with_extra("num_patches", ModelSpecificValue::int_1d(num_patches_i64)),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = self.with_preprocessor_config(config);
        let Ok((grid_h, grid_w)) = processor.grid(width as usize, height as usize) else {
            return 0;
        };
        grid_h * grid_w
    }

    fn model_name(&self) -> &'static str {
        "inkling"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}

trait InklingConfigExt {
    fn image_mean_3(&self) -> Option<[f64; 3]>;
    fn image_std_3(&self) -> Option<[f64; 3]>;
}

impl InklingConfigExt for PreProcessorConfig {
    fn image_mean_3(&self) -> Option<[f64; 3]> {
        self.image_mean.as_deref().and_then(slice_to_three)
    }

    fn image_std_3(&self) -> Option<[f64; 3]> {
        self.image_std.as_deref().and_then(slice_to_three)
    }
}

fn slice_to_three(values: &[f64]) -> Option<[f64; 3]> {
    let [a, b, c] = values else {
        return None;
    };
    Some([*a, *b, *c])
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::preprocessor_config::PatchSize;

    fn image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn preprocess_outputs_inkling_patch_shape() {
        let processor = InklingImageProcessor::new();
        let result = processor
            .preprocess(
                &[image(80, 40, Rgb([128, 128, 128]))],
                &PreProcessorConfig::default(),
            )
            .unwrap();

        assert_eq!(result.encoder_input.shape(), &[3, 2, 40, 40, 3]);
        assert_eq!(result.feature_token_counts, vec![3]);
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("num_patches")
        {
            assert_eq!(shape, &[1]);
            assert_eq!(data, &[3]);
        } else {
            panic!("expected num_patches tensor");
        }
    }

    #[test]
    fn preprocess_multiple_images_are_flattened_by_patch_count() {
        let processor = InklingImageProcessor::new();
        let result = processor
            .preprocess(
                &[
                    image(80, 40, Rgb([128, 128, 128])),
                    image(1, 1, Rgb([255, 255, 255])),
                ],
                &PreProcessorConfig::default(),
            )
            .unwrap();

        assert_eq!(result.encoder_input.shape(), &[4, 2, 40, 40, 3]);
        assert_eq!(result.feature_token_counts, vec![3, 1]);
    }

    #[test]
    fn patch_size_can_come_from_preprocessor_config() {
        let processor = InklingImageProcessor::new();
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(20),
                width: Some(20),
            }),
            ..Default::default()
        };

        let result = processor
            .preprocess(&[image(40, 20, Rgb([0, 0, 0]))], &config)
            .unwrap();

        assert_eq!(result.encoder_input.shape(), &[3, 2, 20, 20, 3]);
    }

    #[test]
    fn padding_uses_inkling_raw_pad_value_before_normalization() {
        let processor = InklingImageProcessor::new();
        let result = processor
            .preprocess(
                &[image(1, 1, Rgb([255, 255, 255]))],
                &PreProcessorConfig::default(),
            )
            .unwrap();
        let flat = result.encoder_input_flat();
        let pad =
            InklingImageProcessor::normalized_padding(&INKLING_IMAGE_MEAN, &INKLING_IMAGE_STD);

        assert!(flat.iter().any(|&v| (v - pad[0]).abs() < 1e-6));
        assert!(flat.iter().any(|&v| (v - 1.0 / INKLING_IMAGE_STD[0] as f32
            + INKLING_IMAGE_MEAN[0] as f32 / INKLING_IMAGE_STD[0] as f32)
            .abs()
            < 1e-6));
    }
}
