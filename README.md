# llm-multimodal

Rust multimodal preprocessing library extracted from
`lightseekorg/smg/crates/multimodal`.

## Metadata-only images

`ModelProcessorSpec::prepare_metadata_only` accepts a model context, the same
`PreProcessorConfig` used by the encoder, a modality, per-image JSON objects,
and a token budget. It returns auxiliary fields and prompt replacements without
allocating pixels or embedding tensors. Raw images and metadata-only inputs
share the model's prompt rules through `EncoderMetadata`.

The producer calls `spec.export_metadata(preprocessed.as_metadata(), modality)`
to publish the actual preprocessor output. Publication and consumption are
registered together in `ModelProcessorSpec::metadata_only_codec`; no frontend
model-name switch is needed. Only explicitly declared fields are published,
in media-item order, excluding pixels and embeddings.

```rust,ignore
let payloads = spec.export_metadata(preprocessed.as_metadata(), Modality::Image)?;
// Send payloads with framework-owned identifiers and transfer handles.
let prepared = spec.prepare_metadata_only(
    &model, &processor_config, Modality::Image, &payloads, max_tokens,
)?;
```

Supported image contracts (arrays describe one image):

| Model spec | Required JSON fields |
| --- | --- |
| Qwen VL / Qwen3 VL (including Qwen3.5) | `image_grid_thw: [t, h, w]` |
| Qwen3-Omni (including standalone Thinker), MiniMax-M3 | `image_grid_thw: [t, h, w]` |
| LLaVA | `num_image_tokens: [n]` |
| LLaVA-NeXT | `num_image_tokens: [n]`, `image_sizes: [height, width]` |
| Phi-3 Vision | `num_image_tokens: [n]`, `image_sizes: [HD_height, HD_width]` |
| Inkling | `num_image_tokens: [n]` |
| Llama4 | `aspect_ratios: [height_tiles, width_tiles]` |
| Kimi-K2.5 | `grid_thws: [t, h, w]` |
| Kimi-K3 | `grid_thws: [t, h, w]`, `image_sizes: [original_width, original_height]` |

Grids describe processed patches and require `t = 1`. Kimi-K3 dimensions are
the original dimensions used in its prompt text, not padded grid dimensions.
Its fixed-output-token processor configuration is honored. LLaVA token counts
must be the encoder's actual feature lengths after cropping/merging, not counts
guessed from raw image dimensions. Phi-3 uses its processed HD dimensions,
not the original image size. MiniMax's start/end tokens and Inkling's image
marker are preserved but excluded from the embedding count. Qwen3-Omni and
MiniMax use the preprocessor's merge size (default 2).

All image-capable specs currently registered by this library implement the
contract. Audio/video metadata-only inputs remain unsupported; existing raw
audio/video paths are unchanged. Unknown fields and unsupported modalities
are rejected. A standalone image processor without a registered model spec
(such as Pixtral or Phi-4) does not yet have this protocol.

The caller is responsible for request/media count limits, the complete
text-plus-media context length, matching producer configuration, and ensuring
the referenced embeddings arrive with the reported feature lengths. HTTP,
UUIDs and transfer protocols are deliberately outside this library.

These are library-level contracts, not a claim of end-to-end EPD support in
every consumer. In particular, producers must publish the listed metadata;
Kimi's Python `vision_chunk` mapping must be handled by the framework adapter.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
