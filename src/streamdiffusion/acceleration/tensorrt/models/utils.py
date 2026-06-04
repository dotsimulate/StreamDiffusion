from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


def get_kvo_cache_info(unet: UNet2DConditionModel, height=512, width=512):
    latent_height = height // 8
    latent_width = width // 8

    kvo_cache_shapes = []
    kvo_cache_structure = []
    current_h, current_w = latent_height, latent_width

    for _, block in enumerate(unet.down_blocks):
        if hasattr(block, "attentions") and block.attentions is not None:
            block_structure = []
            for attn_block in block.attentions:
                attn_count = 0
                for transformer in attn_block.transformer_blocks:
                    attn = transformer.attn1
                    hidden_dim = attn.to_k.out_features
                    seq_length = current_h * current_w
                    kvo_cache_shapes.append((seq_length, hidden_dim))
                    attn_count += 1
                block_structure.append(attn_count)
            kvo_cache_structure.append(block_structure)

        if hasattr(block, "downsamplers") and block.downsamplers is not None:
            current_h //= 2
            current_w //= 2

    if hasattr(unet.mid_block, "attentions") and unet.mid_block.attentions is not None:
        block_structure = []
        for attn_block in unet.mid_block.attentions:
            attn_count = 0
            for transformer in attn_block.transformer_blocks:
                attn = transformer.attn1
                hidden_dim = attn.to_k.out_features
                seq_length = current_h * current_w
                kvo_cache_shapes.append((seq_length, hidden_dim))
                attn_count += 1
            block_structure.append(attn_count)
        kvo_cache_structure.append(block_structure)

    for _, block in enumerate(unet.up_blocks):
        if hasattr(block, "attentions") and block.attentions is not None:
            block_structure = []
            for attn_block in block.attentions:
                attn_count = 0
                for transformer in attn_block.transformer_blocks:
                    attn = transformer.attn1
                    hidden_dim = attn.to_k.out_features
                    seq_length = current_h * current_w
                    kvo_cache_shapes.append((seq_length, hidden_dim))
                    attn_count += 1
                block_structure.append(attn_count)
            kvo_cache_structure.append(block_structure)

        if hasattr(block, "upsamplers") and block.upsamplers is not None:
            current_h *= 2
            current_w *= 2

    kvo_cache_count = sum(sum(block) for block in kvo_cache_structure)

    return kvo_cache_shapes, kvo_cache_structure, kvo_cache_count


def convert_list_to_structure(flat_list, structure):
    formatted_list = []
    flat_idx = 0
    for block_structure in structure:
        block_list = []
        for count in block_structure:
            layer_list = []
            for _ in range(count):
                if flat_idx >= len(flat_list):
                    break
                layer_list.append(flat_list[flat_idx])
                flat_idx += 1
            block_list.append(layer_list)
        formatted_list.append(block_list)
    return formatted_list


def convert_structure_to_list(structured_list):
    flat_list = []
    for block_list in structured_list:
        for layer_list in block_list:
            for item in layer_list:
                flat_list.append(item)
    return flat_list


def create_kvo_cache(
    unet: UNet2DConditionModel, batch_size, cache_maxframes, height=512, width=512, device="cuda", dtype=torch.float16
):
    kvo_cache_shapes, kvo_cache_structure, _ = get_kvo_cache_info(unet, height, width)

    bucket_keys: List[Tuple[int, int]] = []
    key_to_idx: Dict[Tuple[int, int], int] = {}
    layer_to_bucket: List[Tuple[int, int]] = []
    outputs_by_bucket: List[List[int]] = []
    for layer_idx, (s, h) in enumerate(kvo_cache_shapes):
        b = key_to_idx.get((s, h))
        if b is None:
            b = len(bucket_keys)
            key_to_idx[(s, h)] = b
            bucket_keys.append((s, h))
            outputs_by_bucket.append([])
        slot = len(outputs_by_bucket[b])
        layer_to_bucket.append((b, slot))
        outputs_by_bucket[b].append(layer_idx)

    # layers_in_bucket is the OUTERMOST dim so bucket[layer_slot] is stride-identical
    # to a standalone (2, maxframes, B, S, H) tensor — TRT's contiguous-input
    # requirement is satisfied without an extra .contiguous() call.
    buckets = [
        torch.zeros(len(outputs_by_bucket[b]), 2, cache_maxframes, batch_size, s, h, dtype=dtype, device=device)
        for b, (s, h) in enumerate(bucket_keys)
    ]
    per_layer_views = [buckets[b][slot] for (b, slot) in layer_to_bucket]

    return per_layer_views, kvo_cache_structure, buckets, outputs_by_bucket


def get_fi_eligible_mask(
    unet: UNet2DConditionModel,
    height: int = 512,
    width: int = 512,
    max_fi_up_blocks: int = 2,
) -> List[bool]:
    """Return a bool mask (length = kvo_cache_count) marking FI-eligible self-attn layers.

    Per the StreamV2V thesis (§3.4.2, Appendix B.2): Feature Fusion degrades quality
    when applied to high-resolution features because averaging details across frames
    causes blur.  Apply only to the mid_block and the first ``max_fi_up_blocks``
    up-blocks (thesis ablation default: 2, i.e. "mid + up01").  Down-blocks and the
    final (highest-resolution) up-block are always excluded.

    Walk order is identical to get_kvo_cache_info so indices are aligned 1-to-1.
    """
    mask: List[bool] = []

    # down_blocks — never eligible; high-res features would blow up cosine-matrix memory
    for block in unet.down_blocks:
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for _ in attn_block.transformer_blocks:
                    mask.append(False)

    # mid_block — always eligible (lowest resolution in the network)
    if hasattr(unet.mid_block, "attentions") and unet.mid_block.attentions is not None:
        for attn_block in unet.mid_block.attentions:
            for _ in attn_block.transformer_blocks:
                mask.append(True)

    # up_blocks — eligible only for the first max_fi_up_blocks blocks
    for up_idx, block in enumerate(unet.up_blocks):
        eligible = up_idx < max_fi_up_blocks
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for _ in attn_block.transformer_blocks:
                    mask.append(eligible)

    return mask


def create_fi_cache(
    unet: UNet2DConditionModel,
    batch_size: int,
    cache_maxframes: int,
    fi_eligible_mask: Optional[List[bool]] = None,
    height: int = 512,
    width: int = 512,
    max_fi_up_blocks: int = 2,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Allocate the Feature-Injection output cache for FI-eligible self-attn layers.

    Only allocates tensors for layers where fi_eligible_mask is True.  Shape follows
    the same bucketing scheme as create_kvo_cache but stores only block **output**
    (no K/V dimension):

        per_fi_layer_views[i] : Tensor(cache_maxframes, batch_size, seq_len, hidden_dim)

    Returns
    -------
    per_fi_layer_views : List[Tensor]
        One cache view per FI-eligible layer, in kvo walk order.
    fi_layer_indices : List[int]
        Global kvo-layer index for each FI layer — used to align engine binding names
        (``fio_cache_in_<global_idx>`` / ``fio_cache_out_<global_idx>``).
    fi_buckets : List[Tensor]
        Contiguous backing tensors (one per unique (seq, hidden) pair among eligible
        layers).  layers_in_bucket is the outermost dim — same TRT contiguity guarantee
        as create_kvo_cache.
    fi_outputs_by_bucket : List[List[int]]
        Maps bucket index → list of fi-local layer indices inside that bucket.
    """
    kvo_cache_shapes, _, _ = get_kvo_cache_info(unet, height, width)

    if fi_eligible_mask is None:
        fi_eligible_mask = get_fi_eligible_mask(unet, height, width, max_fi_up_blocks)

    assert len(fi_eligible_mask) == len(kvo_cache_shapes), (
        f"fi_eligible_mask length {len(fi_eligible_mask)} != kvo layer count {len(kvo_cache_shapes)}"
    )

    # Collect eligible layers and their (seq, hidden) shapes, preserving walk order
    fi_layer_indices: List[int] = []
    fi_layer_shapes: List[Tuple[int, int]] = []
    for global_idx, (shape, eligible) in enumerate(zip(kvo_cache_shapes, fi_eligible_mask)):
        if eligible:
            fi_layer_indices.append(global_idx)
            fi_layer_shapes.append(shape)

    # Bucket by (seq, hidden) — same pattern as create_kvo_cache
    bucket_keys: List[Tuple[int, int]] = []
    key_to_idx: Dict[Tuple[int, int], int] = {}
    fi_local_to_bucket: List[Tuple[int, int]] = []
    fi_outputs_by_bucket: List[List[int]] = []

    for fi_local_idx, (s, h) in enumerate(fi_layer_shapes):
        b = key_to_idx.get((s, h))
        if b is None:
            b = len(bucket_keys)
            key_to_idx[(s, h)] = b
            bucket_keys.append((s, h))
            fi_outputs_by_bucket.append([])
        slot = len(fi_outputs_by_bucket[b])
        fi_local_to_bucket.append((b, slot))
        fi_outputs_by_bucket[b].append(fi_local_idx)

    # Shape: (fi_count_in_bucket, cache_maxframes, B, S, H)
    # No "2" dim — storing output only, not K+V.
    # layers_in_bucket is OUTERMOST for the same TRT stride-identity guarantee as kvo_cache.
    fi_buckets = [
        torch.zeros(
            len(fi_outputs_by_bucket[b]),
            cache_maxframes,
            batch_size,
            s,
            h,
            dtype=dtype,
            device=device,
        )
        for b, (s, h) in enumerate(bucket_keys)
    ]
    per_fi_layer_views = [fi_buckets[b][slot] for (b, slot) in fi_local_to_bucket]

    return per_fi_layer_views, fi_layer_indices, fi_buckets, fi_outputs_by_bucket
