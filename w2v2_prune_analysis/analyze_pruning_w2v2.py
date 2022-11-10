import pandas as pd
import torch
import json

import transformers

from omegaconf import DictConfig

from torch import nn
from torch.nn.utils import prune

from src.networks.wav2vec2.w2v2_speech import (
    Wav2vec2ForSpeechRecognition,
    Wav2vec2ForSpeechRecognitionConfig,
)
from src.networks.wav2vec2.w2v2_speaker import (
    Wav2vec2ForSpeakerRecognition,
    Wav2vec2ForSpeakerRecognitionConfig,
)
from src.networks.heads import LinearHeadConfig, LinearProjectionHeadConfig
from src.optim.loss import CtcLoss


transformers.logging.set_verbosity(transformers.logging.CRITICAL)


def collect_masks(model):
    mask_dict = {}

    for name, buffer in model.named_buffers():
        if "mask" in name:
            mask_dict[name] = buffer

    return mask_dict


def compare_masks(model, other_model, layer_idx: int = None):
    masks = collect_masks(model)
    other_masks = collect_masks(other_model)

    total_masked_valued = 0
    total_overlap = 0

    with torch.no_grad():
        for name, mask in masks.items():
            if layer_idx is not None and str(layer_idx) not in name:
                continue

            other_mask = other_masks[name]

            values = torch.numel(mask)
            masked_values = torch.sum(mask == 0).item()
            assert masked_values == torch.sum(other_mask == 0).item()

            summed_mask = mask + other_mask
            count_0 = torch.sum(summed_mask == 0).item()
            count_1 = torch.sum(summed_mask == 1).item()
            count_2 = torch.sum(summed_mask == 2).item()
            assert count_0 + count_1 + count_2 == values

            overlap_count = count_0

            total_masked_valued += masked_values
            total_overlap += overlap_count

    return total_overlap / total_masked_valued


def prune_model(model, factor: float):
    for name, module in model.named_modules():
        if "encoder" not in name:
            continue

        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, "weight", factor)


def load_asr_model():
    asr_ckpt = "/home/nik/phd/repo/disjoint_mtl/w2v2_prune_analysis/tmp/asr/impressed-cops-2/disjoint-mtl-speech/3vl35lnv/checkpoints/epoch_0002.step_000320000.val-wer_0.0440.best.ckpt"

    with open(
        "/home/nik/phd/repo/disjoint_mtl/data/librispeech/meta/character_vocabulary.json",
        "r",
    ) as f:
        idx_to_char = json.load(f)["idx_to_char"]

    asr_model = Wav2vec2ForSpeechRecognition.load_from_checkpoint(
        asr_ckpt,
        root_hydra_config=DictConfig({}),
        loss_fn_constructor=lambda: CtcLoss(),
        idx_to_char=idx_to_char,
        test_names=["ls960h"],
        cfg=Wav2vec2ForSpeechRecognitionConfig(
            huggingface_id="facebook/wav2vec2-base",
            reset_weights=False,
            use_gradient_checkpointing=True,
            freeze_cnn=True,
            freeze_transformer=True,
            num_steps_freeze_cnn=-1,
            num_steps_freeze_transformer=-1,
            head_cfg=LinearHeadConfig(
                blank_token_idx=0,
                blank_initial_bias=10,
            ),
        ),
    )

    return asr_model


def load_sv_model():
    sv_ckpt = "/home/nik/phd/repo/disjoint_mtl/w2v2_prune_analysis/tmp/sv/miserable-boots-0/disjoint-mtl-speaker/1up0of62/checkpoints/epoch_0003.step_000050000.val-loss_1.6412.best.ckpt"

    sv_model = Wav2vec2ForSpeakerRecognition.load_from_checkpoint(
        sv_ckpt,
        root_hydra_config=DictConfig({}),
        loss_fn_constructor=lambda: nn.CrossEntropyLoss(),
        num_speakers=5790,
        test_pairs=[],
        test_names=[],
        cfg=Wav2vec2ForSpeakerRecognitionConfig(
            huggingface_id="facebook/wav2vec2-base",
            reset_weights=False,
            use_gradient_checkpointing=True,
            freeze_cnn=True,
            freeze_transformer=True,
            num_steps_freeze_cnn=-1,
            num_steps_freeze_transformer=-1,
            head_cfg=LinearProjectionHeadConfig(
                use_projection_layer=True,
                projection_layer_dim=128,
                drop_prob=0.05,
                use_cosine_linear=True,
                enable_train_chunk=False,
                train_random_chunk_size=40,
            ),
        ),
    )

    return sv_model


def main():
    layer_wise = True
    prune_rate = []
    overlap_rate = []
    layer_label = []

    for i in range(1, 101, 1):
        print(i)
        asr_model = load_asr_model()
        sv_model = load_sv_model()

        factor = i / 100
        prune_model(asr_model, factor)
        prune_model(sv_model, factor)

        if layer_wise:
            for layer_idx in range(0, 12):
                overlap = compare_masks(asr_model, sv_model, layer_idx)

                prune_rate.append(factor)
                overlap_rate.append(overlap)
                layer_label.append(str(layer_idx))
        else:
            overlap = compare_masks(asr_model, sv_model)

            prune_rate.append(factor)
            overlap_rate.append(overlap)
            layer_label.append('all')

    import seaborn as sns
    import matplotlib.pyplot as plt

    df_plot = pd.DataFrame(
        {"Pruning rate": prune_rate, "Overlap rate": overlap_rate, "label": layer_label}
    )

    print(df_plot)

    sns.lineplot(data=df_plot, x="Pruning rate", y="Overlap rate", hue="label").set(
        title="Overlap between ASR and SV wav2vec2 model",
    )
    plt.savefig("asr_vs_sv___per_layer.png")


if __name__ == "__main__":
    main()
