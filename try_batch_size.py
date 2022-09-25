########################################################################################
#
# Use a fake batch to determine the maximum size without CUDA vram issues.
#
# Author(s): Nik Vaessen
########################################################################################

import hydra
import torch

from dotenv import load_dotenv

from omegaconf import DictConfig, OmegaConf

from src.util.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
    random_experiment_id,
)

################################################################################
# set custom resolvers

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)
OmegaConf.register_new_resolver("random_name", random_experiment_id)

########################################################################################
# fake batch generation settings

max_tokens = 2_800_000
speech_max_frames = 400_000
batch_size = max_tokens // speech_max_frames

gt_tokens = 400

########################################################################################
# fake train loop


@hydra.main(config_path="config", config_name="train_speech", version_base="1.2")
def run(cfg: DictConfig):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from src.main import construct_network_module, construct_data_module

    network = construct_network_module(cfg, construct_data_module(cfg))
    optim = torch.optim.Adam(params=network.parameters(), lr=1e-4)

    audio_tensor = torch.rand((batch_size, speech_max_frames))
    length = [speech_max_frames for _ in range(batch_size)]

    gt_tensor = torch.randint(0, 40, size=(batch_size, max_tokens))
    gt_lengths = [max_tokens for _ in range(batch_size)]

    print(f"{audio_tensor.shape=}")

    network = network.to("cuda")
    audio_tensor = audio_tensor.to("cuda")
    gt_tensor = gt_tensor.to("cuda")

    for i in range(10):
        optim.zero_grad()

        with torch.autocast("cuda"):
            _, (pred, pred_length) = network(audio_tensor, length)
            loss = network.loss_fn(
                predictions=pred,
                ground_truths=gt_tensor,
                prediction_lengths=pred_length,
                ground_truth_lengths=gt_lengths,
            )

        loss.backward()
        optim.step()
        print(loss)


if __name__ == "__main__":
    load_dotenv()
    run()
