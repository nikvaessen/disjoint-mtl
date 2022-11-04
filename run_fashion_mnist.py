########################################################################################
#
# This run script encapsulates the training and evaluation of the fashion+mnist
# experiments.
#
# Author(s): Nik Vaessen
########################################################################################

import os
import time
import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.util.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)

################################################################################
# set custom resolvers

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)


################################################################################
# wrap around main hydra script


@hydra.main(config_path="fashion_and_mnist/config", config_name="main", version_base="1.2")
def run(cfg: DictConfig):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from fashion_and_mnist.experiment import main_from_cfg

    return main_from_cfg(cfg)


################################################################################
# execute hydra application

if __name__ == "__main__":
    load_dotenv()

    env_var = os.environ

    if "SLURM_ARRAY_TASK_ID" in env_var:
        import random
        job_id = int(env_var["SLURM_ARRAY_TASK_ID"])
        random.seed(job_id)
        sleep_sec = random.randint(1, 10) / 5000
        print(f"detected slurm array job: sleeping for {sleep_sec} sec")
        time.sleep(sleep_sec)

    run()
