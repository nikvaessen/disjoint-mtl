########################################################################################
#
# This run script encapsulates the training and evaluation of a speaker
# recognition model defined by the hydra configuration.
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
    random_experiment_id,
)

################################################################################
# set custom resolvers

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)
OmegaConf.register_new_resolver("random_name", random_experiment_id)


################################################################################
# wrap around main hydra script


@hydra.main(config_path="config", config_name="train_speaker", version_base="1.2")
def run(cfg: DictConfig):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from src.main import run_train_eval_script

    return run_train_eval_script(cfg)


################################################################################
# execute hydra application

if __name__ == "__main__":
    load_dotenv()

    env_var = os.environ

    if "SLURM_ARRAY_TASK_ID" in env_var:
        job_id = int(env_var["SLURM_ARRAY_TASK_ID"])
        sleep_sec = 10 * int(job_id)
        print(f"detected slurm array job: sleeping for {sleep_sec} sec")
        time.sleep(sleep_sec)

    run()
