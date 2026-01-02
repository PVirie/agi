import torch
import logging
import os
from datetime import datetime


class Safe_nn_Module:

    def __init__(self, name=None, device=None, persistence_path=None, module=None, **kwargs):
        super(Safe_nn_Module, self).__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.device = device
        self.persistence_path = persistence_path
        self.module = module


    def save(self, num_to_keep=2):
        if self.module is not None:
            module_to_save = self.module
        else:
            module_to_save = self
        if self.persistence_path is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save({
                "model_state_dict": module_to_save.state_dict(),
            }, f"{self.persistence_path}/{self.name}_{stamp}.pth")
            logging.info(f"Saved {self.name} parameters")

            # now delete oldest checkpoint to save space
            all_checkpoints = [f for f in os.listdir(self.persistence_path) if f.startswith(self.name) and f.endswith(".pth")]
            if len(all_checkpoints) > num_to_keep:
                all_checkpoints.sort()
                num_to_delete = len(all_checkpoints) - num_to_keep
                for i in range(num_to_delete):
                    os.remove(os.path.join(self.persistence_path, all_checkpoints[i]))
                    # logging.info(f"Deleted old checkpoint: {all_checkpoints[i]}")


    def load(self):
        if self.module is not None:
            module_to_save = self.module
        else:
            module_to_save = self
        if self.persistence_path is not None:

            # first load all checkpoints and find the latest one
            all_checkpoints = [f for f in os.listdir(self.persistence_path) if f.startswith(self.name) and f.endswith(".pth")]
            if len(all_checkpoints) == 0:
                logging.info(f"No checkpoints found for {self.name} at {self.persistence_path}.")
                return
            
            all_checkpoints.sort()
            for checkpoint_file in reversed(all_checkpoints):
                checkpoint_path = os.path.join(self.persistence_path, checkpoint_file)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    module_to_save.load_state_dict(checkpoint["model_state_dict"])
                    logging.info(f"Loaded {self.name} parameters from {checkpoint_path}")
                    return
                except Exception as e:
                    logging.warning(f"Failed to load checkpoint {checkpoint_path} for {self.name}: {e}")
                    # retry with the next oldest checkpoint
                    continue
            else:
                logging.info(f"Failed to load any checkpoints for {self.name} from {self.persistence_path}.")
                
