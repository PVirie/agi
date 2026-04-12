import torch
import logging
import os
from datetime import datetime


class Safe_nn_Module:

    def __init__(self, name=None, device=None, persistence_path=None, modules=None, **kwargs):
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.device = device
        self.persistence_path = persistence_path
        # can save multiple modules with a dict of name: module
        self.modules = modules if modules is not None else {self.name: self}


    def save(self, num_to_keep=2, override_persistence_path=None):

        save_path = self.persistence_path
        if override_persistence_path is not None:
            logging.info(f"Saving to override path: {override_persistence_path}")
            save_path = override_persistence_path

        for name, module_to_save in self.modules.items():
            if save_path is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save({
                    "model_state_dict": module_to_save.state_dict(),
                }, f"{save_path}/{name}_{stamp}.pth")
                logging.info(f"Saved {name} parameters")

                # now delete oldest checkpoint to save space
                all_checkpoints = [f for f in os.listdir(save_path) if f.startswith(name) and f.endswith(".pth")]
                if len(all_checkpoints) > num_to_keep:
                    all_checkpoints.sort()
                    num_to_delete = len(all_checkpoints) - num_to_keep
                    for i in range(num_to_delete):
                        os.remove(os.path.join(save_path, all_checkpoints[i]))
                        # logging.info(f"Deleted old checkpoint: {all_checkpoints[i]}")


    def load(self, override_persistence_path=None):

        load_path = self.persistence_path
        if override_persistence_path is not None:
            logging.info(f"Loading from override path: {override_persistence_path}")
            load_path = override_persistence_path

        if load_path is None:
            return

        for name, module_to_save in self.modules.items():
            # first load all checkpoints and find the latest one
            all_checkpoints = [f for f in os.listdir(load_path) if f.startswith(name) and f.endswith(".pth")]
            if len(all_checkpoints) == 0:
                logging.info(f"No checkpoints found for {name} at {load_path}.")
                return
            
            all_checkpoints.sort()
            for checkpoint_file in reversed(all_checkpoints):
                checkpoint_path = os.path.join(load_path, checkpoint_file)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    module_to_save.load_state_dict(checkpoint["model_state_dict"])
                    logging.info(f"Loaded {name} parameters from {checkpoint_path}")
                    return
                except Exception as e:
                    logging.warning(f"Failed to load checkpoint {checkpoint_path} for {name}: {e}")
                    # retry with the next oldest checkpoint
                    continue
            else:
                logging.info(f"Failed to load any checkpoints for {name} from {load_path}.")
                
