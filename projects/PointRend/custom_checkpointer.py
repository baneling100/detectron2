import os
import pickle
import logging

import torch
import torch_xla.core.xla_model as xm

from fvcore.common.checkpoint import Checkpointer

from detectron2.utils.file_io import PathManager
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

__all__ = [
    "CustomDetectionCheckpointer",
]

class CustomDetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = xm.is_master_ordinal()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            xm.save(data, f)
        self.tag_last_checkpoint(basename)

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = torch.load(filename, map_location=torch.device("cpu")) # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible