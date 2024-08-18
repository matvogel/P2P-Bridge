from typing import Dict

import torch
from ema_pytorch import EMA
from loguru import logger
from torch import optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

from models.p2pb import P2PB
from models.unet_pvc import PVCNN2Unet


def load_optim_sched(cfg: Dict, model: torch.nn.Module, model_ckpt: str = None) -> tuple:
    """Load optimizer and scheduler according to the configuration. Loads weights from checkpoint if available.

    Args:
        cfg (Dict): Configuration dictionary.
        model (torch.nn.Module): Model to optimize.
        model_ckpt (str, optional): Path to checkpoint. Defaults to None.

    Raises:
        NotImplementedError: If the optimizer type is not implemented.

    Returns:
        tuple: Optimzer, Scheduler
    """

    # setup optimizer
    if cfg.training.optimizer.type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )
    elif cfg.training.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )
    else:
        raise NotImplementedError(cfg.training.optimizer.type)

    # setup lr scheduler
    if cfg.training.scheduler.type == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.training.scheduler.lr_gamma)
    elif cfg.training.scheduler.type == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.9)
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if model_ckpt is not None and not cfg.restart:
        try:
            optimizer.load_state_dict(model_ckpt["optimizer_state"])
        except Exception as e:
            logger.warning(e)
    logger.info("Optimizer and scheduler prepared")

    return optimizer, lr_scheduler


def load_model(cfg: Dict) -> torch.nn.Module:
    """
    Load a model based on the given configuration.

    Args:
        cfg (Dict): The configuration dictionary.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = PVCNN2Unet(cfg)
    logger.info(
        f"Generated model with following number of params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}"
    )
    return model


def load_diffusion(cfg: Dict) -> tuple:
    """Loads diffusion model including backbone and prepares it for training.

    Args:
        cfg (Dict): Configuration dictionary.

    Returns:
        tuple: Diffusion model, checkpoint
    """
    # setup model
    backbone = load_model(cfg).to(cfg.local_rank)
    model = P2PB(cfg=cfg, model=backbone)

    gpu = cfg.local_rank

    model = model.cuda()

    # setup DDP model
    if cfg.distribution_type == "multi":

        def ddp_transform(m):
            return DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)

        model.multi_gpu_wrapper(ddp_transform)

    # setup data parallel model
    elif cfg.distribution_type == "single":

        def dp_transform(m):
            return DataParallel(m)

        model.multi_gpu_wrapper(dp_transform)

    # load the model weights
    cfg.start_step = 0
    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path, map_location=torch.device("cpu"))

        if not cfg.restart:
            cfg.start_step = ckpt["step"] + 1

            try:
                model_state = ckpt["model_state"]

                if cfg.distribution_type in ["multi", "single"]:
                    model_dict = extract_from_state_dict(model_state, "model.")
                    ema_dict = extract_from_state_dict(model_state, "ema.")
                else:
                    model_dict = extract_from_state_dict(model_state, "model.module.")
                    ema_dict = extract_from_state_dict(model_state, "ema.")

                model.model.load_state_dict(model_dict)
                if cfg.use_ema:
                    if ema_dict != {}:
                        model.ema.load_state_dict(ema_dict)
                        logger.success("Loaded EMA from checkpoint!")
                logger.success("Loaded Model from checkpoint!")

            except RuntimeError as e:
                logger.warning("Could not load model state dict. Trying to load without strict flag.")
                logger.warning(e)
                model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            logger.info("Restarting training...")
            logger.info("Loading Model, generat new EMA from checkpoint and restart optimizer from scratch.")
            model_state = ckpt["model_state"]
            model_dict = extract_from_state_dict(model_state, "model.")
            try:
                # only load the model parameters and let rest start from scratch
                model.model.load_state_dict(model_dict)
                # set ema
                if cfg.use_ema:
                    model.ema = EMA(model.model, beta=0.999)
            except RuntimeError:
                logger.warning("Could not load model state dict. Trying to load adaptively.")
                load_matched_weights(model.model, model_dict)
                if cfg.use_ema:
                    model.ema = EMA(model.model, beta=0.999)
        logger.info("Loaded model from %s" % cfg.model_path)
    else:
        ckpt = None

    torch.cuda.empty_cache()
    return model, ckpt


def extract_from_state_dict(state_dict: Dict, pattern: str) -> Dict:
    """
    Extracts key-value pairs from a state dictionary based on a given pattern.

    Args:
        state_dict (Dict): The state dictionary to extract from.
        pattern (str): The pattern to match the keys against.

    Returns:
        Dict: A dictionary containing the matched key-value pairs.
    """
    matched_kv_pairs = {k.replace(pattern, ""): v for k, v in state_dict.items() if k.startswith(pattern)}
    return matched_kv_pairs


def load_matched_weights(model: torch.nn.Module, state_dict_to_load: Dict):
    """
    Loads matched weights from a state dictionary into a model.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        state_dict_to_load (Dict): The state dictionary containing the weights to load.

    Returns:
        None
    """
    own_state = model.state_dict()
    for name, param in state_dict_to_load.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
            except Exception as e:
                print(f"Failed to load parameter {name}. Exception: {e}")
        elif "." in name:
            sub_module_names = name.split(".")
            sub_module = model
            for sub_module_name in sub_module_names[:-1]:
                if hasattr(sub_module, sub_module_name):
                    sub_module = getattr(sub_module, sub_module_name)
                else:
                    break
            else:
                sub_param_name = sub_module_names[-1]
                if hasattr(sub_module, sub_param_name):
                    sub_param = getattr(sub_module, sub_param_name)
                    if sub_param.shape == param.shape:
                        sub_param.data.copy_(param)
        else:
            print(f"Parameter {name} not found in model. Skipping.")
