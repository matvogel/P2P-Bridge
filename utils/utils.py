import torch
import torch.nn.init as init


def write_array_to_xyz(path, array):
    with open(path, "w") as f:
        fmt = " ".join(["%8f"] * array.shape[1])
        fmt = "\n".join([fmt] * array.shape[0])
        data = fmt % tuple(array.ravel())
        f.write(data)


def smart_load_model_weights(model, pretrained_dict):
    """
    Loads pretrained weights into a model's state dictionary, handling size mismatches if necessary.

    Args:
        model (nn.Module): The model to load the weights into.
        pretrained_dict (dict): A dictionary containing the pretrained weights.

    Returns:
        None
    """
    # Get the model's state dict
    model_dict = model.state_dict()

    # New state dict
    new_state_dict = {}
    device = model.device

    for name, param in model_dict.items():
        if name in pretrained_dict:
            # Load the pretrained weight
            pretrained_param = pretrained_dict[name]

            if param.size() == pretrained_param.size():
                # If sizes match, load the pretrained weights as is
                new_state_dict[name] = pretrained_param
            else:
                # Handle size mismatch
                # Resize pretrained_param to match the size of param
                reshaped_param = resize_weight(param.size(), pretrained_param, device=device, layer_name=name)
                new_state_dict[name] = reshaped_param
        else:
            # If no pretrained weight, use the model's original weights
            new_state_dict[name] = param

    # Update the model's state dict
    model.load_state_dict(new_state_dict)


def resize_weight(target_size, weight, layer_name="", device="cpu"):
    """
    Resize the weight tensor to the target size.
    Handles different layer types including attention layers.
    Uses Xavier or He initialization for new weights.
    Args:
        target_size: The desired size of the tensor.
        weight: The original weight tensor.
        layer_name: Name of the layer (used to determine initialization strategy).
        device: The target device ('cpu', 'cuda', etc.)
    """
    # Initialize the target tensor on the specified device
    target_tensor = torch.zeros(target_size, device=device)

    # Copy existing weights
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(target_size, weight.shape))
    slice_objects = tuple(slice(0, min_dim) for min_dim in min_shape)
    target_tensor[slice_objects] = weight[slice_objects].to(device)

    # Mask to identify new weights (those that are still zero)
    mask = (target_tensor == 0).type(torch.float32)

    # Initialize new weights
    if "attention" in layer_name or "conv" in layer_name:
        # He initialization for layers typically followed by ReLU
        new_weights = torch.empty(target_size, device=device)
        init.kaiming_uniform_(new_weights, a=0, mode="fan_in", nonlinearity="relu")
    else:
        # Xavier initialization for other layers
        new_weights = torch.empty(target_size, device=device)
        init.xavier_uniform_(new_weights, gain=init.calculate_gain("linear"))

    # Apply the initialization only to new weights
    target_tensor = target_tensor * (1 - mask) + new_weights * mask

    return target_tensor


# from https://github.com/luost26/score-denoise
class NormalizeUnitSphere(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2  # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl**2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert "pcl_noisy" not in data, "Point clouds must be normalized before applying noise perturbation."
        data["pcl_clean"], center, scale = self.normalize(data["pcl_clean"])
        data["center"] = center
        data["scale"] = scale
        return data
