import torch

def load_checkpoint(weights_path, device, model):
    """
    Load a pre-trained model checkpoint and prepare it for inference.
    Args:
        weights_path (str): Path to the checkpoint file containing the model weights.
        device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
        model (torch.nn.Module): The model instance to load the weights into.
    Returns:
        torch.nn.Module: The model with loaded weights, moved to the specified device
                        and set to evaluation mode.
    Note:
        This function assumes the checkpoint is in Torchreid format with weights
        stored under the 'state_dict' key. Modify accordingly for other formats.
        Uses strict=False when loading state dict to allow partial loading.
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith('classifier') and
                     not k.endswith('running_mean') and
                     not k.endswith('running_var')}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
