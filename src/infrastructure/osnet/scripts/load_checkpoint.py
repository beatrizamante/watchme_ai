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
    state_dict = checkpoint.get('state_dict', checkpoint)

    model_param_names = set(name for name, _ in model.named_parameters())
    model_buffer_names = set(name for name, _ in model.named_buffers())
    model_all_names = model_param_names | model_buffer_names

    filtered_state_dict = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key.startswith('classifier') or key.startswith('fc'):
            skipped_keys.append(key)
            continue

        if key in model_all_names:
            filtered_state_dict[key] = value
        else:
            skipped_keys.append(key)
            print(f"Skipping incompatible key: {key}")

    print(f"Kept {len(filtered_state_dict)} keys, skipped {len(skipped_keys)} keys")

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys:
        print(f"⚠️  Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {unexpected_keys}")

    model.eval()
    model = model.to(device)

    print("✅ Checkpoint loaded successfully")
    return model
