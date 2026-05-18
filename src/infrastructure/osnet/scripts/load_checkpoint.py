import torch
import torch.backends.cudnn

def load_checkpoint(weights_path, device, model):
    """
    Load a pre-trained model checkpoint, skipping classifier/fc layers.
    Consistency between restarts is guaranteed by loading the same checkpoint + model.eval().
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    _, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    model.eval()
    model = model.to(device)

    print("Checkpoint loaded successfully")
    return model
