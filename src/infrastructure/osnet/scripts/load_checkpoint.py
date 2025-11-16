import torch
import torch.backends.cudnn
import numpy as np

def load_checkpoint(weights_path, device, model):
    """
    Load a pre-trained model checkpoint with deterministic initialization for missing keys.
    Ensures consistent embeddings across server restarts.
    """

    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

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

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys (initializing deterministically): {missing_keys}")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        for name, param in model.named_parameters():
            if name in missing_keys:
                with torch.no_grad():
                    if 'weight' in name:
                        if len(param.shape) >= 2:
                            torch.nn.init.xavier_uniform_(param, gain=1.0)
                        else:
                            torch.nn.init.constant_(param, 1.0)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param, 0.0)
                    print(f"🔧 Deterministically initialized: {name} with shape {param.shape}")

        for name, buffer in model.named_buffers():
            if name in missing_keys:
                with torch.no_grad():
                    if 'running_mean' in name or 'running_var' in name:
                        if 'running_mean' in name:
                            torch.nn.init.constant_(buffer, 0.0)
                        else:  # running_var
                            torch.nn.init.constant_(buffer, 1.0)
                        print(f"Deterministically initialized buffer: {name}")

    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    model.eval()
    model = model.to(device)

    print(" Checkpoint loaded with deterministic initialization")
    return model
