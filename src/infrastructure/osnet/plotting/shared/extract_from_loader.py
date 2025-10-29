import torch

def extract_from_loader(model, device, loader, loader_name):
        """Helper function to extract features from a loader."""
        features_list = []
        pids_list = []
        camids_list = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                if isinstance(data, dict):
                    imgs = data['img'] if 'img' in data else data['imgs']
                    pids = data['pid'] if 'pid' in data else data['pids']
                    camids = data['camid'] if 'camid' in data else data['camids']
                else:
                    imgs, pids, camids = data[:3]

                if isinstance(imgs, torch.Tensor):
                    if imgs.dim() == 5:
                        batch_size, seq_len, c, h, w = imgs.shape
                        imgs = imgs.view(batch_size * seq_len, c, h, w)
                        imgs = imgs.to(device)

                        features = model(imgs)

                        feature_dim = features.shape[-1]
                        features = features.view(batch_size, seq_len, feature_dim)
                        features = features.mean(dim=1)
                    else:
                        imgs = imgs.to(device)
                        features = model(imgs)

                    if features.dim() == 1:
                        features = features.unsqueeze(0)

                    features_list.append(features.cpu())

                    if isinstance(pids, torch.Tensor):
                        pids_list.extend(pids.tolist())
                        camids_list.extend(camids.tolist())
                    else:
                        pids_list.extend(pids)
                        camids_list.extend(camids)

                if batch_idx % 50 == 0:
                    print(f"{loader_name}: Processed batch {batch_idx}/{len(loader)}")

        features_tensor = torch.cat(features_list, dim=0) if features_list else torch.empty(0)
        print(f"{loader_name} features: {features_tensor.shape}")

        return features_tensor, pids_list, camids_list
