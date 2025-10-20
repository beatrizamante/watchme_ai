import torch
import torchreid


def extract_features(weights_path, device, datamanager):
    """
    Extract features for all test images.

    Returns:
        tuple: (query_features, query_pids, query_camids,
               gallery_features, gallery_pids, gallery_camids)
    """
    print("Extracting features from test set...")

    feature_extractor = torchreid.utils.feature_extractor(
        model_name='osnet_x1_0',
        model_path=str(weights_path),
        device=device
    )

    test_loader = datamanager.test_loader

    query_features = []
    query_pids = []
    query_camids = []
    gallery_features = []
    gallery_pids = []
    gallery_camids = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            imgs, pids, camids = data[:3]

            features = feature_extractor(imgs)

            if hasattr(datamanager, 'num_query'):
                num_query = datamanager.num_query
                if batch_idx * test_loader.batch_size < num_query:
                    query_features.append(features.cpu())
                    query_pids.extend(pids.tolist())
                    query_camids.extend(camids.tolist())
                else:
                    gallery_features.append(features.cpu())
                    gallery_pids.extend(pids.tolist())
                    gallery_camids.extend(camids.tolist())
            else:
                gallery_features.append(features.cpu())
                gallery_pids.extend(pids.tolist())
                gallery_camids.extend(camids.tolist())

                if batch_idx < len(test_loader) * 0.2:
                    query_features.append(features.cpu())
                    query_pids.extend(pids.tolist())
                    query_camids.extend(camids.tolist())

            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")

    query_features = torch.cat(query_features, dim=0) if query_features else torch.empty(0)
    gallery_features = torch.cat(gallery_features, dim=0) if gallery_features else torch.empty(0)

    print(f"Extracted features: Query={query_features.shape}, Gallery={gallery_features.shape}")

    return (query_features, query_pids, query_camids,
            gallery_features, gallery_pids, gallery_camids)
