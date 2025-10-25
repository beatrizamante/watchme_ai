from src.infrastructure.osnet.scripts.shared.extract_from_loader import extract_from_loader

def extract_features(model, device, datamanager):
    """
    Extract features for video ReID with nested dict structure.
    """
    print("Extracting features from test set...")
    model.eval()

    test_loader = datamanager.test_loader
    dataset_name = list(test_loader.keys())[0]

    query_loader = test_loader[dataset_name]['query']
    gallery_loader = test_loader[dataset_name]['gallery']

    query_features, query_pids, query_camids = extract_from_loader(model, device, query_loader, "Query")
    gallery_features, gallery_pids, gallery_camids = extract_from_loader(model, device, gallery_loader, "Gallery")

    print(f"\nFinal results:")
    print(f"Query: {query_features.shape}, PIDs: {len(query_pids)}, CamIDs: {len(query_camids)}")
    print(f"Gallery: {gallery_features.shape}, PIDs: {len(gallery_pids)}, CamIDs: {len(gallery_camids)}")

    return (query_features, query_pids, query_camids,
            gallery_features, gallery_pids, gallery_camids)
