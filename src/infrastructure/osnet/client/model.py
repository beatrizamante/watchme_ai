import torchreid
from config import settings

def create_osnet_model(num_classes=None, pretrained=True):
    """
    Create an OSNet model using torchreid
    
    Args:
        num_classes: Number of identity classes in your dataset
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: OSNet model instance
    """
    model = torchreid.models.build_model(
        name='osnet_x1_0', 
        num_classes=num_classes or settings.OSNET_NUM_CLASSES,
        loss='triplet',  
        pretrained=pretrained
    )
    
    return model

def create_datamanager(dataset_name, data_dir):
    """
    Create a data manager for training/testing
    
    Args:
        dataset_name: Name of the dataset (e.g., 'market1501', 'dukemtmcreid')
        data_dir: Path to dataset directory
    
    Returns:
        datamanager: Torchreid data manager
    """
    datamanager = torchreid.data.ImageDataManager(
        root=data_dir,
        sources=dataset_name,
        targets=dataset_name,
        height=settings.OSNET_IMG_HEIGHT,
        width=settings.OSNET_IMG_WIDTH,
        batch_size_train=settings.OSNET_BATCH_SIZE,
        batch_size_test=settings.OSNET_BATCH_SIZE,
        transforms=['random_flip', 'random_crop'],
        num_instances=settings.OSNET_NUM_INSTANCES,
        train_sampler='RandomIdentitySampler'
    )
    
    return datamanager
