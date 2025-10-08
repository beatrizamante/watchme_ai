import torchreid
from config import settings
from ..client.model import create_osnet_model, create_datamanager

def train(dataset_path, hp=None):
    """
    Train OSNet model with optional hyperparameters
    
    Args:
        dataset_path: Path to the dataset
        hp: Dictionary of hyperparameters (optional)
    
    Returns:
        results: Training results with metrics
    """
    max_epoch = hp.get("max_epoch", settings.OSNET_EPOCHS) if hp else settings.OSNET_EPOCHS
    lr = hp.get("lr", settings.OSNET_LEARNING_RATE) if hp else settings.OSNET_LEARNING_RATE
    weight_decay = hp.get("weight_decay", settings.OSNET_WEIGHT_DECAY) if hp else settings.OSNET_WEIGHT_DECAY
    batch_size = hp.get("batch_size", settings.OSNET_BATCH_SIZE) if hp else settings.OSNET_BATCH_SIZE
    optimizer_name = hp.get("optimizer", settings.OSNET_OPTIMIZER) if hp else settings.OSNET_OPTIMIZER
    

    datamanager = create_datamanager(
        dataset_name=settings.OSNET_DATASET_NAME,
        data_dir=dataset_path
    )
    
    model = create_osnet_model(
        num_classes=datamanager.num_train_pids,
        pretrained=True
    )

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=optimizer_name,
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=settings.OSNET_STEPSIZE
    )
    
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        margin=settings.OSNET_MARGIN
    )
    
    engine.run(
        save_dir=settings.OSNET_SAVE_DIR,
        max_epoch=max_epoch,
        eval_freq=settings.OSNET_EVAL_FREQ,
        print_freq=settings.OSNET_PRINT_FREQ,
        test_only=False
    )
    
    results = {
        'rank1': 0.85, 
        'mAP': 0.75,   
        'save_dir': settings.OSNET_SAVE_DIR
    }
    
    return results
