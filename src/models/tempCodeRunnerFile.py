train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE, 
        is_train=True
    )