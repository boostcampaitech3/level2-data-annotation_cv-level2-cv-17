def update_args(args, wandb_cfg):
    # wandb_cfg changes with every experiment
    # so, it must be updated to args
    sweep_cfg = get_sweep_cfg()  # for getting parameter name
    for key in sweep_cfg['parameters'].keys():
        vars(args)[f"{key}"] = wandb_cfg[f"{key}"]
    return args


def get_sweep_cfg():
    # you must read this : https://docs.wandb.ai/guides/sweeps/configuration
    sweep_cfg = dict(
        name='sweep',           # I recommend to change this everytime
        method='bayes',         # 'grid' or 'bayes' or 'random'
        metric=dict(
            name='valid/loss',  # anything you are logging on wandb
            goal='minimize'
        )
    )
    if sweep_cfg['method'] == 'grid':
        sweep_cfg['parameters'] = dict(
            image_size=dict(value=1024),
            input_size=dict(value=512),
            batch_size=dict(values=[16, 32]),
            learning_rate=dict(values=[1e-4, 1e-3]),
            max_epoch=dict(value=1),
        )
    elif sweep_cfg['method'] == 'bayes':
        # 'bayes' use distribution
        sweep_cfg['parameters'] = dict(
            image_size=dict(distribution='constant', value=1024),
            input_size=dict(distribution='constant', value=512),
            batch_size=dict(distribution='categorical', values=[16, 32]),
            learning_rate=dict(distribution='uniform', min=1e-4, max=1e-3),
            max_epoch=dict(distribution='constant', value=1),
        )
    elif sweep_cfg['method'] == 'random':
        # 'random' use distribution & custom probabilities
        sweep_cfg['parameters'] = dict(
            image_size=dict(distribution='constant', value=1024),
            input_size=dict(distribution='constant', value=512),
            batch_size=dict(distribution='categorical', values=[16, 32]),
            learning_rate=dict(distribution='uniform', min=1e-4, max=1e-3),
            max_epoch=dict(values=[1, 2], probabilities=[0.7, 0.3]),
        )
    return sweep_cfg
