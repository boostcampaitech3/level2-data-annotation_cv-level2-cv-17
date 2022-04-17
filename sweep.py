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
            name='valid_metric/hmean',  # anything you are logging on wandb
            goal='maximize'
        )
    )
    # you can control logging parameter
    # Ex. value : not log information on wandb sweep
    #     values : log information on wandb sweep 
    # optm : 'adam' | 'sgd'
    # schd : 'multisteplr' | 'reducelr' | 'cosignlr'
    if sweep_cfg['method'] == 'grid':
        sweep_cfg['parameters'] = dict(
            image_size=dict(values=[1024]),
            input_size=dict(values=[512]),
            batch_size=dict(values=[32]),
            learning_rate=dict(values=[1e-4, 1e-3]),
            max_epoch=dict(values=[100]),
            optm=dict(values=['adam']),
            schd=dict(values=['multisteplr'])
        )
    elif sweep_cfg['method'] == 'bayes':
        sweep_cfg['parameters'] = dict(
            image_size=dict(distribution='categorical', values=[1024]),
            input_size=dict(distribution='categorical', values=[512]),
            batch_size=dict(distribution='categorical', values=[32]),
            learning_rate=dict(distribution='uniform', min=1e-4, max=1e-3),
            max_epoch=dict(distribution='categorical', values=[100]),
            optm=dict(distribution='categorical', values=['sgd']),
            schd=dict(distribution='categorical', values=['cosignlr'])
        )
    elif sweep_cfg['method'] == 'random':
        sweep_cfg['parameters'] = dict(
            image_size=dict(distribution='categorical', values=[1024]),
            input_size=dict(distribution='categorical', values=[512]),
            batch_size=dict(distribution='categorical', values=[32]),
            learning_rate=dict(distribution='uniform', min=1e-4, max=1e-3),
            max_epoch=dict(distribution='categorical', values=[100]),
            optm=dict(distribution='categorical', values=['adam']),
            schd=dict(distribution='categorical', values=['reducelr'])
        )
    return sweep_cfg
