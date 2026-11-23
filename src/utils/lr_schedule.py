import math 

def adjust_learning_rate(param_group:dict , base_lr:float, epoch_progress:float, args)-> float:
    """ Decay the learning rate after half cycle cosine after linear wrap """
    min_lr = 5e-6
    if epoch_progress < args.warmup_epochs and args.warmup_epochs > 0:
        lr = base_lr * epoch_progress / args.warmup_epochs
    else :
        denom = max(args.num_epochs - args.warmup_epochs, 1e-8)
        progress = min(max((epoch_progress - args.warmup_epochs) / denom, 0.0), 1.0)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    param_group["lr"] = lr
    return (lr)
