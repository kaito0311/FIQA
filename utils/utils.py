import numpy as np 

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
        return lr
    return _lr_adjuster


class UtilMeasureTimeRunning:
    def __init__(self) -> None:
        self.diction_time = dict() 
    
    def __setitem__(self, name, time):
        if name not in self.diction_time.keys():
            self.diction_time[name] = 0 
        
        self.diction_time[name] += time 
        

    def __getitem__(self, name):
        return self.diction_time[name]
    
    