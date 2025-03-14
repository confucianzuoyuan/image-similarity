import numpy as np
import os
import torch
import random


def seed_everything(seed):
    """
    为了保证训练过程可复现，使用确定的随机数种子。对 torch，numpy 和 random 都使用相同的种子。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """计算并存储平均值和当前值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """等待一段时间之后，如果验证集的损失还没有下降，则提前结束训练（Early stops）。"""

    def __init__(self, patience=7, verbose=False, delta=0.0001, path="checkpoint.pt"):
        """
        参数:
            - patience (int): 从上一次验证损失下降后需要等待的次数，默认值：7
            - verbose (bool): 如果设置为 True, 每次验证损失下降时，将打印一条信息，默认值：False
            - delta (float): 模型改进的标准，也就是损失下降多少算是改进，默认值：0
            - path (str): 检查点的保存路径。默认值: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"早停计数器（EarlyStopping counter）: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """当验证集的损失减小时，保存模型。"""
        if self.verbose:
            print(
                f"验证集的损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型 ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
