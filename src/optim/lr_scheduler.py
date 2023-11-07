# pylint: disable=too-many-arguments,too-few-public-methods,too-many-instance-attributes,line-too-long
import math
import torch


class LRSchedulerConfig():
    def __init__(
            self,
            lr_schedule: str,
            n_steps: int,
            lr_pretrained: float,
            lr: float,
            lr_warmup: int,
            lr_warmup_scale: float,
            lr_decay_degree: float
    ):
        self.lr_schedule = lr_schedule
        self.n_steps = n_steps
        self.lr_pretrained = lr_pretrained
        self.lr = lr
        if lr_warmup >= 0:
            self.lr_warmup = lr_warmup
        else:
            assert 0. <= lr_warmup_scale <= 1.
            self.lr_warmup = int(lr_warmup_scale * self.n_steps)
        self.lr_warmup_scale = lr_warmup_scale
        self.lr_decay_degree = lr_decay_degree

    def setup(self, optimizer: torch.optim.Optimizer):
        return CustomLRScheduler(
            optimizer=optimizer,
            mode=self.lr_schedule,
            base_lr=self.lr,
            num_iters=self.n_steps,
            warmup_iters=self.lr_warmup,
            decay_degree=self.lr_decay_degree
        )


class CustomLRScheduler():
    """
    Custom learning rate scheduler for pytorch optimizer.
    Assumes 1 <= self.iter <= 1 + num_iters.
    """
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            mode: str,
            base_lr: float,
            num_iters: int,
            warmup_iters: int=1000,
            from_iter: int=0,
            decay_degree: float=0.9,
            decay_steps: int=5000
        ):
        self.optimizer = optimizer
        self.mode = mode
        self.base_lr = base_lr
        self.lr = base_lr
        self.iter = from_iter
        self.N = num_iters + 1
        self.warmup_iters = warmup_iters
        self.decay_degree = decay_degree
        self.decay_steps = decay_steps

        self.lr_coefs = []
        for param_group in optimizer.param_groups:
            self.lr_coefs.append(param_group['lr'] / base_lr)

        if mode == 'cos':
            self._lr_schedule = self._lr_schedule_cos
        elif mode == 'linear':
            self._lr_schedule = self._lr_schedule_linear
        elif mode == 'poly':
            self._lr_schedule = self._lr_schedule_poly
        elif mode == 'step':
            self._lr_schedule = self._lr_schedule_step
        elif mode == 'constant':
            self._lr_schedule = self._lr_schedule_constant
        elif mode == 'sqroot':
            self._lr_schedule = self._lr_schedule_sqroot
        else:
            raise ValueError(f'Unknown mode {mode}!')

    def _lr_schedule_cos(self):
        if self.warmup_iters < self.iter < self.N:
            self.lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * (self.iter - self.warmup_iters) / (self.N - self.warmup_iters) * math.pi))

    def _lr_schedule_linear(self):
        if self.warmup_iters < self.iter < self.N:
            self.lr = self.base_lr * (1 - 1.0 * (self.iter - self.warmup_iters) / (self.N - self.warmup_iters))

    def _lr_schedule_poly(self):
        if self.warmup_iters < self.iter < self.N:
            self.lr = self.base_lr * pow((1 - 1.0 * (self.iter - self.warmup_iters) / (self.N - self.warmup_iters)), self.decay_degree)

    def _lr_schedule_step(self):
        if self.warmup_iters < self.iter:
            self.lr = self.base_lr * (0.1 ** (self.decay_steps // (self.iter - self.warmup_iters)))

    def _lr_schedule_constant(self):
        self.lr = self.base_lr

    def _lr_schedule_sqroot(self):
        self.lr = self.base_lr * self.warmup_iters**0.5 * min(self.iter * self.warmup_iters**-1.5, self.iter**-0.5)

    def _lr_warmup(self):
        if self.warmup_iters > 0 and self.iter < self.warmup_iters and self.mode != 'sqroot':
            self.lr = self.base_lr * 1.0 * self.iter / self.warmup_iters

    def _adjust_learning_rate(self, optimizer, lr):
        assert lr >= 0
        for i, _ in enumerate(optimizer.param_groups):
            optimizer.param_groups[i]['lr'] = lr * self.lr_coefs[i]

    def step(self, step=-1):
        """Update current step"""
        self.iter = step if step >= 0 else self.iter + 1
        self._lr_schedule()
        self._lr_warmup()
        self._adjust_learning_rate(self.optimizer, self.lr)

    def reset(self):
        self.lr = self.base_lr
        self.iter = 0
        self._adjust_learning_rate(self.optimizer, self.lr)
