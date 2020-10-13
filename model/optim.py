#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import math

import torch

logger = logging.getLogger(__file__)


class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.
    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))
                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    if group['weight_decay'] != 0:
                        p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

    def backward(self, losses):
        with torch.autograd.set_detect_anomaly(True):
            if not isinstance(losses, (tuple, list)):
                losses = [losses]
            full_loss = sum(losses, 0)
            full_loss.backward()
            return full_loss

class NoamOpt:
    def __init__(self, embeddings_size, warmup, optimizer, linear_schedule=False, lr=None, total_steps=None,
                 apex_level=None, loss_weight=None, extra_module_lr_rate=1.0):
        self.embeddings_size = embeddings_size
        self.warmup = warmup
        self.optimizer = optimizer
        self.linear_schedule = linear_schedule
        self.apex_level = apex_level
        self.lr = lr
        self.total_steps = total_steps
        self.loss_weight = loss_weight
        self.extra_module_lr_rate = extra_module_lr_rate

        self._step = 0
        
    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except ValueError as e:
            logger.info("Optimizer cannot be loaded from checkpoint: {}".format(e))
        except KeyError as e:
            logger.info("Optimizer cannot be loaded from checkpoint: {}".format(e))

    def backward(self, losses):
        if not isinstance(losses, (tuple, list)):
            losses = [losses]
        if self.loss_weight is None:
            full_loss = sum(losses, 0)
        else:
            full_loss = torch.sum(torch.stack(losses, 0) * torch.exp(self.loss_weight[1])) + torch.sum(self.loss_weight[1])

        if self.apex_level is not None:
            try:
                from apex.amp import scale_loss
            except ImportError:
                raise ImportError("Please install apex.")

            for loss_id, loss in enumerate(losses):
                with scale_loss(loss, self.optimizer, loss_id=loss_id) as scaled_loss:
                    scaled_loss.backward()
        else:
            full_loss.backward()
        return full_loss

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate_linear() if self.linear_schedule else self.rate()
        for p in self.optimizer.param_groups:
            if p.__contains__('extra'):
                p['lr'] = rate * self.extra_module_lr_rate
            else:
                p['lr'] = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
            
        return self.lr * (self.embeddings_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x

    def rate_linear(self, step=None):
        if step is None:
            step = self._step
        assert self.lr is not None and self.total_steps is not None

        return self.lr * self.warmup_linear(step/self.total_steps, self.warmup)
