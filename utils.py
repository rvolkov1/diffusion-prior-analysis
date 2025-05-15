import torch
import math

_LOG_2PI = math.log(2 * math.pi)
_LOG_PI = math.log(math.pi)

def student_t_from_inv_gamma_nll(pred, target):
    # Parametrized in terms of inverse gamma dist on variance
    loc, alpha, beta = pred
    dof = 2 * alpha
    scale = torch.sqrt(beta / alpha)

    return student_t_nll(dof, loc, scale, target)


def student_t_nll(dof, loc, scale, target):
    # Adapted from torch.distributions.studentT
    y = (target - loc) / scale
    Z = (
        scale.log()
        + 0.5 * dof.log()
        + 0.5 * _LOG_PI
        + torch.lgamma(0.5 * dof)
        - torch.lgamma(0.5 * (dof + 1.0))
    )
    log_prob = -0.5 * (dof + 1.0) * torch.log1p(y ** 2.0 / dof) - Z

    return -torch.sum(log_prob, axis=-1)