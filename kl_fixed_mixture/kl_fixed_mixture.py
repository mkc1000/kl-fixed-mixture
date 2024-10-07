import torch

def logx_and_kl(loga, logb, alpha):
    logx = torch.logaddexp(loga + torch.log(alpha), logb + torch.log(1-alpha))
    kl_i = torch.exp(logx) * (logx - logb)
    return logx, torch.sum(kl_i, dim=-1, keepdims=True)

def dkl_dalpha(loga, logb, logx, alpha):
    a_minus_b = torch.exp(loga) - torch.exp(logb)
    dkl_dalpha = torch.sum((a_minus_b * (logx - logb + 1)), dim=-1, keepdims=True)
    return dkl_dalpha

def findalphafork(loga, logb, k, n_steps=20, verbose=False):
    """Identify a value of lambd such that, with x= alpha*a + (1-alpha)*b,
    KL(x || b) = k, or get as close as possible if k is too big
    [Batched]
    a : (n, d)
    logb : (n, d)
    k : (n, 1)
    """
    alpha_top = torch.ones_like(k)
    alpha_bot = torch.zeros_like(k)
    # x = alpha * torch.exp(a) + (1-alpha) * torch.exp(logb)
    alpha = torch.ones_like(k)
    for i in range(n_steps):
        logx, kl = logx_and_kl(loga, logb, alpha)
        alpha_bot[(k >= kl).view(-1)] = alpha[(k >= kl).view(-1)]
        alpha_top[(k <= kl).view(-1)] = alpha[(k <= kl).view(-1)]
        dkl_dalp = dkl_dalpha(loga, logb, logx, alpha)
        aim = k - kl
        delta = aim / (dkl_dalp + 1e-12)
        alpha = torch.clamp(alpha+delta, min=alpha_bot*15/16+alpha_top*1/16, max=alpha_bot*1/2+alpha_top*1/2)
        if verbose:
            print("kl", kl)
            print("alpha", alpha)
    return alpha

def a_minus_b_over_x(loga, logb, alpha):
    mean = (loga + logb) / 2
    loga_, logb_ = loga - mean, logb - mean
    a, b = torch.exp(loga_), torch.exp(logb_)
    x = alpha * a + (1-alpha) * b
    # out = (a - b) / x
    # out[torch.isnan(out)] = 1.0
    return (a - b) / x

def amiss(tensor):
    return torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor))

class KLFixedMixture(torch.autograd.Function):
    """
    Return x = alpha * a + (1-alpha) * b minimizing [KL(x || b) - k]**2
    Autograd is implemented for a and k, but not b
    Squared error between k and k_achieved should be added to any loss function
    so that k can be moved to a region where changing it affects the value of x
    """
    @staticmethod
    def forward(ctx, k, loga, logb):
        dim = loga.dim()
        if dim == 1:
            loga = loga.unsqueeze(0)
            logb = logb.unsqueeze(0)
            k = k.unsqueeze(0)
        alpha = findalphafork(loga, logb, k)
        logx, kl_achieved = logx_and_kl(loga, logb, alpha)
        ctx.save_for_backward(loga, logb, logx, k, alpha)
        if dim == 1:
            logx = logx.squeeze(0)
            kl_achieved = kl_achieved.squeeze(0)
        return logx, kl_achieved

    @staticmethod
    def backward(ctx, grad_wrt_logx, grad_wrt_kl_achieved, debug=False):
        dim = grad_wrt_logx.dim()
        if dim == 1:
            grad_wrt_logx = grad_wrt_logx.unsqueeze(0)
        loga, logb, logx, k, alpha = ctx.saved_tensors
        a, b, x = torch.exp(loga), torch.exp(logb), torch.exp(logx)
        dkl_dalp = dkl_dalpha(loga, logb, logx, alpha)
        dalp_dk = 1 / (dkl_dalp + 1e-12)
        # dx_dk = (a - b) * dalp_dk
        # dlogx_dx = 1/x
        a_minus_b_over_x_ = a_minus_b_over_x(loga, logb, alpha)
        dlogx_dk = dalp_dk * a_minus_b_over_x_
        grad_wrt_k = torch.sum(grad_wrt_logx * dlogx_dk, dim=-1, keepdim=True)
        dkl_da = alpha * (1 + logx - logb)
        dalp_da = - dalp_dk * dkl_da
        grad_wrt_a_part = torch.sum(grad_wrt_logx * a_minus_b_over_x_, dim=-1, keepdim=True) * dalp_da
        # grad_wrt_a = grad_wrt_a_part + (alpha / x * grad_wrt_logx)
        # grad_wrt_loga = grad_wrt_a * a
        grad_wrt_loga = a * grad_wrt_a_part + (alpha * grad_wrt_logx) * torch.exp(loga - logx)
        # different form if alpha >= 1
        grad_wrt_k[alpha.squeeze(-1) == 1] = 0
        grad_wrt_loga[alpha.squeeze(-1) == 1] = grad_wrt_logx[alpha.squeeze(-1) == 1]
        if dim == 1:
            grad_wrt_k = grad_wrt_k.squeeze(0)
            grad_wrt_loga = grad_wrt_loga.squeeze(0)
        if debug:
            if amiss(grad_wrt_loga) or amiss(grad_wrt_k):
                for var_name, var_value in locals().items():
                    print(f"{var_name}: {var_value}")
                total_mask = torch.logical_or(torch.isnan(grad_wrt_loga), torch.isinf(grad_wrt_loga))
                row_mask = torch.any(total_mask, dim=-1)
                col_mask = torch.any(total_mask, dim=-2)
                for var_name, var_value in locals().items():
                    if torch.is_tensor(var_value):
                        try:
                            print(f"{var_name} selected: {var_value[row_mask][:, col_mask]}")
                        except:
                            pass
                        try:
                            print(f"{var_name} selected: {var_value[0][:, row_mask]}")
                        except:
                            pass
                        try:
                            print(f"{var_name} selected whole row: {var_value[row_mask]}")
                        except:
                            pass
                    else:
                        print(var_name, str(type(var_value)))
                row_mask = torch.logical_or(torch.isnan(grad_wrt_k), torch.isinf(grad_wrt_k))
                for var_name, var_value in locals().items():
                    if torch.is_tensor(var_value):
                        try:
                            print(f"{var_name} selected (k error): {var_value[row_mask]}")
                        except:
                            pass
                    else:
                        print(var_name, str(type(var_value)))
        return grad_wrt_k, grad_wrt_loga, None
