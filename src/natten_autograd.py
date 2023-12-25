import torch.autograd.forward_ad as fwAD
from natten.functional import NeighborhoodAttention2DQKAutogradFunction, NeighborhoodAttention2DAVAutogradFunction
from torch.cuda.amp import custom_fwd, custom_bwd

try:
    from natten import _C
except ImportError:
    raise ImportError(
        f"Failed to import NATTEN's CPP backend. "
        + f"This could be due to an invalid/incomplete install. "
        + f"Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        f" correct torch build: "
        + f"shi-labs.com/natten"
    )

class Natten2DQKAutogradFn(NeighborhoodAttention2DQKAutogradFunction):
    @staticmethod
    def jvp(ctx, dsim_dq_p, dsim_dq_t, dsim_dk_p, dsim_dk_t, dsim_drpb_p, dsim_drpb_t, _dsim_dksz, _dsim_ddil):
        q_p, q_t, k_p, k_t = ctx.to_save
        return _C.na2d_qk_forward(q_t, k_p, None, ctx.kernel_size, ctx.dilation) + _C.na2d_qk_forward(q_p, k_t, None, ctx.kernel_size, ctx.dilation)

    @staticmethod
    @custom_fwd
    def forward(ctx, q_p, q_t, k_p, k_t, rpb_p, rpb_t, kernel_size, dilation):
        q_p = q_p.contiguous()
        if q_t is not None:
            q_t = q_t.contiguous()
        k_p = k_p.contiguous()
        if k_t is not None:
            k_t = k_t.contiguous()
        if rpb_p is not None:
            assert q_t is None and k_t is None and rpb_t is None, "rpb not supported for forward-mode autodiff"
            rpb_p = rpb_p.to(k_p.dtype)
        attn = _C.na2d_qk_forward(q_p, k_p, rpb_p, kernel_size, dilation)
        # TODO: does this give tangents a longer lifetime than necessary? should we store them on ctx instead?
        ctx.save_for_backward(q_p, q_t, k_p, k_t)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.bias = rpb_p is not None
        return attn

    # the only change here is that I changed the positions of the tensors in the saved_tensors list
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        q_p, _, k_p, _, _ = ctx.saved_tensors
        outputs = _C.na2d_qk_backward(
            grad_out.contiguous(),
            q_p,
            k_p,
            ctx.bias,
            ctx.kernel_size,
            ctx.dilation,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None

class Natten2DAVAutogradFn(NeighborhoodAttention2DAVAutogradFunction):
    @staticmethod
    def jvp(ctx, da_dp_p, da_dp_t, da_dv_p, da_dv_t, _da_dksz, _da_ddil):
        p_p, p_t, v_p, v_t = ctx.to_save
        return _C.na2d_av_forward(p_t, v_p, ctx.kernel_size, ctx.dilation) + _C.na2d_av_forward(p_p, v_t, ctx.kernel_size, ctx.dilation)

    @staticmethod
    @custom_fwd
    def forward(ctx, p_p, p_t, v_p, v_t, kernel_size, dilation):
        p_p = p_p.contiguous().to(v_p.dtype)
        if p_t is not None:
            p_t = p_t.contiguous().to(v_p.dtype)
        v_p = v_p.contiguous()
        if v_t is not None:
            v_t = v_t.contiguous()
        out = _C.na2d_av_forward(p_p, v_p, kernel_size, dilation)
        # TODO: does this give tangents a longer lifetime than necessary? should we store them on ctx instead?
        ctx.save_for_backward(p_p, p_t, v_p, v_t)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    # the only change here is that I changed the positions of the tensors in the saved_tensors list
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        p_p, _, v_p, _, _ = ctx.saved_tensors
        outputs = _C.na2d_av_backward(
            grad_out.contiguous(),
            p_p,
            v_p,
            ctx.kernel_size,
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None


def natten2dqk(query, key, kernel_size, dilation):
    q_t = fwAD.unpack_dual(query).tangent
    k_t = fwAD.unpack_dual(key).tangent
    return Natten2DQKAutogradFn.apply(
        query, q_t,
        key, k_t,
        None, None,
        kernel_size,
        dilation,
    )

# I've kept the "attn" param name for kwarg compatibility, but I think it is not attention but is actually a precursor, the attention *probabilities*.
# attention is what you get after you matmul the probabilities with V. I will use the initial p for this concept, and a for the output.
def natten2dav(attn, value, kernel_size, dilation):
    p_t = fwAD.unpack_dual(attn).tangent
    v_t = fwAD.unpack_dual(value).tangent
    return Natten2DAVAutogradFn.apply(
        attn, p_t,
        value, v_t,
        kernel_size,
        dilation,
    )