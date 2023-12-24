from natten.functional import NeighborhoodAttention2DQKAutogradFunction, NeighborhoodAttention2DAVAutogradFunction

class Natten2DQKAutogradFn(NeighborhoodAttention2DQKAutogradFunction): ...
class Natten2DAVAutogradFn(NeighborhoodAttention2DAVAutogradFunction): ...


def natten2dqk(query, key, kernel_size, dilation):
    return Natten2DQKAutogradFn.apply(
        query, key, None, kernel_size, dilation
    )

def natten2dav(attn, value, kernel_size, dilation):
    return Natten2DAVAutogradFn.apply(
        attn, value, kernel_size, dilation
    )