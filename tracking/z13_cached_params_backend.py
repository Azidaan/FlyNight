"""Separate inference-only cache; never changes the ongoing compiled backend."""


def enable(localizer):
    if localizer.arm == 'B':
        return {'applied': False, 'reason': 'B contains no FlyVis parameters'}
    import torch
    net = localizer.bb.net
    parameters = tuple(net.parameters())
    assert all(not p.requires_grad for p in parameters), 'Cache is inference-only'
    original = net._param_api
    with torch.no_grad():
        cached = original()
        reference = original()
        assert all(torch.equal(a, b) for a, b in (
            (cached.nodes.bias, reference.nodes.bias),
            (cached.nodes.time_const, reference.nodes.time_const),
            (cached.edges.weight, reference.edges.weight)))
    versions = tuple(p._version for p in parameters)
    def get_params():
        if tuple(p._version for p in parameters) != versions:
            raise RuntimeError('Model weights changed after immutable cache was created')
        if any(p.requires_grad for p in parameters):
            raise RuntimeError('Gradient-enabled model cannot reuse inference cache')
        return cached
    net._param_api = get_params
    return {'applied': True, 'parameter_preparation': 'once after checkpoint load',
            'weight_version_guard': True}
