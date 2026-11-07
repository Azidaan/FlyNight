"""CPU: согласовать глобальное устройство и две копии, сохранённые при импорте."""
def load_cpu_branches(checkpoint):
    import torch
    from tracking.flyvis_adapter import activate_flyvis
    assert not torch.cuda.is_initialized()
    flyvis=activate_flyvis()
    import flyvis.network.initialization as initialization
    import flyvis.task.decoder as decoder
    # Эти модули используют `from flyvis import device`: одна замена flyvis.device
    # не обновляет уже импортированные ссылки генератора и декодера.
    for module in (flyvis,initialization,decoder):module.device=torch.device('cpu')
    torch.set_default_device('cpu')
    from tracking.b2_pilot_runtime import load_branches
    models=load_branches(checkpoint,'cpu')
    for name,(branch,backbone) in models.items():
        assert all(p.device.type=='cpu' for p in branch.parameters()),name
        if hasattr(backbone,'net'):assert all(p.device.type=='cpu' for p in backbone.net.parameters()),name
        elif hasattr(backbone,'parameters'):assert all(p.device.type=='cpu' for p in backbone.parameters()),name
    assert not torch.cuda.is_initialized()
    return models
