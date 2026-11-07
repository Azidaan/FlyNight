"""Явный CPU, включая сохранённые при импорте ссылки устройств FlyVis."""
def load_cpu_branches(checkpoint):
    import importlib
    import torch
    from tracking.flyvis_adapter import activate_flyvis
    assert not torch.cuda.is_initialized()
    flyvis=activate_flyvis()
    # flyvis.network экспортируется также как атрибут другого модуля;
    # import_module однозначно выбирает полное имя нужного модуля.
    initialization=importlib.import_module('flyvis.network.initialization')
    decoder=importlib.import_module('flyvis.task.decoder')
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
