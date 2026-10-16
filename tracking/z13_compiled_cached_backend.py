"""Separate compiled inference plus the verified immutable parameter cache."""
from tracking.z13_compiled_backend import setup, enable as compile_model
from tracking.z13_cached_params_backend import enable as cache_parameters


def install():
    from tracking.z13_spatial_localizer import SpatialLocalizer
    original = SpatialLocalizer.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.compiler_info = compile_model(self)
        # B returns before any tensor operation; C/D use guarded frozen params.
        self.parameter_cache_info = cache_parameters(self)

    SpatialLocalizer.__init__ = init
