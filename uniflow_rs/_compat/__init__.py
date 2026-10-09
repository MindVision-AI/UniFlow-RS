"""Internal numerical runtime for the released UniFlow-RS checkpoint."""

from importlib import import_module

__all__ = [
    "load_flux2_autoencoder",
    "load_flux2_klein_base_for_uniflow_rs",
    "load_qwen3_embedder",
]

_EXPORTS = {
    name: ("uniflow_rs._compat.flux2_compat", name)
    for name in __all__
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
