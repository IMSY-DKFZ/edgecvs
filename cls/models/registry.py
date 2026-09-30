# models/registry.py
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type, Any, Callable


def _autodiscover_models():
    """
    Recursively import every module under models/ so @ModelRegistry.register
    decorators fire. Subpackages are imported via their __init__.py which
    re-exports public symbols — no per-file path tracking needed.
    """
    models_dir = Path(__file__).parent
    root_package = models_dir.name  # "models"

    _skip_modules = {"registry"}

    def _walk(directory: Path, package: str):
        for module_info in pkgutil.iter_modules([str(directory)]):
            name = module_info.name
            if name.startswith("_"):
                continue
            if module_info.ispkg:
                # Import the subpackage — its __init__.py triggers all re-exports
                try:
                    importlib.import_module(f"{package}.{name}")
                except Exception:
                    pass
                _walk(directory / name, f"{package}.{name}")
            else:
                if name in _skip_modules:
                    continue
                try:
                    importlib.import_module(f"{package}.{name}")
                except Exception:
                    pass

    _walk(models_dir, root_package)


def _lazy_import_vmodel(vmodel):
    module_by_vmodel = {
        "SegFormerAuxSeg": "models.mtl.segformer_auxseg",
    }
    module_name = module_by_vmodel.get(vmodel)
    if module_name:
        importlib.import_module(module_name)


class ModelRegistry:
    _registry: Dict[str, Type[Any]] = {}
    _matchers: Dict[str, Callable[[Any], bool]] = {}
    _build_msgs: Dict[str, Callable[[Any], str]] = {}

    @classmethod
    def register(cls, name: str, matcher: Callable[[Any], bool] = None, build_msg: Callable[[Any], str] = None):
        """
        Decorator to register a model class.
        
        Args:
            name: Unique identifier for the model.
            matcher: Optional function (CFG -> bool) to determine if this model should be selected.
            build_msg: Optional function (CFG -> str) to generate the custom print message on build.
        """
        def decorator(model_class):
            if name in cls._registry:
                if cls._registry[name] is not model_class:
                    raise ValueError(f"Model '{name}' is already registered by a different class.")
                return model_class  # same class re-imported — idempotent
            cls._registry[name] = model_class
            
            if matcher:
                cls._matchers[name] = matcher
            if build_msg:
                cls._build_msgs[name] = build_msg
                
            return model_class
        return decorator

    @classmethod
    def get_model_class(cls, name: str) -> Type[Any]:
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def determine_model_from_cfg(cls, CFG) -> str:
        # 1. Explicit model type provided
        if hasattr(CFG, 'model_type') and CFG.model_type in cls._registry:
            return CFG.model_type

        # 2. Check registered matchers
        for name, matcher in cls._matchers.items():
            if matcher(CFG):
                return name

        # 3. If vmodel is explicit but autodiscovery missed it, import its
        #    module directly. This avoids circular-import timing problems
        #    during package-wide discovery.
        vmodel = getattr(CFG, 'vmodel', None)
        lazy_import_error = None
        if vmodel:
            try:
                _lazy_import_vmodel(vmodel)
            except Exception as e:
                lazy_import_error = e

            if vmodel in cls._registry:
                return vmodel

            for name, matcher in cls._matchers.items():
                if matcher(CFG):
                    return name

        # 4. If vmodel is explicitly set but no matcher fired, the model
        #    likely failed to import (missing optional dependency). Fail loudly
        #    rather than silently falling back to CLSModel.
        if vmodel and vmodel != "CLS":
            extra = f" Import error: {lazy_import_error}" if lazy_import_error else ""
            raise RuntimeError(
                f"vmodel='{vmodel}' is set in config but was not registered. "
                f"The model likely failed to import due to a missing dependency. "
                f"Registered models: {list(cls._registry.keys())}."
                f"{extra}"
            )

        # 4. Legacy fallback: configs saved before the vmodel convention use
        #    aux_seg: true to signal a segmentation model.
        if CFG.get("aux_seg"):
            return "ASPPAuxSeg"

        # 5. Default fallback
        return "CLSModel"
        
    @classmethod
    def get_build_msg(cls, name: str, CFG) -> str:
        if name in cls._build_msgs:
            return cls._build_msgs[name](CFG)
        return f"Load {name} model"


_autodiscover_models()
