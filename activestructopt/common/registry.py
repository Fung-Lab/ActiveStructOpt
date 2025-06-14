"""
Borrowed from MatDeepLearn, 
    which borrowed this from https://github.com/Open-Catalyst-Project.

Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.

Import the global registry object using

``from activestructopt.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a model: ``@registry.register_model``
"""
import importlib
from pathlib import Path

# Copied from 
#     https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports():
    root_folder = Path(__file__).parent
    while root_folder.stem != 'activestructopt':
        root_folder = root_folder.parent
    project_root = root_folder.parent.resolve().absolute()

    import_keys = ["sampler", "dataset", "model", "objective", "optimizer", 
        "simulation"]
    for key in import_keys:
        dir_list = (project_root / "activestructopt" / key).rglob("*.py")
        for f in dir_list:
            module_name = ".".join(f.resolve().absolute().relative_to(
                project_root).with_suffix("").parts)
            importlib.import_module(module_name)

class Registry:
    r"""Class for registry object which acts as central source of truth."""
    mapping = {
        # Mappings to respective classes.
        "sampler_name_mapping": {},
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "optimizer_name_mapping": {},
        "objective_name_mapping": {},
        "simulation_name_mapping": {},
    }

    @classmethod
    def register_sampler(cls, name):
        def wrap(func):
            cls.mapping["sampler_name_mapping"][name] = func
            return func
        return wrap
    
    @classmethod
    def register_dataset(cls, name):
        def wrap(func):
            cls.mapping["dataset_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_energy(cls, name):
        def wrap(func):
            cls.mapping["energy_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_optimizer(cls, name):
        def wrap(func):
            cls.mapping["optimizer_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_objective(cls, name):
        def wrap(func):
            cls.mapping["objective_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def register_simulation(cls, name):
        def wrap(func):
            cls.mapping["simulation_name_mapping"][name] = func
            return func
        return wrap

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        return cls.mapping[mapping_name].get(name, None)

    @classmethod
    def get_sampler_class(cls, name):
        return cls.get_class(name, "sampler_name_mapping")
    
    @classmethod
    def get_dataset_class(cls, name):
        return cls.get_class(name, "dataset_name_mapping")

    @classmethod
    def get_model_class(cls, name):
        return cls.get_class(name, "model_name_mapping")

    @classmethod
    def get_optimizer_class(cls, name):
        return cls.get_class(name, "optimizer_name_mapping")

    @classmethod
    def get_objective_class(cls, name):
        return cls.get_class(name, "objective_name_mapping")

    @classmethod
    def get_simulation_class(cls, name):
        return cls.get_class(name, "simulation_name_mapping")

registry = Registry()
