"""
This folder contains the core functionality on which the rest of the package
is built.

The core functionality is split into three parts:

- `_module.py`: Defines the Module class and related utilities for tree-based
  operations on module hierarchies.
- `_registry.py`: Provides a system for registering modules. This is useful for
  serializing and deserializing modules without having to know the module
  structure beforehand.
- `_serialize.py`: Provides functionality for serializing and deserializing
  modules.
- `_typecheck.py`: Provides a function for checking types based on the type
  hints of the function arguments. This saves a lot of boilerplate tests,
  since the type hints will automatically be checked.
"""
