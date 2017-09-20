# Python interfaces for TA1 primitives

A collection of Python interfaces for TA1 primitives. They all extend the unified base class
exposing a common set of methods. The purposes of non-base interfaces are:

* they can declare stricter or specialized input and output types
* they can signal to TA2 systems which primitives are doing what on a different level
  than just through input and output types
* other primitives can declare that they are accepting as an argument just a subset
  of primitives using that interface class as a type of the argument
* specialized interfaces can declare additional methods on those primitives (discouraged
  because it can introduce special casing in TA2 systems, but until typing system is
  well developed this can help address some limitations)
* can be used a workaround where a input and output types are not enough

Every working group is invited to create merge requests with interface
for their sets of primitives. The idea is that we have a common interfaces in
Python so that it is easier to understand how interfaces should work because
currently each interface is described in a different way.

[Primitives Annotation Schema](https://datadrivendiscovery.org/wiki/display/gov/Primitives+Annotation+Schema)
defines `interface_type` field. The idea is that each of those values `<interface>` map to
a `primitive_interfaces.<interface>` Python module, which exposes one abstract class, the interface for that
interface.

See [supervised_learning.py](https://gitlab.datadrivendiscovery.org/berkeley/primitives-interfaces/blob/unified_interface/primitive_interfaces/supervised_learning.py)
for an example of an interface.

## Installation

You can run
```
pip install .
```
in the root directory of this repository to install the `primitive_interfaces` package.

You can also install from git directly
```
pip install git+ssh://git@gitlab.datadrivendiscovery.org/d3m/primitive-interfaces.git@unified_interface
```

Alternatively, you can run
```
pip install -e .
```
to install the package in [editable mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode), which means that you can keep editing
your local copy of the source code without the need to reinstall after each edit.

## Examples

Examples of primitives using the unified interface can be found
[in this repository](https://gitlab.datadrivendiscovery.org/d3m/primitive-examples).