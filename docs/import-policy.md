# Import policy

`malchan` is still in early development, so the package does not keep a separate compatibility namespace for the old `machine_learning` import path.

When an old import such as `machine_learning...` is found, replace it with `malchan...` or a package-relative import in the owning module.
