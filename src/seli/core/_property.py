import weakref

__all__ = [
    "cached_property",
]


class cached_property(property):
    """
    A property that caches the result of a function.

    The cache is stored as a weak reference to the instance, so it will be
    automatically removed when the instance is garbage collected. This also
    makes the cached property work with immutable instances, and be invisible
    to the module system.
    """

    def __init__(self, func):
        super().__init__(func)
        self.func = func
        self._cache = weakref.WeakKeyDictionary()

    def __get__(self, instance, _):
        if instance is None:
            return self

        if instance not in self._cache:
            self._cache[instance] = self.func(instance)

        return self._cache[instance]
