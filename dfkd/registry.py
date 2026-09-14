class Registry(dict):
    """Explicit plugin registration; duplicate and missing names fail early."""

    def register(self, name):
        def decorator(factory):
            if name in self:
                raise KeyError(f"Already registered: {name}")
            self[name] = factory
            return factory

        return decorator

    def resolve(self, name):
        if name not in self:
            raise ValueError(f"Unknown {name!r}; choose from {sorted(self)}")
        return self[name]
