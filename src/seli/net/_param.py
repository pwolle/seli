# @typecheck
# class Param(Module):
#     def __init__(
#         self,
#         value: jax.Array,
#         /,
#         trainable: bool = True,
#         collections: list[str] | None = None,
#     ):
#         self.value = value
#         self.trainable = trainable
#         self._collections = collections or []

#     def __call__(self) -> jax.Array:
#         return self.value

#     def __repr__(self) -> str:
#         return f"Param({self.value})"

#     @property
#     def collections(self) -> list[str]:
#         collection = []

#         if self.trainable:
#             collection.append("trainable")

#         collection.extend(self._collections)
#         return collection


# @typecheck
# class Param(Module):
#     def __init__(self, inti):
#         pass
