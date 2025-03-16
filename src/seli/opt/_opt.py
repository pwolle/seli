from collections.abc import Callable
from typing import Any, ParamSpec, Self, TypeVar

from jax import Array
from jaxtyping import Float

from seli.core._module import Module, NodeType
from seli.opt._grad import get_arrays, set_arrays, value_and_grad

P = ParamSpec("P")
T = TypeVar("T")
M = TypeVar("M", bound=NodeType)


def _return_model_and_loss(
    func: Callable[P, T],
) -> Callable[P, tuple[T, NodeType]]:
    def wrapped(model: NodeType, *args, **kwargs):
        result = func(model, *args, **kwargs)
        return result, model

    return wrapped


class Optimizer(Module, name="opt.Optimizer"):
    """
    Base class for all optimizers.
    """

    @property
    def collection(self) -> str | None:
        if not hasattr(self, "_collection"):
            return None

        return self._collection

    @collection.setter
    def collection(self, value: str | None):
        self._collection = value

    def minimize(
        self,
        loss_fn: Callable[P, Float[Array, ""]],
        model: M,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Self, M, Float[Array, ""]]:
        loss_fn = _return_model_and_loss(loss_fn)
        loss_fn = value_and_grad(
            loss_fn,
            collection=self.collection,
            has_aux=True,
        )

        (loss_value, model), grads = loss_fn(model, *args, **kwargs)
        _, arrays = get_arrays(model, self.collection)

        # subset of arrays that is used for gradient descent
        arrays_subset: dict[str, Array] = {}

        for key in grads.keys():
            if key not in arrays:
                error = f"Gradient for {key} not found in arrays"
                raise ValueError(error)

            arrays_subset[key] = arrays[key]

        # process gradients
        grads = self(loss=loss_value, grads=grads, values=arrays_subset)

        for key, grad in grads.items():
            # perform gradient descent with modified gradients
            arrays_subset[key] = arrays_subset[key] - grad

        # update model
        model = set_arrays(model, arrays_subset)
        return self, model, loss_value

    def call_param(self, grad: Float[Array, "*s"], **_) -> Float[Array, "*s"]:
        return grad

    def call_model(
        self,
        grads: dict[str, Float[Array, "..."]],
        **_,
    ) -> dict[str, Float[Array, "..."]]:
        return grads

    def __call__(
        self,
        loss: Float[Array, ""],
        grads: dict[str, Float[Array, "..."]],
        values: dict[str, Float[Array, "..."]],
    ) -> dict[str, Float[Array, "..."]]:
        grads = self.call_model(
            loss=loss,
            params=values,
            grads=grads,
        )

        for key, grad in grads.items():
            grads[key] = self.call_param(
                key=key,
                grad=grad,
                param=values[key],
            )

        return grads
