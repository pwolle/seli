import logging
from collections.abc import Callable
from typing import Any, ParamSpec, Self, TypeVar

from jax import Array
from jaxtyping import Float

from seli.core._jit import jit
from seli.core._module import Module, NodeType
from seli.core._typecheck import typecheck
from seli.opt._grad import get_arrays, set_arrays, value_and_grad
from seli.opt._loss import Loss

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")
M = TypeVar("M", bound=NodeType)


@typecheck
class Optimizer(Module, name="opt.Optimizer"):
    """
    Base class for all optimizers.
    """

    def minimize(
        self,
        loss_fn: Loss,
        model: M,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Self, M, Float[Array, ""]]:
        return _minimize_jit(self, loss_fn, model, *args, **kwargs)

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
        model: NodeType,
        loss: Float[Array, ""],
        grads: dict[str, Float[Array, "..."]],
        values: dict[str, Float[Array, "..."]],
    ) -> dict[str, Float[Array, "..."]]:
        grads = self.call_model(
            model=model,
            loss=loss,
            params=values,
            grads=grads,
        )

        for key, grad in grads.items():
            grads[key] = self.call_param(
                loss=loss,
                key=key,
                grad=grad,
                param=values[key],
            )

        return grads


def _return_model_and_loss(
    func: Callable[P, T],
) -> Callable[P, tuple[T, NodeType]]:
    def wrapped(model: NodeType, *args, **kwargs):
        result = func(model, *args, **kwargs)
        return result, model

    return wrapped


@typecheck
def _minimize(
    optimizer: Optimizer,
    loss_fn: Loss,
    model: M,
    *args: Any,
    **kwargs: Any,
) -> tuple[Optimizer, M, Float[Array, ""]]:
    loss_fn_wrapped = _return_model_and_loss(loss_fn)
    loss_fn_wrapped = value_and_grad(
        loss_fn_wrapped,
        collection=loss_fn.collection,
        has_aux=True,
    )

    (loss_value, model), grads = loss_fn_wrapped(model, *args, **kwargs)
    arrays = get_arrays(model, loss_fn.collection)

    # subset of arrays that is used for gradient descent
    arrays_subset: dict[str, Array] = {}
    missed_keys: list[str] = []

    for key in grads.keys():
        if key not in arrays:
            logger.error(f"Gradient at {key} but not found in module")
            missed_keys.append(key)
            continue

        arrays_subset[key] = arrays[key]

    for key in missed_keys:
        grads.pop(key)

    # process gradients
    grads = optimizer(
        model=model,
        loss=loss_value,
        grads=grads,
        values=arrays_subset,
    )

    for key, grad in grads.items():
        # perform gradient descent with modified gradients
        arrays_subset[key] = arrays_subset[key] - grad

    # update model
    model = set_arrays(model, arrays_subset)
    return optimizer, model, loss_value


@jit
def _minimize_jit(
    optimizer: Optimizer,
    loss_fn: Loss,
    model: M,
    *args: Any,
    **kwargs: Any,
) -> tuple[Optimizer, M, Float[Array, ""]]:
    return _minimize(
        optimizer,
        loss_fn,
        model,
        *args,
        **kwargs,
    )
