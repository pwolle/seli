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
    Base class for all gradient basedoptimizers.
    """

    def minimize(
        self,
        loss_fn: Loss,
        model: M,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Self, M, Float[Array, ""]]:
        """
        Minimize the loss function with the given optimizer.

        Parameters
        ----------
        loss_fn : Loss
            The loss function to minimize.

        model : NodeType
            The model to minimize the loss function for.

        args : Any
            Additional arguments to pass to the loss function.

        kwargs : Any
            Additional keyword arguments to pass to the loss function.

        Returns
        -------
        optimizer : Optimizer
            The optimizer.

        model : NodeType
            The model.

        loss : Float[Array, ""]
            The loss value.
        """
        return _minimize_jit(self, loss_fn, model, *args, **kwargs)

    def call_param(
        self,
        loss: Float[Array, ""],
        key: str,
        grad: Float[Array, "*s"],
        param: Float[Array, "*s"],
    ) -> Float[Array, "*s"]:
        """
        Process the gradients of a single parameter. This function is useful
        for implementing custom optimizers that essentially run the same
        function for all parameters. This is the case for most well known
        optimizers.

        Parameters
        ----------
        loss : Float[Array, ""]
            The absolute loss value.

        key : str
            The key of the parameter.

        grad : Float[Array]
            The gradients of the parameter.

        param : Float[Array]
            The parameter values.

        Returns
        -------
        grad : Float[Array]
            The processed gradients of the parameter.
        """
        return grad

    def call_model(
        self,
        model: NodeType,
        loss: Float[Array, ""],
        grads: dict[str, Float[Array, "..."]],
        values: dict[str, Float[Array, "..."]],
    ) -> dict[str, Float[Array, "..."]]:
        """
        Process the gradients of the whole model. The absolute loss value and
        parameter values are also provided to the optimizer.

        This function is useful for implementing custom optimizers that work
        on the whole model at once.

        Parameters
        ----------
        model : NodeType
            The model to process.

        loss : Float[Array, ""]
            The absolute loss value.

        grads : dict[str, Float[Array, "..."]]
            The gradients of the model parameters.

        values : dict[str, Float[Array, "..."]]
            The parameter values of the model.

        Returns
        -------
        grads : dict[str, Float[Array, "..."]]
            The processed gradients of the model parameters.
        """
        return grads

    def __call__(
        self,
        model: NodeType,
        loss: Float[Array, ""],
        grads: dict[str, Float[Array, "..."]],
        values: dict[str, Float[Array, "..."]],
    ) -> dict[str, Float[Array, "..."]]:
        """
        Process the gradients of the whole model. The absolute loss value and
        parameter values are also provided to the optimizer.

        Parameters
        ----------
        model : NodeType
            The model to process.

        loss : Float[Array, ""]
            The absolute loss value.

        grads : dict[str, Float[Array, "..."]]
            The gradients of the model parameters.

        values : dict[str, Float[Array, "..."]]
            The parameter values of the model.
        """
        grads = self.call_model(
            model=model,
            loss=loss,
            values=values,
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
    """
    Minimize the loss function with the given optimizer. Helper function for
    the jit compiled `Optimizer.minimize` method.
    """
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
    # print(grads.keys())

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
    """
    Minimize the loss function with the given optimizer. Helper function for
    the jit compiled `Optimizer.minimize` method.
    """
    return _minimize(
        optimizer,
        loss_fn,
        model,
        *args,
        **kwargs,
    )
