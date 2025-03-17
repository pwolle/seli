import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from seli.core._module import Module, NodeType
from seli.core._typecheck import typecheck


@typecheck
class Loss(Module, name="opt.Loss"):
    """
    Base class for all loss functions.
    """

    @property
    def collection(self) -> str | None:
        if not hasattr(self, "_collection"):
            return "param"

        return self._collection

    @collection.setter
    def collection(self, value: str | None):
        self._collection = value

    def __call__(self, model: NodeType, *args, **kwargs) -> Float[Array, ""]:
        error = "Subclasses must implement this method"
        raise NotImplementedError(error)


class MeanSquaredError(Loss, name="opt.MeanSquaredError"):
    """
    Mean squared error loss function.
    """

    def __call__(
        self,
        model: NodeType,
        y_true: Float[Array, "..."],
        *model_args,
        **model_kwargs,
    ) -> Float[Array, ""]:
        y_pred = model(*model_args, **model_kwargs)
        return jnp.mean(jnp.square(y_pred - y_true))


class MeanAbsoluteError(Loss, name="opt.MeanAbsoluteError"):
    """
    Mean absolute error loss function.
    """

    def __call__(
        self,
        model: NodeType,
        y_true: Float[Array, "..."],
        *model_args,
        **model_kwargs,
    ) -> Float[Array, ""]:
        y_pred = model(*model_args, **model_kwargs)
        return jnp.mean(jnp.abs(y_pred - y_true))
