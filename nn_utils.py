# Stolen from Andy's smc branch

import jax
from jax import vmap
import jax.numpy as np
import flax.linen as nn
from jax.tree_util import register_pytree_node_class
import jax.scipy as scipy

from typing import (NamedTuple, Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inv_softplus(x, eps=1e-4):
    return np.log(np.exp(x - eps) - 1)

def vectorize_pytree(*args):
    """
    Flatten an arbitrary PyTree into a vector.
    :param args:
    :return:
    """
    flat_tree, _ = jax.tree_util.tree_flatten(args)
    flat_vs = [x.flatten() for x in flat_tree]
    return np.concatenate(flat_vs, axis=0)

# converts an (n(n+1)/2,) vector of Lie parameters
# to an (n, n) matrix
def lie_params_to_constrained(out_flat, dim, eps=1e-4):
    D, A = out_flat[:dim], out_flat[dim:]
    # ATTENTION: we changed this!
    # D = np.maximum(softplus(D), eps)
    D = softplus(D) + eps
    # Build a skew-symmetric matrix
    S = np.zeros((dim, dim))
    i1, i2 = np.tril_indices(dim - 1)
    S = S.at[i1+1, i2].set(A)
    S = S.T
    S = S.at[i1+1, i2].set(-A)

    O = scipy.linalg.expm(S)
    J = O.T @ np.diag(D) @ O
    return J

class MLP(nn.Module):
    """
    Define a simple fully connected MLP with ReLU activations.
    """
    features: Sequence[int]
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    output_nonlinearity: bool = False 

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init, )(x))
        x = nn.Dense(self.features[-1], 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init)(x)
        if self.output_nonlinearity:
            x = nn.relu(x)
        return x

class Identity(nn.Module):
    """
    A layer which passes the input through unchanged.
    """
    features: int

    def __call__(self, inputs):
        return inputs

class Static(nn.Module):
    """
    A layer which just returns some static parameters.
    """
    features: int
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init,
                            (self.features, ))
        return kernel