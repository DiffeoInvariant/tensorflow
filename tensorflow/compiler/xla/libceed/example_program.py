from jax import numpy as jnp
import jax

def cr_fn(x, y):
    return x @ x + y * y

cr_jac = jax.jacfwd(cr_fn, argnums=(0, 1))
cr_jac(jnp.eye(3), jnp.eye(3))

def cr_j(x, y):
    return cr_jac(x, y)
