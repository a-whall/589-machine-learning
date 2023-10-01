import jax
import jax.numpy as jnp


# Disable annoying GPU/TPU not found warning
jax.config.update('jax_platform_name', 'cpu')


# Define a mathematical function
def tanh(x):
    
    y = jnp.exp(-2.0 * x)
    
    return (1.0 - y) / (1.0 + y)


# Obtain its gradient function
grad_tanh = jax.grad(tanh)

# Evaluate it
print(grad_tanh(1.0))

# Expected output:
# ----------------
# JAX unsupressable warning...
# 0.4199743