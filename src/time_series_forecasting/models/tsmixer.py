import jax
import jax.numpy as jnp
from flax import linen as nn


class ResidualBlock(nn.Module):
    hidden_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        # Temporal
        out = nn.BatchNorm(use_running_average=not train)(x)
        out = jnp.transpose(out, axes=(0, 2, 1))
        out = nn.relu(nn.Dense(out.shape[-1])(out))
        out = jnp.transpose(out, axes=(0, 2, 1))
        out = nn.Dropout(self.dropout_rate, deterministic=not train)(out)

        # Residual
        x = x + out

        # Feature
        out = nn.BatchNorm(use_running_average=not train)(x)
        out = nn.relu(nn.Dense(self.hidden_size)(out))
        out = nn.Dropout(self.dropout_rate, deterministic=not train)(out)
        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout_rate, deterministic=not train)(out)

        return x + out


class TSMixer(nn.Module):
    output_size: int
    num_blocks: int
    hidden_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_size, self.dropout_rate)(x, train)

        out = jnp.transpose(x, axes=(0, 2, 1))
        out = nn.Dense(self.output_size)(out)
        out = jnp.transpose(out, axes=(0, 2, 1))

        return out
