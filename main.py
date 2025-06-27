import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import time

# change the train_step function definition and function call to see change
# on line 53 and 78
# On my machine
# Pure function took 1.7 seconds
# Impure Func   took 2.7 seconds

# random data
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
images = jax.random.uniform(subkey, shape=(50000, 32, 32, 3), minval=0.0, maxval=1.0, dtype=jnp.float32)

key, subkey = jax.random.split(key)
labels = jax.random.randint(subkey, shape=(50000,), minval=0, maxval=10)

# model
class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=3, out_features=128, kernel_size=(4, 4), strides=(4, 4), padding='VALID', rngs=rngs)
        self.out = nnx.Linear(in_features=128, out_features=10, rngs=rngs)


    def __call__(self, x_BHWC):
        x_BPPD = self.conv(x_BHWC)

        b, h, w, d = x_BPPD.shape
        x_BLD = jnp.reshape(x_BPPD, [b, h*w, d])
        x_BD = x_BLD[:, 0]

        x_BC = self.out(x_BD)
        
        return x_BC

# initialise
model = Model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(0.001))

# train step
def loss_fn(model, batch):
  logits = model(batch['images'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['labels']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(optimizer: nnx.Optimizer, batch):
# def train_step(model, optimizer: nnx.Optimizer, batch):
# pure function
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  optimizer.update(grads)  # In-place updates.

# batches
num_train = images.shape[0]
batch_size = 64

perm = jax.random.permutation(jax.random.PRNGKey(0), num_train)
shuffled_imgs = images[perm]
shuffled_lbls = labels[perm]

batches = [
    {"images": shuffled_imgs[i : i + batch_size],
     "labels": shuffled_lbls[i : i + batch_size]}
    for i in range(0, num_train, batch_size)
]

# time a epoch
start_time = time.time()
for batch in batches:
    train_step(optimizer, batch)
    # train_step(model, optimizer, batch)
    # pure function
end_time = time.time()
duration = end_time - start_time
print(f"Model training took: {duration:.4f} seconds")

