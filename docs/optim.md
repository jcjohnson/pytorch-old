# torch.optim

The Optim package in Torch is targeted for one to optimize their neural networks
using a wide variety of optimization methods such as SGD, Adam etc.

Currently, the following optimization methods are supported, typically with
options such as weight decay and other bells and whistles.

- SGD
- AdaDelta
- Adagrad
- Adam
- AdaMax
- Averaged SGD
- RProp
- RMSProp


The usage of the Optim package itself is as follows.

1. Construct an optimizer
2. Use `optimizer.step(...)` to optimize.
   - Call `optimizer.zero_grad()` to zero out the gradient buffers when appropriate

## 1. Constructing the optimizer

One first constructs an `Optimizer` object by giving it a list of parameters
to optimize, as well as the optimizer options,such as learning rate, weight decay, etc.

Example:

`optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)`

### Per-parameter options

In a more advanced usage, one can specify per-layer options by passing each parameter group along with it's custom options.

Any parameter group that does not have an attribute defined will use the default attributes.
This is very useful when one wants to specify per-layer learning rates for example.

Example:

`optim.SGD([{'params': model1.parameters()}, {'params': model2.parameters(), 'lr': 1e-3}, lr=1e-2, momentum=0.9)`

Here, `model1`'s parameters will use the default learning rate of `1e-2` and momentum of `0.9`
However, `model2`'s parameters will use a learning rate of `1e-3`, and the default momentum of `0.9`

Then, you can use the optimizer by calling `optimizer.zero_grad()` and `optimizer.step(...)`. Read the next sections.

## 2. `Optimizer.step(...)`

The step function has the following signature:

`Optimizer.step([closure])`

It optionally takes a closure which computes your function to optimize f(x) and returns the loss.

The closure needs to do the following:
- Optimizer.zero_grad()
- Compute the loss, of the (parameters, input, target)
- Call loss.backward()
- return the loss

