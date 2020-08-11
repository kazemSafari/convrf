"""data wouldn't be tracked by autograd,
and the computed gradients would be incorrect if x is needed in a backward pass.
A safer alternative is to use x. detach(),
which also returns a Tensor that shares data with requires_grad=False,
but will have its in-place changes reported by autograd if x is needed in backward."""