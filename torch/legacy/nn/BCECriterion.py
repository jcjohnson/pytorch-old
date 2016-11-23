import torch
from .Criterion import Criterion

# TODO: use THNN
class BCECriterion(Criterion):
    eps = 1e-12

    def __init__(self, weights=None, sizeAverage=True):
        if weights is not None and weights.dim() != 1:
            raise ValueError("weights input should be 1D Tensor")

        super(BCECriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.buffer = None
        self.weights = weights

    def updateOutput(self, input, target):
         # - log(input) * target - log(1 - input) * (1 - target)
        if input.nelement() != target.nelement():
            raise RuntimeError("input and target size mismatch")

        self.buffer = self.buffer or input.new()

        buffer = self.buffer
        weights = self.weights

        buffer.resize_as_(input)

        if weights is not None and target.dim() != 1:
            weights = self.weights.view(1, target.size(1)).expand_as(target)

        # log(input) * target
        torch.add(buffer, input, self.eps).log_()
        if weights is not None:
            buffer.mul_(weights)

        output = torch.dot(target, buffer)

        # log(1 - input) * (1 - target)
        torch.mul(buffer, input, -1).add_(1+self.eps).log_()
        if weights is not None:
            buffer.mul_(weights)

        output = output + torch.sum(buffer)
        output = output - torch.dot(target, buffer)

        if self.sizeAverage:
            output = output / input.nelement()

        self.output = - output

        return self.output


    def updateGradInput(self, input, target):
         # - (target - input) / ( input (1 - input) )
         # The gradient is slightly incorrect:
         # It should have be divided by (input + self.eps) (1 - input + self.eps)
         # but it is divided by input (1 - input + self.eps) + self.eps
         # This modification requires less memory to be computed.
         if input.nelement() != target.nelement():
            raise RuntimeError("input and target size mismatch")

         self.buffer = self.buffer or input.new()

         buffer = self.buffer
         weights = self.weights
         gradInput = self.gradInput

         if weights is not None and target.dim() != 1:
             weights = self.weights.view(1, target.size(1)).expand_as(target)


         buffer.resize_as_(input)
         # - x ( 1 + self.eps -x ) + self.eps
         torch.add(buffer, input, -1).add_(-self.eps).mul_(input).add_(-self.eps)

         gradInput.resize_as_(input)
         # y - x
         torch.add(gradInput, target, -1, input)
         # - (y - x) / ( x ( 1 + self.eps -x ) + self.eps )
         gradInput.div_(buffer)

         if weights is not None:
             gradInput.mul_(weights)

         if self.sizeAverage:
             gradInput.div_(target.nelement())

         return gradInput

