import torch
import torch._C as _C
import collections
from collections import OrderedDict
from itertools import chain
from .variable import Variable


class Function(_C._FunctionBase):

    __call__ = _C._FunctionBase._do_forward

    def save_for_backward(self, *tensors):
        self.to_save = tensors

    def mark_dirty(self, *args):
        self.dirty_tensors = args

    def mark_shared_storage(self, *pairs):
        self.shared_pairs = pairs

    def mark_non_differentiable(self, *args):
        self.non_differentiable = args

    def register_hook(self, name, hook):
        self.backward_hooks = self.backward_hooks or OrderedDict()
        assert name not in self.backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        self.backward_hooks[name] = hook

    def remove_hook(self, name):
        assert self.backward_hooks and name in self.backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self.backward_hooks[name]

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError


class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace


def _iter_filter(condition):
    def _iter(self, obj):
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, collections.Mapping):
            for o in obj.values():
                for var in _iter(self, o):
                    yield var
        elif isinstance(obj, collections.Iterable) and not isinstance(obj, (str, bytes)):
            for o in obj:
                for var in _iter(self, o):
                    yield var
        else:
            raise ValueError("NestedInputFunction doesn't know how to process "
                "an input object of type " + torch.typename(obj))
    return _iter


class NestedInputFunction(Function):

    _iter_variables = _iter_filter(lambda o: isinstance(o, Variable))
    _iter_tensors = _iter_filter(torch.is_tensor)
    _iter_None_tensors = _iter_filter(lambda o: o is None or torch.is_tensor(o))

    def _do_forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        flat_args = tuple(self._iter_variables((args, kwargs)))
        return super(NestedInputFunction, self).__call__(*flat_args)

    def backward(self, *gradients):
        result = self.backward_extended(*gradients)
        return tuple(self._iter_None_tensors(result))

    __call__ = _do_forward

    def forward(self, *args):
        result = self.forward_extended(*self.args, **self.kwargs)
        del self.args
        del self.kwargs
        return result

    def save_for_backward(self, *args, **kwargs):
        self.to_save = tuple(self._iter_tensors((args, kwargs)))

    def mark_dirty(self, *args, **kwargs):
        self.dirty_tensors = tuple(self._iter_tensors((args, kwargs)))

    def mark_non_differentiable(self, *args, **kwargs):
        self.non_differentiable = tuple(self._iter_tensors((args, kwargs)))

    def forward_extended(self, *args, **kwargs):
        raise NotImplementedError

    def backward_extended(self, *gradients):
        raise NotImplementedError

