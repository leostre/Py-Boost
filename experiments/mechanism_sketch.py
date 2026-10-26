"""Keep full-width statistics in original order, avoiding a numerical confound."""
from py_boost.multioutput.sketching import TopOutputsSketch


class IdentityAtFullTopK(TopOutputsSketch):
    def __call__(self, grad, hess):
        if self.topk >= grad.shape[1]:
            return grad, hess
        return super().__call__(grad, hess)
