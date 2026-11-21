"""Minimal numerical support for the CAT feed-forward experiments."""
from .model import MLP
from .local_adjoint import local_adjoint_gradients, relax_local_adjoint
from .optimizers import Adam, CovariantMoment, MetricWave, Momentum, RMSProp, SGD
from .training import evaluate, train_model, train_model_local_adjoint
from .direct_feedback import DirectFeedbackAlignment, NeighborFeedbackAlignment
from .validated_training import train_with_validation
__all__ = [
    "MLP", "SGD", "Momentum", "RMSProp", "Adam", "CovariantMoment",
    "MetricWave", "local_adjoint_gradients", "relax_local_adjoint",
    "train_model", "train_model_local_adjoint", "evaluate",
    "DirectFeedbackAlignment", "NeighborFeedbackAlignment", "train_with_validation",
]
