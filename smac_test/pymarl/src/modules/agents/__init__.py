REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .scare_agent import SCARERNNAgent
REGISTRY["scare"] = SCARERNNAgent