from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .modular_reward_learner import ModularRewardQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["modular_reward_learner"] = ModularRewardQLearner
REGISTRY["tactic_learner"] = ModularRewardQLearner
