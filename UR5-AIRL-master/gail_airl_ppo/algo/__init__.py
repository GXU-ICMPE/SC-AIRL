from .ppo_da import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .airl_da import DAAIRL
from .sac import SAC
ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'daairl': DAAIRL,
    'ppo': PPO,
    'sac': SAC
}
