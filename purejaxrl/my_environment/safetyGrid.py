import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

@struct.dataclass
class EnvState:
    x: int
    time: int


@struct.dataclass
class EnvParams:
    hole_reward: int = -5
    gridsize: int = 10
    num_grids: int = 100
    goal_reward: int = 50
    step_reward: int = -1
    max_steps_in_episode: int = 50

class SafeGridworld(environment.Environment):

    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)
        self.init_state = 0
        self.hole_states = jnp.array([30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46, 95, 96, 85, 86, 75, 76, 65, 66, 55, 56, 50, 51, 52, 53, 54,60, 61, 62, 63, 64, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 90, 91, 92, 93, 94])
        self.goal_state = jnp.array([97])
        self.actionsDic = jnp.array([EnvParams.gridsize, 1, -EnvParams.gridsize, -1])

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for SafeGridworld-v1
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ):

        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)

        # Transition
        p = jax.random.uniform(key)

        next_state_high_prob = state + self.actionsDic[action]

        total_actions = jnp.array([0, 1, 2, 3])
        jnp.delete(total_actions, action)
        other_action = jax.random.choice(key, total_actions)

        next_state_low_prob = state + self.actionsDic[other_action]


        print(state, next_state_high_prob, next_state_low_prob)




    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:

        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.isin(state.x, self.hole_states)

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done1, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "SafeGridworld-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        """Observation space of the environment."""

        return spaces.Discrete(params.num_grids)


    def state_space(self, params: EnvParams) -> spaces.Discrete:
        """State space of the environment."""

        return spaces.Discrete(params.num_grids)

