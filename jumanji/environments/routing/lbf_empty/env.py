# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

import jumanji.environments.routing.lbf_empty.utils as utils
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.lbf_empty.constants import MOVES
from jumanji.environments.routing.lbf_empty.generator import RandomGenerator
from jumanji.environments.routing.lbf_empty.observer import GridObserver, VectorObserver
from jumanji.environments.routing.lbf_empty.types import Observation, State
from jumanji.environments.routing.lbf_empty.viewer import LevelBasedForagingViewer
from jumanji.types import TimeStep, restart, termination, transition, truncation
from jumanji.viewer import Viewer


class LevelBasedForagingEmpty(Environment[State, specs.MultiDiscreteArray, Observation]):
    """
    An implementation of the Level-Based Foraging environment where agents need to
    cooperate to collect food and split the reward.

    Original implementation: https://github.com/semitable/lb-foraging

    - `observation`: `Observation`
        - `agent_views`: Depending on the `observer` passed to `__init__`, it can be a
          `GridObserver` or a `VectorObserver`.
            - `GridObserver`: Returns an agent's view with a shape of
              (num_agents, 3, 2 * fov + 1, 2 * fov +1).
            - `VectorObserver`: Returns an agent's view with a shape of
              (num_agents, 3 * (num_food + num_agents).
        - `action_mask`: JAX array (bool) of shape (num_agents, 6)
          indicating for each agent which size actions
          (no-op, up, down, left, right, load) are allowed.
        - `step_count`: int32, the number of steps since the beginning of the episode.

    - `action`: JAX array (int32) of shape (num_agents,). The valid actions for each
        agent are (0: noop, 1: up, 2: down, 3: left, 4: right, 5: load).

    - `reward`: JAX array (float) of shape (num_agents,)
        When one or more agents load food, the food level is rewarded to the agents, weighted
        by the level of each agent. The reward is then normalized so that, at the end,
        the sum of the rewards (if all food items have been picked up) is one.

    - Episode Termination:
        - All food items have been eaten.
        - The number of steps is greater than the limit.

    - `state`: `State`
        - `agents`: Stacked Pytree of `Agent` objects of length `num_agents`.
            - `Agent`:
                - `id`: JAX array (int32) of shape ().
                - `position`: JAX array (int32) of shape (2,).
                - `level`: JAX array (int32) of shape ().
                - `loading`: JAX array (bool) of shape ().
        - `food_items`: Stacked Pytree of `Food` objects of length `num_food`.
            - `Food`:
                - `id`: JAX array (int32) of shape ().
                - `position`: JAX array (int32) of shape (2,).
                - `level`: JAX array (int32) of shape ().
                - `eaten`: JAX array (bool) of shape ().
        - `step_count`: JAX array (int32) of shape (), the number of steps since the beginning
          of the episode.
        - `key`: JAX array (uint) of shape (2,)
            JAX random generation key. Ignored since the environment is deterministic.

    Example:
    ```python
    from jumanji.environments import LevelBasedForaging
    env = LevelBasedForaging()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec.generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```

    Initialization Args:
    - `generator`: A `Generator` object that generates the initial state of the environment.
        Defaults to a `RandomGenerator` with the following parameters:
            - `grid_size`: 8
            - `fov`: 8 (full observation of the grid)
            - `num_agents`: 2
            - `num_food`: 2
            - `max_agent_level`: 2
            - `force_coop`: True
    - `time_limit`: The maximum number of steps in an episode. Defaults to 200.
    - `grid_observation`: If `True`, the observer generates a grid observation (default is `False`).
    - `normalize_reward`: If `True`, normalizes the reward (default is `True`).
    - `penalty`: The penalty value (default is 0.0).
    - `viewer`: Viewer to render the environment. Defaults to `LevelBasedForagingViewer`.
    """

    def __init__(
        self,
        generator: Optional[RandomGenerator] = None,
        viewer: Optional[Viewer[State]] = None,
        time_limit: int = 400,
        grid_observation: bool = False,
        reward_one_for_harvest: bool = False,
        normalize_reward: bool = True,
        penalty: float = 0.0,
        grid_sizeX: int = 15,
        grid_sizeY: int = 15,
        fov: int = 2,
        num_agents: int = 3,
        num_food: int = 0,
        enable_diagonal_adjancecy: bool = True,
        agent_food_level_pairs: List[Tuple[int, int]] = [(1, 1)],
        fix_agents_at_positions: Optional[List[Tuple[int, int]]] = None,
        fix_targets_at_positions: Optional[List[Tuple[int, int]]] = None,
        others_influence: bool = False,
    ) -> None:
        fix_agents_at_positions = jnp.array(
            fix_agents_at_positions
        )  # jnp.ones_like(jnp.array(fix_agents_at_positions)) * -1
        fix_targets_at_positions = jnp.array(fix_targets_at_positions)
        self._generator = RandomGenerator(
            grid_sizeX=grid_sizeX,
            grid_sizeY=grid_sizeY,
            fov=fov,
            num_agents=num_agents,
            force_coop=True,
            agent_food_level_pairs=agent_food_level_pairs,
            fix_agents_at_positions=fix_agents_at_positions,
            others_influence=others_influence,
        )
        self.time_limit = time_limit
        self.grid_sizeX: int = self._generator.grid_sizeX
        self.grid_sizeY: int = self._generator.grid_sizeY
        self.num_agents: int = self._generator.num_agents
        self.fov = self._generator.fov
        self.normalize_reward = normalize_reward
        self.penalty = penalty
        self.enable_diagonal_adjancecy = enable_diagonal_adjancecy
        self.reward_one_for_harvest = reward_one_for_harvest

        assert self.num_agents > 0, "Number of agents must be greater than 0."
        assert self.grid_sizeX > 0, "Grid size X must be greater than 0."
        assert self.grid_sizeY > 0, "Grid size Y must be greater than 0."
        assert len(fix_agents_at_positions) == self.num_agents
        self.fix_agents_at_positions = fix_agents_at_positions
        self.fix_targets_at_positions = fix_targets_at_positions
        self.others_influence = others_influence
        self.num_food = num_food

        self._observer: Union[VectorObserver, GridObserver]
        if not grid_observation:
            self._observer = VectorObserver(
                fov=self.fov,
                grid_sizeX=self.grid_sizeX,
                grid_sizeY=self.grid_sizeY,
                num_agents=self.num_agents,
                num_food=0,
            )
        else:
            self._observer = GridObserver(
                fov=self.fov,
                grid_sizeX=self.grid_sizeX,
                grid_sizeY=self.grid_sizeY,
                num_agents=self.num_agents,
                num_food=0,
            )

        super().__init__()

        # create viewer for rendering environment
        self._viewer = viewer or LevelBasedForagingViewer(
            self.grid_sizeX, self.grid_sizeY, "LevelBasedForaging"
        )

    def __repr__(self) -> str:
        return (
            "LevelBasedForaging(\n"
            + f"\t grid_width={self.grid_sizeX},\n"
            + f"\t grid_height={self.grid_sizeY},\n"
            + f"\t num_agents={self.num_agents}, \n"
            + f"\t num_food={0}, \n"
            + f"\t max_agent_level={self._generator.max_agent_level}\n"
            ")"
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment.

        Args:
            key (chex.PRNGKey): Used to randomly generate the new `State`.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the new initial state
            of the environment and `TimeStep` object corresponding to the initial timestep.
        """
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)
        timestep = restart(observation, shape=self.num_agents)
        timestep.extras = self._get_extra_info(state, timestep)

        return state, timestep

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep]:
        """Simulate one step of the environment.

        Args:
            state (State): State  containing the dynamics of the environment.
            actions (chex.Array): Array containing the actions to take for each agent.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the next state and
            `TimeStep` object corresponding the timestep returned by the environment.
        """
        # Set action to NOOP if agent is fixed.
        actions = jnp.where(self.fix_agents_at_positions[:, 0] != -1, 0, actions)
        # Move agents, fix collisions that may happen and set loading status.
        moved_agents = utils.update_agent_positions(
            state.agents, actions, self.grid_sizeX, self.grid_sizeY, self.others_influence
        )

        reward = 0
        reward -= 0.01

        state = State(
            agents=moved_agents,
            step_count=state.step_count + 1,
            key=state.key,
        )
        observation = self._observer.state_to_observation(state)

        # First condition is truncation, second is termination.
        terminate = False
        truncate = state.step_count >= self.time_limit

        timestep = jax.lax.switch(
            terminate + 2 * truncate,
            [
                # !terminate !trunc
                lambda rew, obs: transition(reward=rew, observation=obs, shape=self.num_agents),
                # terminate !truncate
                lambda rew, obs: termination(reward=rew, observation=obs, shape=self.num_agents),
                # !terminate truncate
                lambda rew, obs: truncation(reward=rew, observation=obs, shape=self.num_agents),
                # terminate truncate
                lambda rew, obs: termination(reward=rew, observation=obs, shape=self.num_agents),
            ],
            reward,
            observation,
        )
        timestep.extras = self._get_extra_info(state, timestep)

        return state, timestep

    def _get_extra_info(self, state: State, timestep: TimeStep) -> Dict:
        return {"percent_eaten": 0}

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the `LevelBasedForaging` environment.

        Args:
            state (State): The current environment state to be rendered.

        Returns:
            Optional[NDArray]: Rendered environment state.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animation from a sequence of states.

        Args:
            states (Sequence[State]): Sequence of `State` corresponding to subsequent timesteps.
            interval (int): Delay between frames in milliseconds, default to 200.
            save_path (Optional[str]): The path where the animation file should be saved.

        Returns:
            matplotlib.animation.FuncAnimation: Animation object that can be saved as a GIF, MP4,
            or rendered with HTML.
        """
        return self._viewer.animate(states=states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the environment.

        The spec's shape depends on the `observer` passed to `__init__`.

        The GridObserver returns an agent's view with a shape of
            (num_agents, 3, 2 * fov + 1, 2 * fov +1).
        The VectorObserver returns an agent's view with a shape of
        (num_agents, 3 * num_food + 3 * num_agents).
        See a more detailed description of the observations in the docs
        of `GridObserver` and `VectorObserver`.

        Returns:
            specs.Spec[Observation]: Spec for the `Observation` with fields grid,
            action_mask, and step_count.
        """
        max_food_level = self.num_agents * self._generator.max_agent_level
        return self._observer.observation_spec(
            self._generator.max_agent_level,
            max_food_level,
            self.time_limit,
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        Returns:
            specs.MultiDiscreteArray: Action spec for the environment with shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(MOVES)] * self.num_agents),
            dtype=jnp.int32,
            name="action",
        )

    @cached_property
    def reward_spec(self) -> specs.Array:
        """Returns the reward specification for the `LevelBasedForaging` environment.

        Since this is a multi-agent environment each agent gets its own reward.

        Returns:
            specs.Array: Reward specification, of shape (num_agents,) for the  environment.
        """
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )
