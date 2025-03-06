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

from typing import List, Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.lbf_empty.types import Agent, State


class RandomGenerator:
    """
    Randomly generates Level-Based Foraging (LBF) grids.

    Ensures that no two food items are adjacent and no food is placed on the grid's edge.
    Handles placement of a specified number of agents and food items within a defined grid size.
    """

    def __init__(
        self,
        grid_sizeX: int,
        grid_sizeY: int,
        num_agents: int,
        fov: int,
        agent_food_level_pairs: List[Tuple[int, int]],
        fix_agents_at_positions: List[Tuple[int, int]] = None,
        others_influence: bool = False,
        force_coop: bool = False,
    ):
        """
        Initializes the LBF grid generator.

        Args:
            grid_size (int): The size of the grid.
            num_agents (int): The number of agents.
            num_food (int): The number of food items.
            fov (int): Field of view of an agent.
            max_agent_level (int): Maximum level of agents.
            force_coop (bool): Whether to force cooperation among agents.
        """
        assert num_agents > 0, "Number of agents must be positive."

        self.grid_sizeX = grid_sizeX
        self.grid_sizeY = grid_sizeY
        self.fov = grid_size if fov is None else fov
        self.num_agents = num_agents
        self.agent_food_level_pairs = jnp.array(agent_food_level_pairs)
        self.max_agent_level = max([pair[0] for pair in agent_food_level_pairs])
        self.fix_agents_at_positions = fix_agents_at_positions
        self.active_agents_mask = self.fix_agents_at_positions[:, 0] != -1
        self.others_influence = others_influence
        self.force_coop = force_coop

    def sample_agents(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Randomly samples agent positions on the grid, avoiding positions occupied by food.
        Returns an array where each row corresponds to the (x, y) coordinates of an agent.
        """
        agent_flat_positions = jax.random.choice(
            key=key,
            a=self.grid_sizeX * self.grid_sizeY,
            shape=(self.num_agents,),
            replace=not self.others_influence,  # Avoid agent positions overlaping
            p=mask,
        )
        # Unravel indices to get x and y coordinates
        agent_positions_x, agent_positions_y = jnp.unravel_index(
            agent_flat_positions, (self.grid_sizeX, self.grid_sizeY)
        )

        agent_positions_x = jnp.where(
            self.fix_agents_at_positions[:, 0] != -1,
            self.fix_agents_at_positions[:, 0],
            agent_positions_x,
        )

        agent_positions_y = jnp.where(
            self.fix_agents_at_positions[:, 1] != -1,
            self.fix_agents_at_positions[:, 1],
            agent_positions_y,
        )

        # Stack x and y coordinates to form a 2D array
        return jnp.stack([agent_positions_x, agent_positions_y], axis=1)

    def sample_levels(
        self, min_level: int, max_level: int, shape: chex.Shape, key: chex.PRNGKey
    ) -> chex.Array:
        """Samples levels within specified bounds."""
        return jax.random.randint(key, shape=shape, minval=min_level, maxval=max_level + 1)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a state containing grid, agent, and food item configurations."""
        key_agents, key_levels, key = jax.random.split(key, 3)

        # Generate positions for agents. The mask contains 0's where food is placed,
        # 1's where agents can be placed.
        mask = jnp.ones((self.grid_sizeX, self.grid_sizeY), dtype=bool)

        # Reserve location for fixed agents
        if self.others_influence:
            mask = mask.at[
                self.fix_agents_at_positions[self.active_agents_mask, 0],
                self.fix_agents_at_positions[self.active_agents_mask, 1],
            ].set(False)

        mask = mask.ravel()
        agent_positions = self.sample_agents(key=key_agents, mask=mask)

        sampled_level_agent, _ = jax.random.choice(key_levels, self.agent_food_level_pairs)
        # Generate levels for agents and food items
        agent_levels = jnp.full((self.num_agents,), sampled_level_agent)

        # Create pytrees for agents and food items
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            position=agent_positions,
            level=agent_levels,
            loading=jnp.zeros((self.num_agents,), dtype=bool),
        )
        step_count = jnp.array(0, jnp.int32)

        return State(key=key, step_count=step_count, agents=agents)
