"""Expert trajectory generator for Bobby Carrot normal levels.

Produces optimal-quality (obs, action) sequences for each level by
combining a crumble-aware greedy target planner with an online BFS
path-finder that re-plans after every step (handles conveyors, crumble
collapses, and other dynamic tile effects automatically).

Usage:
    from Bobby_Carrot.rl_models.expert_solver import ExpertSolver
    solver = ExpertSolver(env)
    trajectory, solved = solver.solve()   # [(obs_array, action_int), ...]
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent.parent
_GAME_PYTHON = _HERE / "Game_Python"
if str(_GAME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GAME_PYTHON))

# ---------------------------------------------------------------------------
# Action constants (must match rl_env.py)
# ---------------------------------------------------------------------------
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
ACTION_DOWN = 3

_DELTAS = {
    ACTION_LEFT:  (-1,  0),
    ACTION_RIGHT: ( 1,  0),
    ACTION_UP:    ( 0, -1),
    ACTION_DOWN:  ( 0,  1),
}


# ---------------------------------------------------------------------------
# Move-validity helper (mirrors Bobby.update_dest / get_valid_actions logic)
# ---------------------------------------------------------------------------

def _is_move_valid(
    old_tile: int,
    new_tile: int,
    action: int,
    keys: Tuple[int, int, int] = (0, 0, 0),
    allow_crumble: bool = True,
) -> bool:
    """Return True iff moving with `action` from `old_tile` to `new_tile` is legal."""
    # Hard impassable
    if new_tile < 18:
        return False
    if new_tile == 31:         # collapsed crumble (hole)
        return False
    if new_tile == 46:         # collected egg (death)
        return False
    # Crumble gate
    if new_tile == 30 and not allow_crumble:
        return False
    # Locked doors
    if new_tile == 33 and keys[0] == 0:
        return False
    if new_tile == 35 and keys[1] == 0:
        return False
    if new_tile == 37 and keys[2] == 0:
        return False

    # --- Arrow redirect tiles (destination entry restrictions) ---
    # Tile 24: can't enter going RIGHT(1) or DOWN(3)
    if new_tile == 24 and action in (1, 3):
        return False
    # Tile 25: can't enter going LEFT(0) or DOWN(3)
    if new_tile == 25 and action in (0, 3):
        return False
    # Tile 26: can't enter going LEFT(0) or UP(2)
    if new_tile == 26 and action in (0, 2):
        return False
    # Tile 27: can't enter going RIGHT(1) or UP(2)
    if new_tile == 27 and action in (1, 2):
        return False

    # --- Conveyor entry restrictions ---
    if new_tile in (28, 40, 41) and action in (2, 3):   # horizontal: no UP/DOWN entry
        return False
    if new_tile in (29, 42, 43) and action in (0, 1):   # vertical: no LEFT/RIGHT entry
        return False
    if new_tile == 40 and action == 1:   # conveyor-left: can't enter going RIGHT
        return False
    if new_tile == 41 and action == 0:   # conveyor-right: can't enter going LEFT
        return False
    if new_tile == 42 and action == 3:   # conveyor-up: can't enter going DOWN
        return False
    if new_tile == 43 and action == 2:   # conveyor-down: can't enter going UP
        return False

    # --- Arrow exit restrictions (source tile) ---
    if old_tile == 24 and action in (0, 2):   # tile 24: can't exit LEFT or UP
        return False
    if old_tile == 25 and action in (1, 2):   # tile 25: can't exit RIGHT or UP
        return False
    if old_tile == 26 and action in (1, 3):   # tile 26: can't exit RIGHT or DOWN
        return False
    if old_tile == 27 and action in (0, 3):   # tile 27: can't exit LEFT or DOWN
        return False

    # --- Conveyor exit restrictions ---
    if old_tile in (28, 40, 41) and action in (2, 3):
        return False
    if old_tile in (29, 42, 43) and action in (0, 1):
        return False
    if old_tile == 40 and action == 1:
        return False
    if old_tile == 41 and action == 0:
        return False
    if old_tile == 42 and action == 3:
        return False
    if old_tile == 43 and action == 2:
        return False

    return True


# ---------------------------------------------------------------------------
# BFS path-finder (operates on a snapshot of the current tile map)
# ---------------------------------------------------------------------------

def bfs_find_path(
    start: Tuple[int, int],
    targets: Set[Tuple[int, int]],
    map_data: List[int],
    keys: Tuple[int, int, int] = (0, 0, 0),
    allow_crumble: bool = True,
) -> Optional[List[int]]:
    """BFS: return the shortest valid action sequence from *start* to any tile in *targets*.

    Returns None when no path exists under current map topology.
    """
    if not targets:
        return None
    if start in targets:
        return []

    # State: (position, path)  — path is list of actions taken
    # visited: set of positions already expanded
    visited: Set[Tuple[int, int]] = {start}
    queue: deque[Tuple[Tuple[int, int], List[int]]] = deque([(start, [])])

    while queue:
        pos, path = queue.popleft()
        px, py = pos

        for action, (dx, dy) in _DELTAS.items():
            nx, ny = px + dx, py + dy
            new_pos = (nx, ny)

            if not (0 <= nx < 16 and 0 <= ny < 16):
                continue
            if new_pos in visited:
                continue

            old_tile = map_data[px + py * 16]
            new_tile = map_data[nx + ny * 16]

            # When the destination IS a target tile (e.g. a crumble gate that is
            # the explicit waypoint), allow stepping onto it even if allow_crumble
            # is False — the crumble flag only prevents routing *through* other
            # crumbles to reach non-crumble destinations.
            effective_allow = allow_crumble or (new_pos in targets and new_tile == 30)

            if not _is_move_valid(old_tile, new_tile, action, keys, effective_allow):
                continue

            new_path = path + [action]
            if new_pos in targets:
                return new_path

            visited.add(new_pos)
            queue.append((new_pos, new_path))

    return None


# ---------------------------------------------------------------------------
# Expert Solver
# ---------------------------------------------------------------------------

class ExpertSolver:
    """Generates a complete expert trajectory for a Bobby Carrot level.

    Strategy (per step):
    1. Find targets reachable *without* crossing any crumble tile (same section).
    2. If targets exist in the current section, go to the nearest one.
    3. If the current section is empty, cross the safest crumble gate.
       - "Safe" means crossing it doesn't orphan uncollected targets in other sections.
    4. If no safe crumble exists but targets remain, cross any crumble.
    5. Once all items are collected, navigate to the finish tile.

    Re-planning happens after EVERY step so conveyor slides, crumble collapses
    and item pickups are automatically handled without additional book-keeping.
    """

    def __init__(
        self,
        env,  # BobbyCarrotEnv
        max_steps: int = 2000,
        stuck_threshold: int = 40,
    ) -> None:
        self.env = env
        self.max_steps = max_steps
        self.stuck_threshold = stuck_threshold  # steps without progress → restart

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[Tuple[np.ndarray, int]], bool]:
        """Run the solver and return (trajectory, solved).

        trajectory — list of (obs_array, action_int) pairs
        solved     — True iff the level was completed
        """
        obs = self.env.reset()
        trajectory: List[Tuple[np.ndarray, int]] = []
        last_progress_step = 0
        prev_collected = 0

        for step in range(self.max_steps):
            # Stall detection: no new items collected for too long → give up
            total_collected = (self.env.bobby.carrot_count
                               + self.env.bobby.egg_count)
            if total_collected > prev_collected:
                prev_collected = total_collected
                last_progress_step = step
            elif step - last_progress_step > self.stuck_threshold:
                break  # Solver is stuck; return what we have

            action = self._choose_action()
            if action is None:
                break   # No reachable waypoint

            trajectory.append((obs.copy(), action))
            obs, _reward, done, info = self.env.step(action)

            if done:
                solved = bool(info.get("level_completed", False))
                return trajectory, solved

        return trajectory, False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keys(self) -> Tuple[int, int, int]:
        b = self.env.bobby
        return (int(b.key_gray), int(b.key_yellow), int(b.key_red))

    def _choose_action(self) -> Optional[int]:
        """Return a single action toward the best next waypoint, or None."""
        env = self.env
        pos = env.bobby.coord_src
        data = list(env.map_info.data)
        keys = self._keys()

        waypoint = self._next_waypoint(pos, data, keys)
        if waypoint is None:
            return None

        # Try path without crumble crossing first, then with
        path = bfs_find_path(pos, {waypoint}, data, keys, allow_crumble=False)
        if path is None:
            path = bfs_find_path(pos, {waypoint}, data, keys, allow_crumble=True)
        if not path:
            # Already at waypoint (0-length path) or unreachable
            return None

        return path[0]   # Execute only the first step; re-plan next iteration

    def _next_waypoint(
        self,
        pos: Tuple[int, int],
        data: List[int],
        keys: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int]]:
        """Determine the single best waypoint to aim for right now."""
        env = self.env

        # ---------- Phase 1: items remain ----------
        if env.target_positions:
            # Prefer targets reachable without crossing any crumble
            local = env._get_reachable_targets_from(pos)
            if local:
                return self._nearest_reachable(pos, local, data, keys, allow_crumble=False)

            # No local targets → approach a safe crumble gate
            safe_crumbles = env._get_safe_crumble_positions()
            if safe_crumbles:
                nearest = self._nearest_reachable(
                    pos, safe_crumbles, data, keys, allow_crumble=False
                )
                if nearest is not None:
                    return nearest

            # No safe crumble reachable: try ANY crumble
            all_crumbles = self._all_crumble_tiles()
            if all_crumbles:
                nearest = self._nearest_reachable(
                    pos, all_crumbles, data, keys, allow_crumble=False
                )
                if nearest is not None:
                    return nearest

            # Last resort: navigate directly to a target with full crumble crossing.
            # May orphan other targets on complex multi-crumble maps, but keeps the
            # solver alive instead of terminating with an empty trajectory.
            return self._nearest_reachable(
                pos, env.target_positions, data, keys, allow_crumble=True
            )

        # ---------- Phase 2: all collected → go to finish ----------
        if env.finish_positions:
            return self._nearest_reachable(
                pos, env.finish_positions, data, keys, allow_crumble=True
            )

        return None

    def _nearest_reachable(
        self,
        pos: Tuple[int, int],
        candidates: Set[Tuple[int, int]],
        data: List[int],
        keys: Tuple[int, int, int],
        allow_crumble: bool,
    ) -> Optional[Tuple[int, int]]:
        """Return the candidate reachable in the fewest BFS steps."""
        best_target: Optional[Tuple[int, int]] = None
        best_dist: int = 10 ** 9

        for target in candidates:
            path = bfs_find_path(pos, {target}, data, keys, allow_crumble)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_target = target

        return best_target

    def _all_crumble_tiles(self) -> Set[Tuple[int, int]]:
        """Return positions of all active crumble tiles (tile ID 30)."""
        crumbles: Set[Tuple[int, int]] = set()
        data = self.env.map_info.data
        for y in range(16):
            for x in range(16):
                if data[x + y * 16] == 30:
                    crumbles.add((x, y))
        return crumbles
