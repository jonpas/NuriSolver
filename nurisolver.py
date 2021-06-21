#!/usr/bin/env python3

import argparse
import sys
import os
import time
import logging
import unittest
from enum import IntEnum
from operator import itemgetter
from pprint import pprint

import numpy as np  # [y, x] or [height, width]


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("nurisolver.log", "w", encoding="utf-8"), logging.StreamHandler()])


class State(IntEnum):
    UNKNOWN = -2  # White (dot in unsolved source)
    SEA = -1  # Black (dot in solved source)
    ISLAND = 0  # Dot (number in source)
    # > 0  Island value (number in source)


def load(file, dot_value=State.UNKNOWN):
    return np.genfromtxt(file, dtype=np.int8, filling_values=dot_value)


class Plotter():
    FONT_SIZE = 50
    LINE_WIDTH = 2

    def __init__(self, shape, debug=False):
        self.shape = shape  # (height, width)
        height, width = self.shape
        self.debug = debug

        global pygame
        import pygame
        pygame.init()

        screen_shape = (width * self.FONT_SIZE + self.LINE_WIDTH, height * self.FONT_SIZE + self.LINE_WIDTH)
        self.screen = pygame.display.set_mode(screen_shape)

        pygame.display.set_caption("NuriSolver")
        self.font = pygame.font.Font(None, self.FONT_SIZE)
        self.font_small = pygame.font.Font(None, self.FONT_SIZE // 3)
        self.screen.fill(pygame.Color("white"))

        # Grid
        for i in range(width + 1):
            start_pos = (i * self.FONT_SIZE, 0)
            end_pos = (i * self.FONT_SIZE, self.screen.get_height())
            pygame.draw.line(self.screen, pygame.Color("black"), start_pos, end_pos, width=self.LINE_WIDTH)

        for i in range(height + 1):
            start_pos = (0, i * self.FONT_SIZE)
            end_pos = (self.screen.get_width(), i * self.FONT_SIZE)
            pygame.draw.line(self.screen, pygame.Color("black"), start_pos, end_pos, width=self.LINE_WIDTH)

        pygame.display.update()

    def plot_cell(self, y, x, value):
        rect = (x * self.FONT_SIZE, y * self.FONT_SIZE, self.FONT_SIZE, self.FONT_SIZE)

        # Type
        if value == State.SEA:
            self.screen.fill(pygame.Color("black"), rect)
        elif value == State.ISLAND:
            location = (rect[0] + self.FONT_SIZE // 2 - self.LINE_WIDTH * 2, rect[1])
            cell = self.font.render(".", True, pygame.Color("black"))
            self.screen.blit(cell, location)
        elif value > 0:
            long_offset = 10 if value >= 10 else 0
            location = (rect[0] + self.FONT_SIZE // 3 - long_offset, rect[1] + self.FONT_SIZE // 5)
            cell = self.font.render(str(value), True, pygame.Color("black"))
            self.screen.blit(cell, location)

        # Coordinates
        if self.debug:
            location = (rect[0] + self.FONT_SIZE // 10, rect[1] + self.FONT_SIZE // 12)
            coords = self.font_small.render(f"{y},{x}", True, pygame.Color("red"))
            self.screen.blit(coords, location)

        pygame.display.update(rect)

    def plot(self, puzzle):
        assert puzzle.shape == self.shape, f"error! puzzle size {puzzle.shape} not same as plot size {self.shape}"

        height, width = puzzle.shape

        for x in range(width):
            for y in range(height):
                self.plot_cell(y, x, puzzle[y, x])

    def handle_events(self, puzzle, plot_wait=False):
        """Handle pygame events. Optionally wait for input to continue plotting."""
        waited_ev = []
        if plot_wait:
            waited_ev.append(pygame.event.wait())

        for ev in waited_ev + pygame.event.get():
            if ev.type == pygame.QUIT:  # graceful quit
                raise KeyboardInterrupt()
            elif ev.type == pygame.VIDEOEXPOSE:  # redraw on focus
                self.plot(puzzle)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    raise KeyboardInterrupt()
                elif ev.key == pygame.K_SPACE:
                    return False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                return False

        return plot_wait


class Solver():
    def __init__(self, puzzle, plotter=None, verbose_from_step=np.inf):
        self.puzzle = puzzle
        self.plotter = plotter
        self.verbose_from_step = verbose_from_step

        self.prepare()

    def prepare(self):
        self.solved = False
        self.step = 0
        self.height, self.width = self.puzzle.shape

        # Verify puzzle input
        assert self.height > 0 and self.width > 0, f"invalid puzzle size ({self.height}, {self.width})"

        # Adjacent cells can't be islands (only need to check right or down)
        for x in range(self.width - 1):
            for y in range(self.height - 1):
                if self.puzzle[y, x] > 0:
                    assert self.puzzle[y + 1, x] <= 0 or self.puzzle[y, x + 1] <= 0, "adjacent starting islands"

        # Pre-calculate amount of black (Sea) cells
        # width * height - (sum of all numbered cells)
        self.sea_size = self.width * self.height - np.sum(self.puzzle[self.puzzle > 0])

    def save(self):
        return [self.puzzle.copy(), self.step, self.islands.copy(), self.seas.copy()]

    def load(self, state):
        self.puzzle, self.step, self.islands, self.seas = state

        if self.plotter:
            self.plotter.plot(self.puzzle)

    def validate(self):
        """Validates solution for general Nurikabe correctness."""
        single_sea = len(self.seas) == 1
        full_sea = abs(np.sum(self.puzzle[self.puzzle == State.SEA])) == self.sea_size
        no_unknowns = np.sum(self.puzzle[self.puzzle == State.UNKNOWN]) == 0

        logging.debug(f"{single_sea=}, {full_sea=}, {no_unknowns=}")
        return single_sea and full_sea and no_unknowns

    def validate_partial(self):
        """Validates part-solution for correctness."""
        # No pools
        for x in range(self.width - 1):
            for y in range(self.height - 1):
                c1, c2, c3, c4 = (y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)
                pool = [self.puzzle[c1], self.puzzle[c2], self.puzzle[c3], self.puzzle[c4]]

                if all(cell == State.SEA for cell in pool):
                    return False, f"pool detected ({y}, {x})"

        # No cut-off seas
        if len(self.seas) > 1:
            for center, cells in self.seas.items():
                ways = self.extension_ways(cells)
                if len(ways) == 0:
                    return False, f"cut-off sea detected {center}"

        # No cut-off partial islands
        for center, cells in self.islands.items():
            left = self.puzzle[center] - len(cells)
            if left > 0:
                ways = self.extension_ways(cells)
                if len(ways) == 0:
                    pprint(self.islands)
                    return False, f"cut-off incomplete island detected {center} {left}"

        return True, "ok"

    def set_cell(self, y, x, state, center=None):
        assert self.puzzle[y, x] <= 0, f"unable to change island center ({y}, {x})"

        if self.puzzle[y, x] != state:
            self.puzzle[y, x] = state
            self.step += 1

            if self.plotter:
                self.plotter.plot_cell(y, x, state)
                while self.plotter.handle_events(self.puzzle, plot_wait=self.step >= self.verbose_from_step):
                    pass

        # Tracking
        if center is not None:
            if state == State.SEA:
                if self.seas.get(center, None) is None:
                    self.seas[center] = [(y, x)]
                elif (y, x) not in self.seas[center]:
                    self.seas[center].append((y, x))
            elif state == State.ISLAND:
                self.islands[center].append((y, x))

    def four_way(self, y, x, state=None, func=None, check_state=State.UNKNOWN):
        """Perform four-way operation. Island check includes centers"""
        if y > 0 and min(self.puzzle[y - 1, x], 0) == check_state:
            if func:
                func(y - 1, x)
            if state:
                self.set_cell(y - 1, x, state, center=(y - 1, x))
        if y < self.height - 1 and min(self.puzzle[y + 1, x], 0) == check_state:
            if func:
                func(y + 1, x)
            if state:
                self.set_cell(y + 1, x, state, center=(y + 1, x))
        if x > 0 and min(self.puzzle[y, x - 1], 0) == check_state:
            if func:
                func(y, x - 1)
            if state:
                self.set_cell(y, x - 1, state, center=(y, x - 1))
        if x < self.width - 1 and min(self.puzzle[y, x + 1], 0) == check_state:
            if func:
                func(y, x + 1)
            if state:
                self.set_cell(y, x + 1, state, center=(y, x + 1))

    @classmethod
    def distance(cls, cell1, cell2):
        """Calculates Manhattan distance between 2 cells."""
        (y1, x1), (y2, x2) = cell1, cell2
        return np.abs(y1 - y2) + np.abs(x1 - x2)

    def connect_to_island(self, y, x):
        """Connects single Island cell to an Island. Does NOT update the islands map, use with walk_island() only!"""
        ways = []
        self.four_way(y, x, None, lambda ny, nx: ways.append((ny, nx)))
        if len(ways) == 1:
            ny, nx = ways[0]
            self.set_cell(ny, nx, State.ISLAND)
            return ny, nx
        else:
            # Left with a lone island
            self.islands[y, x] = []  # Add cell in walk_island()
            return None

    def walk_island(self, y, x):
        """Walks island cells to center and adds the new cell"""
        cy, cx = y, x
        connect = [(cy, cx)]  # Cells we bridge to the center to form a path, but were not Islands yet

        walked = []  # Cells we walk over (may already be islands) to avoid walking backwards
        path = []  # Cells we still need to walk over and check
        while (cy, cx) not in self.islands:
            walked.append((cy, cx))
            logging.debug(f"{self.step}: Walk island ({cy}, {cx})")

            self.four_way(cy, cx, None, lambda ny, nx: path.append((ny, nx)), check_state=State.ISLAND)
            path = [x for x in path if x not in walked]  # Remove already-walked cells
            logging.debug(f"{self.step}: Walk island path: {path}")

            if len(path) == 0:
                connection = self.connect_to_island(cy, cx)
                if connection is None:
                    break
                cy, cx = connection
                connect.append((cy, cx))
            else:
                cy, cx = path[0]
                path.extend(path[1:])

        self.islands[cy, cx].extend(connect)

    def extension_ways(self, cells, check_state=State.UNKNOWN):
        """Checks if any cell can extend any single way."""
        ways = []

        for (cy, cx) in cells:
            self.four_way(cy, cx, None, lambda ny, nx: ways.append((ny, nx)), check_state=check_state)

        return ways

    def safe_potential_sea_cutoff(self, y, x, imagine_blocks=[]):
        # Find sea center from given cell
        center, cells = next(((center, cells) for center, cells in self.seas.items() if (y, x) in cells), None)

        # Check if any cell can extend any single way
        ways = self.extension_ways(cells)
        ways = [way for way in ways if way not in imagine_blocks]  # Remove imaginative blocks

        if len(ways) == 0:
            # Expand sea to potential blocks as it has to be expanded to prevent cutoff
            for (by, bx) in imagine_blocks:
                logging.debug(f"{self.step}: Safed sea {center} from cutoff by adding ({by}, {bx})")
                self.set_cell(by, bx, State.SEA, center=center)

            return True
        return False

    def extend_islands(self):
        extended = 0

        for center, cells in self.islands.copy().items():
            left = self.puzzle[center] - len(cells)
            if left > 0:
                # Check if any cell can extend any single way
                ways = self.extension_ways(cells)

                if len(ways) == 1:
                    # Island if only one possible way (guaranteed correct)
                    ny, nx = ways[0]
                    logging.debug(f"{self.step}: Extended island {center} to ({ny}, {nx})")
                    self.set_cell(ny, nx, State.ISLAND, center=center)
                    extended += 1
                elif len(ways) == 2:
                    if left == 1:
                        # Diagonal is Sea if island with remaining size 1 can only extend 2 ways
                        (ny1, nx1), (ny2, nx2) = ways

                        # Find same Unknown neighbour
                        ways = [[], []]
                        self.four_way(ny1, nx1, None, lambda ny, nx: ways[0].append((ny, nx)))
                        self.four_way(ny2, nx2, None, lambda ny, nx: ways[1].append((ny, nx)))
                        ways = list(set(ways[0]) & set(ways[1]))  # Same neighbour

                        if len(ways) == 1:
                            ny, nx = ways[0]
                            logging.debug(f"{self.step}: Safed island {center} from ({ny}, {nx}) - diagonal sea")
                            self.set_cell(ny, nx, State.SEA, center=(ny, nx))
                    else:
                        # Can only go one way if the other would block a sea patch, as well as only expand sea patch into the potentially blocked way
                        cutoff = False
                        for i, (ny, nx) in enumerate(ways):
                            # Pretend we continue island on one way
                            # Find neighbouring seas and check if any would be cutoff
                            seas = []
                            self.four_way(ny, nx, None, lambda ny, nx: seas.append((ny, nx)), check_state=State.SEA)
                            for (sy, sx) in seas:
                                if self.safe_potential_sea_cutoff(sy, sx, imagine_blocks=[(ny, nx)]):
                                    cutoff = True
                                    break
                            if cutoff:
                                break

                        # If any cutoff, we must go the other way
                        if cutoff:
                            ny, nx = ways[1] if i == 0 else ways[0]
                            logging.debug(f"{self.step}: Extended island {center} to ({ny}, {nx})")
                            self.set_cell(ny, nx, State.ISLAND, center=center)
                            extended += 1

        return extended

    def wrap_full_islands(self):
        wrapped = 0

        for center, cells in self.islands.copy().items():
            if len(cells) == self.puzzle[center]:
                wrapped += 1

                for (cy, cx) in cells.copy():
                    logging.debug(f"{self.step}: Wrap full islands ({cy}, {cx})")
                    self.four_way(cy, cx, State.SEA)

                # Cleanup full island from further processing
                del self.islands[center]

        return wrapped

    def bridge_islands(self):
        bridged = 0

        for center, cells in self.islands.copy().items():
            if len(cells) > 1:  # Only compare to proper islands and not single patches waiting for merging
                ways = self.extension_ways(cells)
                for (wy, wx) in ways:
                    islands = []
                    self.four_way(wy, wx, None, lambda iy, ix: islands.append((iy, ix)), check_state=State.ISLAND)

                    islands = [island for island in islands if island not in cells]  # Prevent going backwards

                    for (iy, ix) in islands:
                        # Find island center from found cell
                        icenter = next((icenter for icenter, icells in self.islands.items() if (iy, ix) in icells), None)
                        if self.puzzle[center] > 0 and self.puzzle[icenter] > 0:  # Only compare to proper islands and not single patches waiting for merging
                            logging.debug(f"{self.step}: Bridging sea between islands {center} and {icenter}")
                            self.set_cell(wy, wx, State.SEA, center=(wy, wx))

        return bridged

    def extend_seas(self):
        extended = 0

        for center, cells in self.seas.copy().items():
            # Check if any cell can extend any single way
            ways = self.extension_ways(cells)
            if len(ways) == 1:
                # Sea if only one possible way (guaranteed correct)
                ny, nx = ways[0]
                logging.debug(f"{self.step}: Extended sea {center} to ({ny}, {nx})")
                self.set_cell(ny, nx, State.SEA, center=center)
                extended += 1

                # Force merge sea patches to prevent over-extension
                self.merge_sea_patches()

                # Recursively call extend so we re-read self.seas with merged patches to avoid doubling seas
                self.extend_seas()
                break

        return extended

    def unreachable_seas(self):
        unreachables = 0

        for x in range(self.width):
            for y in range(self.height):
                if self.puzzle[y, x] == State.UNKNOWN:
                    # Check reachability to all unfinished islands
                    reachable = False
                    for center, cells in self.islands.copy().items():
                        left = self.puzzle[center] - len(cells)

                        # Calculate distance to each island cell
                        for cell in cells:
                            distance = self.distance(cell, (y, x))
                            if distance <= left:
                                reachable = True
                                break

                    if not reachable:
                        logging.debug(f"{self.step}: Unreachable ({y}, {x})")
                        self.set_cell(y, x, State.SEA, center=(y, x))
                        unreachables += 1

        return unreachables

    def potential_pools(self):
        prevented_pools = 0

        for x in range(self.width - 1):
            for y in range(self.height - 1):
                c1, c2, c3, c4 = (y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)
                pool = [
                    [c1, self.puzzle[c1]],
                    [c2, self.puzzle[c2]],
                    [c3, self.puzzle[c3]],
                    [c4, self.puzzle[c4]],
                ]
                pool.sort(key=itemgetter(1))  # Assuming State.UNKNOWN < State.SEA

                if pool[0][1] == State.UNKNOWN and pool[1][1] == State.SEA and pool[2][1] == State.SEA and pool[3][1] == State.SEA:
                    # 3 Seas, 1 Unknown = Must be island
                    cy, cx = pool[0][0]
                    logging.debug(f"{self.step}: Solving pool ({cy}, {cx})")
                    self.set_cell(cy, cx, State.ISLAND)
                    self.walk_island(cy, cx)
                    prevented_pools += 1
                elif pool[0][1] == State.UNKNOWN and pool[1][1] == State.UNKNOWN and pool[2][1] == State.SEA and pool[3][1] == State.SEA:
                    # 2 Seas, 2 Unknowns = At least one of them must be Island
                    # TODO Find which of the 2 Unknowns can be Island
                    pass

        return prevented_pools

    def merge_island_patches(self):
        merged_islands = 0

        for center, cells in self.islands.copy().items():
            if not any(self.puzzle[cell] > 0 for cell in cells):
                # Check if any cell can extend any single way
                ways = self.extension_ways(cells, check_state=State.ISLAND)
                ways = [way for way in ways if way not in cells]  # Remove same-patch cells

                if len(ways) > 0:
                    # Find island center from found cell
                    ny, nx = ways[0]
                    ncenter, ncells = next(((ncenter, ncells) for ncenter, ncells in self.islands.items() if (ny, nx) in ncells), (None, None))

                    if ncenter is not None and center not in ncenter:
                        logging.debug(f"{self.step}: Merging island patch {ncenter} into {center}")
                        self.islands[ncenter].extend(cells)
                        if self.islands.get(center, None) is not None:
                            del self.islands[center]
                        merged_islands += 1

        return merged_islands

    def merge_sea_patches(self):
        merged_seas = 0

        for center, cells in self.seas.copy().items():
            # Check if any cell can extend any single way
            ways = self.extension_ways(cells, check_state=State.SEA)
            ways = [way for way in ways if way not in cells]  # Remove same-patch cells

            if len(ways) > 0:
                # Find sea patch center from sea cell
                ny, nx = ways[0]
                ncenter = next((ncenter for ncenter, ncells in self.seas.items() if (ny, nx) in ncells), None)

                if center != ncenter:
                    logging.debug(f"{self.step}: Merging sea patch {center} into {ncenter}")
                    self.seas[ncenter].extend(cells)
                    del self.seas[center]
                    merged_seas += 1

        return merged_seas

    def solve(self):
        assert not self.solved, "already solved"

        self.islands = dict()  # (y, x): [coordinates] - Incomplete islands
        self.seas = dict()  # (y, x): [coordinates] - Sea patches (final sea should have one big "patch")

        self.max_guesses = 10  # Maximum amount of consecutive gusesses that can fail TODO Arg
        last_valid_state = self.save()  # Saved game state for guessing
        guesses = 0

        # First logical pass and data preparation
        for x in range(self.width):
            for y in range(self.height):
                # Compose islands
                if self.puzzle[y, x] > 1:  # > 1 as we already encompass '1' islands during this data preparation
                    self.islands[y, x] = [(y, x)]

                # Impossible diagonal neighbours (Sea)
                if 0 < x < self.width - 1 and 0 < y < self.height - 1 and self.puzzle[y, x] > 0:
                    # Only check up-left and up-right (down-left and down-right are just mirrored)
                    if self.puzzle[y - 1, x - 1] > 0:
                        self.set_cell(y - 1, x, State.SEA, center=(y - 1, x))
                        self.set_cell(y, x - 1, State.SEA, center=(y, x - 1))
                    elif self.puzzle[y + 1, x - 1] > 0:
                        self.set_cell(y + 1, x, State.SEA, center=(y + 1, x))
                        self.set_cell(y, x - 1, State.SEA, center=(y, x - 1))

                # Sea between horizontal/vertical island centers (Sea)
                if x < self.width - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y, x + 2] > 0:
                        self.set_cell(y, x + 1, State.SEA, center=(y, x + 1))
                if y < self.height - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y + 2, x] > 0:
                        self.set_cell(y + 1, x, State.SEA, center=(y + 1, x))

                # Neighbours of '1' island (Sea)
                if self.puzzle[y, x] == 1:
                    self.four_way(y, x, State.SEA)

        # Continue logical solving steps or guess on failure
        while True:
            # Only possible island extension (Island)
            logging.debug(f"Extending islands ({self.step})")
            extended_islands = self.extend_islands()
            # TODO nikoli_5 - Connect (4, 17) and (6, 17) - only 1 remaining cell, can't connect to anywhere else
            #               - Connect (2, 3) and (1, 3) - same as above

            # Cells around full island (Sea)
            logging.debug(f"Wrapping full islands ({self.step})")
            wrapped_islands = self.wrap_full_islands()

            # Out-of-range of any island (Sea)
            logging.debug(f"Searching for unreachable seas ({self.step})")
            unreachables = self.unreachable_seas()

            # Prevent pools (square Sea) (at least one of four must be Island)
            logging.debug(f"Resolving potential pools ({self.step})")
            prevented_pools = self.potential_pools()

            # Merge island patches that were possibly left from unsuccessful island walk
            logging.debug(f"Merging island patches ({self.step})")
            merged_patches = self.merge_island_patches()

            # Merge sea patches into continously growing connected sea
            logging.debug(f"Merging sea patches ({self.step})")
            merged_seas = self.merge_sea_patches()

            # Extend seas (after merging sea patches to prevent over-extending)
            logging.debug(f"Extending seas ({self.step})")
            extended_seas = self.extend_seas()

            # Fill spacers between partially-complete islands
            logging.debug(f"Bridging sea between partial islands ({self.step})")
            bridged_islands = self.bridge_islands()

            # Merge sea patches again after bridging to avoid cut-off seas misconception in partial check
            logging.debug(f"Merging sea patches ({self.step})")
            merged_seas += self.merge_sea_patches()

            if (extended_islands == 0
                    and wrapped_islands == 0
                    and unreachables == 0
                    and prevented_pools == 0
                    and merged_patches == 0
                    and merged_seas == 0
                    and extended_seas == 0
                    and bridged_islands == 0):
                valid, msg = self.validate_partial()
                if valid:
                    if self.validate():
                        if guesses > 0:
                            logging.info(f"Solved after {guesses} guesses")
                        self.solved = True
                        break
                else:
                    logging.debug(f"Partial validation failed on guess: {msg}")

                    # Backtrack
                    self.load(last_valid_state)

                if guesses >= self.max_guesses:
                    logging.error(f"Exiting after {guesses} failed guesses")
                    break

                # Save
                last_valid_state = self.save()

                # TODO Guess
                guesses += 1
            else:
                valid, msg = self.validate_partial()
                if not valid:
                    logging.error(f"Partial validation failed: {msg}")
                    break

        return self.solved


# Tests
class TestSolver(unittest.TestCase):
    def test_distance_simple(self):
        distance = Solver.distance((0, 0), (2, 2))
        self.assertEqual(distance, 4)

    def test_distance_hard(self):
        distance = Solver.distance((5, 2), (1, 6))
        self.assertEqual(distance, 8)

    def test_solver(self):
        test_folder = "test"
        solved_suffix = "-solved"

        print("\n--- SOLVER START ---")
        errors, warnings = 0, 0
        for file in sorted(os.listdir(test_folder), key=lambda x: (len(x), x)):
            name, ext = os.path.splitext(file)
            if not name.endswith(solved_suffix):
                print(f"{file}...", end="")
                puzzle = load(os.path.join(test_folder, file))
                solution_file = os.path.join(test_folder, f"{name}{solved_suffix}{ext}")

                if os.path.exists(solution_file):
                    solution = load(solution_file, dot_value=State.SEA)

                    solver = Solver(puzzle)
                    success = solver.solve()
                    if success and np.array_equal(solver.puzzle, solution):
                        print(" OK")
                    else:
                        print(f" ERROR\n{solver.puzzle}")
                        errors += 1
                else:
                    print(" NOT FOUND")
                    warnings += 1

        if errors:
            print("--- SOLVER FAIL ---")
        elif warnings:
            print("--- SOLVER WARN ---")
        else:
            print("--- SOLVER SUCCESS ---")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Nurikabe Solver")
    parser.add_argument("file", type=str, nargs="?", help="read puzzle from file (run tests if none)")
    parser.add_argument("--plot", "-p", action="store_true", help="plot solution (requires pygame)")
    parser.add_argument("--verbose", "-v", type=int, nargs="?", default=0, const=1,
                        help="plot solving steps on mouse button or space key press (requires pygame)")
    parser.add_argument("--debug", "-d", action="store_true", help="log debug steps and plot additional information (requires pygame)")
    args = parser.parse_args()

    if args.verbose:
        args.plot = True

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run tests if no puzzle given
    if args.file is None:
        logging.getLogger().setLevel(logging.ERROR)
        return unittest.main()

    # Read puzzle from file
    if not os.path.exists(args.file):
        parser.error(f"{args.file} not found")

    puzzle = load(args.file)

    # Prepare plot
    plotter, verbose_plotter = None, None
    if args.plot:
        plotter = Plotter(puzzle.shape, debug=args.debug)
        plotter.plot(puzzle)

        if args.verbose:
            verbose_plotter = plotter
            logging.info("Press a mouse button or space key to solve in steps.")

    # Solve
    logging.info(f"Solving...\n{puzzle}")
    solver = Solver(puzzle, plotter=verbose_plotter, verbose_from_step=args.verbose)

    try:
        start = time.process_time()  # ignore sleep (visualization)
        success = solver.solve()
        end = time.process_time()
    except KeyboardInterrupt:
        pygame.quit()
        return 2

    elapsed = (end - start) / 1000.0  # milliseconds

    if success:
        logging.info(f"Solved!\n{solver.puzzle}")
    else:
        logging.info("Unsolved!")

    logging.info(f"Process Time: {elapsed} ms")

    # Plot solution
    if args.plot:
        plotter.plot(solver.puzzle)

        # Keep alive until close
        try:
            while True:
                plotter.handle_events(solver.puzzle)
        except KeyboardInterrupt:
            pass

        pygame.quit()

    return (0 if success else 1)


if __name__ == "__main__":
    sys.exit(main())
