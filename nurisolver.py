#!/usr/bin/env python3

import argparse
import sys
import os
import time
import logging
import unittest
from enum import IntEnum
from operator import itemgetter

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

    def validate(self):
        full_sea = np.sum(self.puzzle[self.puzzle == State.SEA]) == self.sea_size
        no_unknowns = np.sum(self.puzzle[self.puzzle == State.UNKNOWN]) == 0

        return full_sea and no_unknowns

    def set_cell(self, y, x, state):
        assert self.puzzle[y, x] <= 0, f"unable to change island center ({y}, {x})"

        if self.puzzle[y, x] != state:
            self.puzzle[y, x] = state
            self.step += 1

            if self.plotter:
                self.plotter.plot_cell(y, x, state)
                while self.plotter.handle_events(self.puzzle, plot_wait=self.step >= self.verbose_from_step):
                    pass

    def four_way(self, y, x, state=None, func=None, check_state=State.UNKNOWN):
        """Perform four-way operation. Island check includes centers"""
        if y > 0 and min(self.puzzle[y - 1, x], 0) == check_state:
            if func:
                func(y - 1, x)
            if state:
                self.set_cell(y - 1, x, state)
        if y < self.height - 1 and min(self.puzzle[y + 1, x], 0) == check_state:
            if func:
                func(y + 1, x)
            if state:
                self.set_cell(y + 1, x, state)
        if x > 0 and min(self.puzzle[y, x - 1], 0) == check_state:
            if func:
                func(y, x - 1)
            if state:
                self.set_cell(y, x - 1, state)
        if x < self.width - 1 and min(self.puzzle[y, x + 1], 0) == check_state:
            if func:
                func(y, x + 1)
            if state:
                self.set_cell(y, x + 1, state)

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
        while (cy, cx) not in self.islands:
            walked.append((cy, cx))
            logging.debug(f"Walk island ({cy}, {cx})")

            path = []
            self.four_way(cy, cx, None, lambda ny, nx: path.append((ny, nx)), check_state=State.ISLAND)
            path = [x for x in path if x not in walked]  # Remove already-walked cells
            logging.debug(f"Walk island path: {path}")

            if len(path) == 0:
                connection = self.connect_to_island(cy, cx)
                if connection is None:
                    break
                cy, cx = connection
                connect.append((cy, cx))
            else:
                assert len(path) == 1, f"invalid island path ({cy}, {cx} = {path})"
                cy, cx = path[0]

        self.islands[cy, cx].extend(connect)

    def extend_islands(self):
        extended = 0

        for center, cells in self.islands.copy().items():
            left = self.puzzle[center] - len(cells)
            if left > 0:
                ways = []

                # Check if any cell can extend any single way
                for (cy, cx) in cells:
                    self.four_way(cy, cx, None, lambda ny, nx: ways.append((ny, nx)))

                if len(ways) == 1:
                    # Island if only one possible way (guaranteed correct)
                    extended += 1

                    ny, nx = ways[0]
                    logging.debug(f"Extended island {center} to ({ny}, {nx})")
                    self.set_cell(ny, nx, State.ISLAND)
                    self.islands[center].append((ny, nx))
                elif len(ways) == 2 and left == 1:
                    # Diagonal is Sea if island with remaining size 1 can only extend 2 ways
                    (ny1, nx1), (ny2, nx2) = ways

                    # Find same Unknown neighbour
                    ways = [[], []]
                    self.four_way(ny1, nx1, None, lambda ny, nx: ways[0].append((ny, nx)))
                    self.four_way(ny2, nx2, None, lambda ny, nx: ways[1].append((ny, nx)))
                    ways = list(set(ways[0]) & set(ways[1]))  # Same neighbour

                    if len(ways) == 1:
                        ny, nx = ways[0]
                        logging.debug(f"Safed island {center} from ({ny}, {nx}) - diagonal sea")
                        self.set_cell(ny, nx, State.SEA)

        return extended

    def wrap_full_islands(self):
        wrapped = 0

        for center, cells in self.islands.copy().items():
            if len(cells) == self.puzzle[center]:
                wrapped += 1

                for (cy, cx) in cells.copy():
                    logging.debug(f"Wrap full islands ({cy}, {cx})")
                    self.four_way(cy, cx, State.SEA)

                # Cleanup full island from further processing
                del self.islands[center]

        return wrapped

    def extend_seas(self):
        # TODO Check Sea patch for connections
        extended = 0

        for x in range(self.width):
            for y in range(self.height):
                if self.puzzle[y, x] == State.SEA:
                    neighbour_seas = []
                    self.four_way(y, x, None, lambda ny, nx: neighbour_seas.append((ny, nx)), check_state=State.SEA)

                    if len(neighbour_seas) == 0:
                        ways = []
                        self.four_way(y, x, None, lambda ny, nx: ways.append((ny, nx)))

                        if len(ways) == 1:
                            extended += 1

                            ny, nx = ways[0]
                            logging.debug(f"Extend seas ({ny}, {nx})")
                            self.set_cell(ny, nx, State.SEA)

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
                        logging.debug(f"Unreachable ({y}, {x})")
                        self.set_cell(y, x, State.SEA)
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

                if (pool[0][1] == State.UNKNOWN and pool[1][1] == State.SEA and pool[2][1] == State.SEA and pool[3][1] == State.SEA):
                    # 3 Seas, 1 Unknown = Must be island
                    cy, cx = pool[0][0]
                    self.set_cell(cy, cx, State.ISLAND)
                    self.walk_island(cy, cx)
                    prevented_pools += 1
                elif (pool[0][1] == State.UNKNOWN and pool[1][1] == State.UNKNOWN and pool[2][1] == State.SEA and pool[3][1] == State.SEA):
                    # 2 Seas, 2 Unknowns = At least one of them must be Island
                    # TODO Find which of the 2 unknowns can be Island
                    pass

        return prevented_pools

    def merge_island_patches(self):
        merged_islands = 0

        for (cy, cx), cells in self.islands.copy().items():
            if len(cells) == 1:
                ways = []
                self.four_way(cy, cx, None, lambda ny, nx: ways.append((ny, nx)), check_state=State.ISLAND)
                if len(ways) == 1:
                    ny, nx = ways[0]
                    ncenter = next((ncenter for ncenter, ncells in self.islands.items() if (ny, nx) in ncells), None)
                    logging.debug(f"Merging ({cy}, {cx}) into {ncenter}")
                    self.islands[ncenter].append((cy, cx))
                    del self.islands[cy, cx]
                    merged_islands += 1

        return merged_islands

    def solve(self):
        assert not self.solved, "already solved"

        self.islands = dict()  # (y, x): [coordinates] - Incomplete islands

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
                        self.set_cell(y - 1, x, State.SEA)
                        self.set_cell(y, x - 1, State.SEA)
                    elif self.puzzle[y + 1, x - 1] > 0:
                        self.set_cell(y + 1, x, State.SEA)
                        self.set_cell(y, x - 1, State.SEA)

                # Sea between horizontal/vertical island centers (Sea)
                if x < self.width - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y, x + 2] > 0:
                        self.set_cell(y, x + 1, State.SEA)
                if y < self.height - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y + 2, x] > 0:
                        self.set_cell(y + 1, x, State.SEA)

                # Neighbours of '1' island (Sea)
                if self.puzzle[y, x] == 1:
                    self.four_way(y, x, State.SEA)

        # Continue logical solving steps
        while True:
            # Only possible island extension (Island)
            extended_islands = self.extend_islands()
            # TODO nikoli_2 - Extend (9, 5) to (6, 4) - 2 ways to extend, one would combine 2 islands
            #               - Fill sea between 2 islands (only size > 1 to prevent wrapping unconnected island patches, 1s are done at the very start anyways)
            #      nikoli_4 - Same as nikoli_2: (3, 7) and (3, 9)
            #      nikoli_5 - Same as nikoli_2: (4, 17) and (6, 17)

            # Cells around full island (Sea)
            wrapped_islands = self.wrap_full_islands()

            # Connect alone Seas (Sea)
            extended_seas = self.extend_seas()
            # TODO Extend sea patches

            # Out-of-range of any island (Sea)
            unreachables = self.unreachable_seas()

            # Prevent pools (square Sea) (at least one of four must be Island)
            prevented_pools = self.potential_pools()

            # Merge island patches that were possibly left from unsuccessful island walk
            merged_patches = self.merge_island_patches()

            if extended_islands == 0 and wrapped_islands == 0 and extended_seas == 0 and unreachables == 0 and prevented_pools == 0 and merged_patches == 0:
                break
                # TODO Guess & Backtrack

        if self.validate():
            self.solved = True

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
        for file in os.listdir(test_folder):
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
