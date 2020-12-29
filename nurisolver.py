#!/usr/bin/env python3

import argparse
import sys
import os
import time
from enum import IntEnum

import numpy as np  # [y, x] or [height, width]


PLOT_DELAY = 500  # milliseconds


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

    def __init__(self, shape):
        self.shape = shape  # (height, width)
        height, width = self.shape

        global pygame
        import pygame
        pygame.init()

        screen_shape = (width * self.FONT_SIZE + self.LINE_WIDTH, height * self.FONT_SIZE + self.LINE_WIDTH)
        self.screen = pygame.display.set_mode(screen_shape)

        pygame.display.set_caption("NuriSolver")
        self.font = pygame.font.Font(None, self.FONT_SIZE)
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

        if value == State.SEA:
            self.screen.fill(pygame.Color("black"), rect)
        elif value == State.ISLAND:
            location = (rect[0] + self.FONT_SIZE // 2 - self.LINE_WIDTH * 2, rect[1])
            cell = self.font.render(".", True, pygame.Color("black"))
            self.screen.blit(cell, location)
        elif value > 0:
            location = (rect[0] + self.FONT_SIZE // 3, rect[1] + self.FONT_SIZE // 5)
            cell = self.font.render(str(value), True, pygame.Color("black"))
            self.screen.blit(cell, location)

        pygame.display.update(rect)

    def plot(self, puzzle):
        assert puzzle.shape == self.shape, f"error! puzzle size {puzzle.shape} not same as plot size {self.shape}"

        height, width = puzzle.shape

        for x in range(width):
            for y in range(height):
                self.plot_cell(y, x, puzzle[y, x])

    def handle_events(self, puzzle):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:  # graceful quit
                raise KeyboardInterrupt()
            elif ev.type == pygame.VIDEOEXPOSE:  # redraw on focus
                self.plot(puzzle)


class Solver():
    def __init__(self, puzzle, plotter=None):
        self.puzzle = puzzle
        self.plotter = plotter

        self.prepare()  # sets: self.sea_size

    def prepare(self):
        height, width = self.puzzle.shape

        # Verify puzzle input
        assert height > 0 and width > 0, f"invalid puzzle size ({height}, {width})"

        # Adjacent cells can't be islands (only need to check right or down)
        for x in range(width - 1):
            for y in range(height - 1):
                if self.puzzle[y, x] > 0:
                    assert self.puzzle[y + 1, x] <= 0 or self.puzzle[y, x + 1] <= 0, "adjacent starting islands"

        # Pre-calculate amount of black (Sea) cells
        # width * height - (sum of all numbered cells)
        self.sea_size = width * height - np.sum(self.puzzle[self.puzzle > 0])

    def validate(self):
        full_sea = np.sum(self.puzzle[self.puzzle == State.SEA]) == self.sea_size
        no_unknowns = np.sum(self.puzzle[self.puzzle == State.UNKNOWN]) == 0

        return full_sea and no_unknowns

    def set_cell(self, y, x, state):
        assert self.puzzle[y, x] <= 0, f"unable to change island center ({y}, {x})"

        if self.puzzle[y, x] != state:
            self.puzzle[y, x] = state

            if self.plotter:
                pygame.time.wait(PLOT_DELAY)
                self.plotter.plot_cell(y, x, state)
                self.plotter.handle_events(self.puzzle)

    def solve(self):
        # return load("test/wikipedia_easy-solved.txt", dot_value=State.SEA)  # DEBUG

        height, width = self.puzzle.shape

        for x in range(width):
            for y in range(height):
                # Impossible diagonal neighbours (Sea)
                if 0 < x < width - 1 and 0 < y < height - 1 and self.puzzle[y, x] > 0:
                    # Only check up-left and up-right (down-left and down-right are just mirrored)
                    if self.puzzle[y - 1, x - 1] > 0:
                        self.set_cell(y - 1, x, State.SEA)
                        self.set_cell(y, x - 1, State.SEA)
                    elif self.puzzle[y + 1, x - 1] > 0:
                        self.set_cell(y + 1, x, State.SEA)
                        self.set_cell(y, x - 1, State.SEA)

                # Sea between horizontal/vertical island centers (Sea)
                if x < width - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y, x + 2] > 0:
                        self.set_cell(y, x + 1, State.SEA)
                if y < height - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y + 2, x] > 0:
                        self.set_cell(y + 1, x, State.SEA)

                # Neighbours of '1' island (Sea)
                if self.puzzle[y, x] == 1:
                    if y > 0:
                        self.set_cell(y - 1, x, State.SEA)
                    if y < height - 1:
                        self.set_cell(y + 1, x, State.SEA)
                    if x > 0:
                        self.set_cell(y, x - 1, State.SEA)
                    if x < width - 1:
                        self.set_cell(y, x + 1, State.SEA)

        # TODO Only possible island cells (Island)
        # wiki-hard: top-right 2 can only go down (and can then be encompassed in Sea)
        # wiki-hard: right 5 can only go left (but is not yet finished)

        # TODO Out-of-range of any island (Sea)

        # TODO Cells around full island (Sea)

        # TODO Connect alone Seas (Sea)
        # wiki-easy: bottom-left must connect one up

        return self.puzzle  # if self.validate() else None


def test():
    test_folder = "test"
    solved_suffix = "-solved"

    print("--- TEST START ---")
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
                solved = solver.solve()
                if np.array_equal(solved, solution):
                    print(" OK")
                else:
                    print(f" ERROR\n{solved}")
                    errors += 1
            else:
                print(" NOT FOUND")
                warnings += 1

    if errors:
        print("--- TEST FAIL ---")
    elif warnings:
        print("--- TEST WARN ---")
    else:
        print("--- TEST SUCCESS ---")

    return errors


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Nurikabe Solver")
    parser.add_argument("file", type=str, nargs="?", help="read puzzle from file (run tests if none)")
    parser.add_argument("--plot", "-p", action="store_true", help="plot solution (requires pygame)")
    parser.add_argument("--verbose", "-v", action="store_true", help="plot solving steps (requires pygame)")
    args = parser.parse_args()

    if args.verbose:
        args.plot = True

    # Run tests if no puzzle given
    if args.file is None:
        return test()

    # Read puzzle from file
    if not os.path.exists(args.file):
        parser.error(f"{args.file} not found")

    puzzle = load(args.file)

    # Prepare plot
    plotter, verbose_plotter = None, None
    if args.plot:
        plotter = Plotter(puzzle.shape)
        plotter.plot(puzzle)

        if args.verbose:
            verbose_plotter = plotter

    # Solve
    print(f"Solving...\n{puzzle}")
    solver = Solver(puzzle.copy(), plotter=verbose_plotter)

    start = time.process_time()  # ignore sleep (visualization)
    solved = solver.solve()
    end = time.process_time()
    elapsed = (end - start) / 1000.0  # milliseconds

    success = solved is not None
    if success:
        print(f"Solved!\n{solved}")
    else:
        print("Unsolved!")
        solved = puzzle  # plotting needs

    print(f"Process Time: {elapsed} ms")

    # Plot solution
    if args.plot:
        plotter.plot(solved)

        # Keep alive until close
        try:
            while True:
                plotter.handle_events(solved)
        except KeyboardInterrupt:
            pass

        pygame.quit()

    return success


if __name__ == "__main__":
    sys.exit(main())
