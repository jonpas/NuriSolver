#!/usr/bin/env python3

import argparse
import sys
import os
import time
from enum import IntEnum

import numpy as np


PLOT_DELAY = 0.5


class State(IntEnum):
    UNKNOWN = -2  # White
    SEA = -1  # Black
    ISLAND = 0  # Dot
    # > 0  Island value


class Plotter():
    FONT_SIZE = 50
    LINE_WIDTH = 2

    def __init__(self, shape):
        self.shape = shape  # (height, width)

        global pygame
        import pygame
        pygame.init()

        screen_shape = (shape[1] * self.FONT_SIZE + self.LINE_WIDTH, shape[0] * self.FONT_SIZE + self.LINE_WIDTH)
        self.screen = pygame.display.set_mode(screen_shape)

        pygame.display.set_caption("NuriSolver")
        self.font = pygame.font.Font(None, self.FONT_SIZE)
        self.screen.fill(pygame.Color("white"))

        # Grid
        for i in range(self.shape[1] + 1):
            start_pos = (i * self.FONT_SIZE, 0)
            end_pos = (i * self.FONT_SIZE, self.screen.get_height())
            pygame.draw.line(self.screen, pygame.Color("black"), start_pos, end_pos, width=self.LINE_WIDTH)

        for i in range(self.shape[0] + 1):
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
        if puzzle.shape != self.shape:
            print(f"error! puzzle size {puzzle.shape} not same as plot size {self.shape}")
            return 1

        for y in range(puzzle.shape[0]):
            for x in range(puzzle.shape[1]):
                self.plot_cell(y, x, puzzle[y, x])


class Solver():
    def __init__(self, puzzle, plotter=None):
        self.puzzle = puzzle
        self.plotter = plotter

    def set_cell(self, y, x, state):
        assert self.puzzle[y, x] <= 0, f"unable to change island center ({y}, {x})"
        self.puzzle[y, x] = state

        if self.plotter:
            time.sleep(PLOT_DELAY)
            self.plotter.plot_cell(y, x, state)

    def solve(self):
        # TODO
        self.set_cell(1, 0, State.SEA)
        self.set_cell(2, 0, State.SEA)
        return self.puzzle


def load(file, dot_value=State.UNKNOWN):
    return np.genfromtxt(file, dtype=np.int8, delimiter=1, filling_values=dot_value)


def test():
    test_folder = "test"
    solved_suffix = "-solved"

    print("--- TEST START ---")
    errors = 0

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
                errors += 1

    if errors:
        print("--- TEST FAIL ---")
    else:
        print("--- TEST SUCCESS ---")

    return errors


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Nurikabe Solver")
    parser.add_argument("file", type=str, nargs="?", help="read puzzle from file")
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

        if args.verbose:
            verbose_plotter = plotter

        plotter.plot(puzzle)

    # Solve
    print(f"Solving:\n{puzzle}")
    solver = Solver(puzzle.copy(), plotter=verbose_plotter)

    start = time.process_time()  # ignore sleep (visualization)
    solved = solver.solve()
    end = time.process_time()

    print(f"Solved:\n{solved}\nProcess Time: {(end - start) / 1000.0} ms")

    # Plot solution
    if args.plot:
        plotter.plot(solved)

        # Keep alive until close
        try:
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt()
        except KeyboardInterrupt:
            pass

        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
