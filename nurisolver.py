#!/usr/bin/env python3

import argparse
import sys


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Nurikabe Solver")
    parser.add_argument("file", type=str, nargs="?", help="Nurikabe puzzle from file")
    args = parser.parse_args()

    print(args)

    if args.file is None:
        print("No file given, running tests")
    else:
        # TODO Read from file
        pass

    # TODO Solve

    return 0


if __name__ == "__main__":
    sys.exit(main())
