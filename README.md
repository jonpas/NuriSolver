# FERI-NuriSolver

[Nurikabe](https://en.wikipedia.org/wiki/Nurikabe_(puzzle)) puzzle solver at Optimization Methods (slo. Optimizacijske Metode). 

Solves any-size Nurikabe using a series of logical procedures and rules. Harder puzzles use guess and backtrack in addition to logical solving steps.


## Usage

```
usage: nurisolver.py [-h] [--plot] [--guess GUESS] [--verbose [VERBOSE]] [--debug] [file]

Nurikabe Solver

positional arguments:
  file                  read puzzle from file (run tests if none)

optional arguments:
  -h, --help            show this help message and exit
  --plot, -p            plot solution (requires pygame)
  --guess GUESS, -g GUESS
                        guess steps when logic is exhausted, limited by maximum amount of failed guesses
                        (default: 500)
  --verbose [VERBOSE], -v [VERBOSE]
                        plot solving steps on mouse button or space key press (requires pygame), optionally
                        start on given step
  --debug, -d           log debug steps and plot additional information (requires pygame)
```


## Setup

_Targetted at Python 3.9._

- `$ python -m venv venv` (virtual environment)
- `$ source venv/bin/activate`
- `$ pip install -r requirements.txt`
  - `$ pip freeze > requirements.txt` (update requirements)

**Dependencies:**
- [NumPy](https://numpy.org/)
- [PyGame](https://www.pygame.org/) _(optional)_

### Resources

- [Nurikabe (puzzle)](https://en.wikipedia.org/wiki/Nurikabe_(puzzle))
- [microsoft/nurikabe](https://github.com/microsoft/nurikabe)
