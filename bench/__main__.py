"""Entry-point for `python -m bench <command>`."""
from dotenv import load_dotenv
load_dotenv()  # loads .env from the current working directory

from bench.cli import main

if __name__ == "__main__":
    main()
