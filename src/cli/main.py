import argparse
import sys

from cli.commands import COMMANDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sorrow")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, command_args = parser.parse_known_args(argv)
    return COMMANDS[args.command](command_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
