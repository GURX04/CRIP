"""
crip_cli.py
===========
Entry point for the CRIP CLI executable (`crip`).
Maps command invocations directly to the Invoke task runner configured with tasks.py.
"""

from invoke import Program

program = Program(name="crip", binary="crip", version="0.1.0")


def main():
    program.run()


if __name__ == "__main__":
    main()
