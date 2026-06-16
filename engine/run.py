"""Entry point for `docker run` / `python -m engine.run`.

Wired up once the pipeline pieces exist (data load -> backtest -> metrics ->
viz). Currently a placeholder so the Docker image has a valid CMD.
"""


def main() -> None:
    print("backtest-engine: pipeline not yet implemented")


if __name__ == "__main__":
    main()
