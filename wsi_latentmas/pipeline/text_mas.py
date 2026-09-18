"""Canonical entry point for the text-communication VL-MAS baseline."""

from vision_text_mas.matched_vlmas_cli import app


def main() -> None:
    """Run the matched text-communication pipeline."""
    app()


if __name__ == "__main__":
    main()
