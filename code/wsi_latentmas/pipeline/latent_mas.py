"""Canonical entry point for latent, pruning, and reallocation variants."""

from vision_text_mas.latent_hf_ablation_cli import app


def main() -> None:
    """Run the latent-communication pipeline."""
    app()


if __name__ == "__main__":
    main()
