import typer
import yaml
from typing import List, Optional, Literal
from pathlib import Path
import matplotlib.pyplot as plt

from .masking import make_inclusion_mask
from .processing import get_maps, get_maps_diff, prepare_maps
from .estimation import plot_extrapolation_estimate
from .configuration import Settings, dump_config, merge_dicts
from .logger import setup_logger

logger = setup_logger()

app = typer.Typer(help="XTR Estimator Analysis Pipeline")


def parse_extra_args(extra_args: List[str]) -> dict:
    """
    Parses 'masking.sigma=5.0' into {'masking': {'sigma': 5.0}}
    """
    overrides = {}
    for item in extra_args:
        if "=" not in item:
            typer.echo(
                f"Warning: Skipping invalid argument '{item}'. Use key.path=value"
            )
            continue

        key_path, value = item.split("=", 1)

        # Basic type conversion
        if value.lower() == "true":
            parsed_val = True
        elif value.lower() == "false":
            parsed_val = False
        else:
            try:
                parsed_val = float(value) if "." in value else int(value)
            except ValueError:
                parsed_val = value  # Keep as string

        # Build nested dictionary
        keys = key_path.split(".")
        d = overrides
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = parsed_val

    return overrides



def xtr_logic(config: Settings | dict, ax=None, map_dark_base=None) -> tuple:
    if config["general"]["comparison_type"] == "diff":
        map_dark, diffmap = get_maps_diff(config, map_dark=map_dark_base)
    elif config["general"]["comparison_type"] == "triggered":
        if map_dark_base is not None:
            logger.warning("map_dark_base provided but will be ignored in triggered mode.")
        unscaled_dark, unscaled_triggered = get_maps(config)
        diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    else:
        raise ValueError(
            f"Unknown comparison type: {config['general']['comparison_type']}"
        )
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    fig, ax, prediction_tuple = plot_extrapolation_estimate(
        diffmap, map_dark, inclusion_mask, config, ax=ax
    )
    return fig, ax, prediction_tuple, map_dark

def execute_as_main(config: Settings | dict, save2file: bool = False, show: bool = True) -> None:
    """The actual processing logic."""
    # Ensure we have regular dict
    fig, _, prediction_tuple = xtr_logic(config, ax=None)
    filename = f"{config['general']['name_machine']}_extrapolation_estimate.png"
    full_filename = Path(config["general"]["plot_folder"]) / filename

    if config["plot"]["save_to_file"]:
        fig.savefig(full_filename)
    if config["plot"]["show_plot"]:
        plt.show()
    else:
        plt.close(fig)
    return prediction_tuple


def parse_settings(
    data_yaml: Optional[Path | str] = None,
    explicit_tuples: List[tuple] = [],
    extra_overrides: List[str] = [],
) -> Settings:

    def update_nested(section, key, value):
        if value is not None:
            final_payload.setdefault(section, {})[key] = (
                str(value) if isinstance(value, Path) else value
            )

    # 1. Start with Profile YAML (Lowest priority override)
    final_payload = {}
    if data_yaml:
        if isinstance(data_yaml, str):
            data_yaml = Path(data_yaml)
        if data_yaml.exists():
            with open(data_yaml) as f:
                final_payload = yaml.safe_load(f) or {}
    elif data_yaml:
        typer.secho(
            f"⚠️  Warning: YAML file {data_yaml} not found. Proceeding with defaults and CLI flags.",
            fg=typer.colors.YELLOW,
        )

    # 2. Layer on Explicit CLI Flags (Medium priority)
    # Only update if the user actually passed the flag

    for section, key, value in explicit_tuples:
        update_nested(section, key, value)

    # 3. Layer on Extra "Hydra-style" args (Highest priority)
    # e.g., masking.sigma=5.0
    for key, val in extra_overrides.items():
        if isinstance(val, dict) and key in final_payload:
            final_payload[key].update(val)
        else:
            final_payload[key] = val

    # 4. Instantiate & Validate
    try:
        # Pydantic will now raise an error if map_dark/pdb_dark are still missing
        # after merging YAML and CLI flags.
        settings = Settings(**final_payload)

        typer.secho(
            f"🚀 Starting pipeline: {settings.general.name_human}",
            fg=typer.colors.GREEN,
            bold=True,
        )

    except Exception as e:
        typer.secho("\n❌ Configuration Error:", fg=typer.colors.RED, bold=True)
        typer.echo(e)
        raise typer.Exit(code=1)
    return settings


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    ctx: typer.Context,
    profile: Optional[Path] = typer.Option(
        None, "--from_yaml", help="Load an override YAML"
    ),
    # --- General Settings (None as default to allow YAML overrides) ---
    name: Optional[str] = typer.Option(None, "--name"),
    dmin: Optional[float] = typer.Option(
        None, "--dmin", help="Set high resolution limit"
    ),
    diffmap_type: Optional[Literal["tv", "vanilla", "kweighted"]] = typer.Option(
        None, "--diffmap_type", help="Choose diffmap type (tv, vanilla, kweighted)"
    ),
    # --- Input File Settings (Optional in Typer, Mandatory in Pydantic) ---
    map_dark: Optional[Path] = typer.Option(
        None, "--map_dark", help="Path to dark MTZ"
    ),
    pdb_dark: Optional[Path] = typer.Option(
        None, "--pdb_dark", help="Path to dark PDB"
    ),
    map_trig: Optional[Path] = typer.Option(
        None, "--map_triggered", help="Path to triggered MTZ"
    ),
    map_diff: Optional[Path] = typer.Option(
        None, "--map_diff", help="Path to difference MTZ"
    ),
):
    """
    Run analysis. Settings are merged in order:
    Pydantic Defaults -> YAML Profile -> CLI Flags -> Extra Dot-notation Args
    """

    explicit_tuples = [
        ("general", "name_machine", name),
        ("general", "high_resolution_limit", dmin),
        ("map_processing", "diffmap_type", diffmap_type),
        ("input_files", "map_dark", map_dark),
        ("input_files", "pdb_dark", pdb_dark),
        ("input_files", "map_triggered", map_trig),
        ("input_files", "map_diff", map_diff),
    ]
    extra_overrides = parse_extra_args(ctx.args)

    settings = parse_settings(profile, explicit_tuples, extra_overrides)
    # 5. Save and Execute
    config = dump_config(settings)
    execute_as_main(config, save2file=True)


def main():
    typer.echo("Welcome to the XTR Estimator CLI!")
    typer.echo("Use --help for available options.")
    app()


if __name__ == "__main__":
    main()
