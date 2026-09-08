"""Generate a diagram of the browser automation LangGraph.

This utility only builds the graph topology. It does not start Selenium or call
Groq. Mermaid text is generated locally; PNG rendering is optional and may use
LangGraph's remote Mermaid rendering service.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from selenium.webdriver.remote.webdriver import WebDriver

# This file is a standalone utility, not part of an installed Python package.
# Resolve imports relative to the file so it works from the repository root,
# this directory, or any other current working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.graphflow import BrowserRuntime, build_browser_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the browser automation LangGraph as Mermaid."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("browser_flow.mmd"),
        help=(
            "Mermaid output path (default: browser_flow.mmd next to this "
            "script)."
        ),
    )
    parser.add_argument(
        "--png",
        type=Path,
        help="Optionally render a PNG to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Building the graph only stores this runtime dependency in node closures;
    # none of the nodes are executed while exporting the topology.
    runtime = BrowserRuntime(
        driver=cast(WebDriver, object()),
        api_key=None,
    )
    compiled_graph = build_browser_graph(runtime)
    drawable_graph = compiled_graph.get_graph()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(drawable_graph.draw_mermaid(), encoding="utf-8")
    print(f"Mermaid diagram written to: {args.output.resolve()}")

    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        try:
            drawable_graph.draw_mermaid_png(output_file_path=str(args.png))
        except Exception as exc:
            raise SystemExit(
                "Mermaid text was generated, but PNG rendering failed. "
                "PNG rendering may require internet access to the Mermaid API. "
                f"Original error: {exc}"
            ) from exc
        print(f"PNG diagram written to: {args.png.resolve()}")


if __name__ == "__main__":
    main()
