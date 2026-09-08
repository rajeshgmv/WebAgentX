"""Generate a diagram of the browser automation LangGraph.

This utility only builds the graph topology. It does not start Selenium or call
Groq. Mermaid text is generated locally; PNG rendering is optional and may use
LangGraph's remote Mermaid rendering service.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from selenium.webdriver.remote.webdriver import WebDriver

from graphflow import BrowserRuntime, build_browser_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the browser automation LangGraph as Mermaid."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("browser_flow.mmd"),
        help="Mermaid output path (default: browser_flow.mmd).",
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
