#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "template"
DEFAULT_TEMPLATE = "k6_frac_N10_mem32K_40nm.xml.j2"


def bake_layout(
    benchmark_name: str,
    dsps: list[tuple[int, int]],
    mems: list[tuple[int, int]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: Optional[str] = None,
    aspect_ratio: Optional[float] = None,
    template_name: str = DEFAULT_TEMPLATE,
) -> Union[Path, int]:
    """
    Render DSP/BRAM placements into a VTR architecture XML via Jinja2.

    The coordinates are 1-indexed and include the IO ring (height includes the IO perimeter).
    Returns the output Path on success, or -1 if the placement is geometrically invalid.
    """
    if benchmark_name.endswith(".v"):
        benchmark_name = benchmark_name[:-2]

    if height is not None:
        if any(y > height - 5 for _, y in dsps):
            print("Invalid placement: DSP y > height - 5. Aborting.", file=sys.stderr)
            return -1
        if any(y > height - 7 for _, y in mems):
            print("Invalid placement: BRAM y > height - 7. Aborting.", file=sys.stderr)
            return -1

    template_path = TEMPLATE_DIR / template_name
    if not template_path.is_file():
        raise FileNotFoundError(f"Jinja2 template not found: {template_path}")

    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = jinja_env.get_template(template_name)

    rendered = template.render(
        layout_name="my_layout",
        width=width,
        height=height,
        dsps=dsps,
        mems=mems,
        aspect_ratio=aspect_ratio,
    )

    dest = Path(output_path).resolve() if output_path else Path.cwd() / "baked_layout_arch.xml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(rendered)

    print(f"Baked architecture → {dest}")
    print(f"  Size   : {width}x{height}" if width and height else "  Size   : auto_layout")
    print(f"  DSPs   : {dsps}")
    print(f"  BRAMs  : {mems}")

    return dest
