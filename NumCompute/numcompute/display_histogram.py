from html import escape
import numpy as np

def _running_in_notebook() -> bool:
    """Return whether the current IPython shell is a Jupyter notebook."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"

def _notebook_histogram_html(labels, heights, density) -> str:
    """Build a conventional, high-contrast vertical histogram for notebooks."""
    chart_width = 760
    chart_height = 390
    left, right, top, bottom = 55, 15, 20, 105
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom
    plot_bottom = top + plot_height
    maximum_height = max(heights, default=0)
    y_max = maximum_height if maximum_height > 0 else 1
    bar_width = plot_width / len(heights)
    # Use the same evenly spaced y-axis marks for either counts or density.
    y_ticks = np.linspace(0, y_max, 5)

    grid = []
    for tick in y_ticks:
        y = plot_bottom - tick / y_max * plot_height
        label = f"{tick:.3g}" if density else f"{tick:.0f}"
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{chart_width - right}' y2='{y:.1f}' "
            "stroke='black' stroke-width='1'/>"
            f"<text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' "
            f"font-size='13' font-weight='600' fill='black'>{label}</text>"
        )

    bars = []
    for index, (label, height) in enumerate(zip(labels, heights)):
        # All bins have equal width and touch, as in a standard histogram.
        x = left + index * bar_width
        bar_height = height / y_max * plot_height
        y = plot_bottom - bar_height
        tick_x = x + bar_width / 2
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width + 0.2:.2f}' "
            f"height='{bar_height:.2f}' fill='lightblue' stroke='white' stroke-width='1'>"
            f"<title>{escape(label)}: {height:.3g}</title></rect>"
            f"<text x='{tick_x:.2f}' y='{plot_bottom + 16}' "
            f"transform='rotate(45 {tick_x:.2f} {plot_bottom + 16})' "
            f"font-size='12' font-weight='600' fill='black'>{escape(label)}</text>"
        )

    y_axis_label = "Density" if density else "Count"
    return (
        "<div style='max-width:760px;background:white;color:black;"
        "padding:12px;border:1px solid black;border-radius:6px'>"
        "<strong style='color:black'>Histogram</strong>"
        f"<svg viewBox='0 0 {chart_width} {chart_height}' role='img' "
        "aria-label='Vertical histogram' "
        "style='display:block;width:100%;background:white;color:black'>"
        + "".join(grid)
        + "".join(bars)
        + f"<line x1='{left}' y1='{top}' x2='{left}' y2='{plot_bottom}' stroke='black' stroke-width='2'/>"
        + f"<line x1='{left}' y1='{plot_bottom}' x2='{chart_width - right}' "
        + f"y2='{plot_bottom}' stroke='black' stroke-width='2'/>"
        + f"<text x='{(left + chart_width - right) / 2}' y='{chart_height - 8}' "
        + "text-anchor='middle' font-size='14' font-weight='600' fill='black'>"
        + "Value range</text>"
        + f"<text x='15' y='{top + plot_height / 2}' text-anchor='middle' "
        + "font-size='14' font-weight='600' fill='black' "
        + f"transform='rotate(-90 15 {top + plot_height / 2})'>"
        + f"{y_axis_label}</text></svg>"
        "</div>"
    )


def _display_notebook_histogram(labels, heights, density) -> bool:
    """Display labeled bins inline in Jupyter; return False outside notebooks."""
    if not _running_in_notebook():
        return False

    from IPython.display import HTML, display

    display(HTML(_notebook_histogram_html(labels, heights, density)))
    return True


def display_histogram(heights, edges, density=False) -> bool:
    """Display precomputed histogram values in a notebook, if available.

    ``heights`` contains counts, or densities when ``density=True``. Returns
    ``True`` if displayed in Jupyter and ``False`` in other environments.
    """
    heights = np.asarray(heights)
    edges = np.asarray(edges, dtype=float)
    if heights.ndim != 1 or edges.ndim != 1 or edges.size != heights.size + 1:
        raise ValueError("edges must contain exactly one more value than heights.")

    labels = [
        f"{left:.3g} to {right:.3g}" for left, right in zip(edges[:-1], edges[1:])
    ]
    return _display_notebook_histogram(labels, heights.tolist(), density)