"""Render the archived Validation metrics to SVG without rerunning the experiment."""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLORS = ("#1d4ed8", "#087f75", "#a34713", "#8248a0", "#546376")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def chart(filename, title, x_title, y_title, labels, series, *, ceiling=None, style="bars"):
    width, height = 920, 470
    left, top, plot_w, plot_h = 92, 77, 784, 286
    max_value = ceiling if ceiling is not None else max(max(values) for _, values in series) * 1.12
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 470" role="img">',
           f"<title>{escape(title)}</title>",
           f'<rect width="{width}" height="{height}" fill="#fafbfc"/>']

    def text(x, y, label, anchor="start", size=13, color="#1a2637"):
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
                   f'text-anchor="{anchor}" font-family="system-ui,sans-serif">{escape(str(label))}</text>')

    text(left, 34, title, size=21)
    text(left, 56, "Development Validation · 36 tasks · A2 and V3 development", color="#546376")
    for n in range(5):
        value = max_value * n / 4
        y = top + plot_h * (1 - n / 4)
        svg.append(f'<path d="M {left},{y} h {plot_w}" stroke="#d8dfe8"/>')
        text(left - 9, y + 4, f"{value:.2f}" if max_value <= 1 else f"{value:.1f}", "end", 12)
    svg.append(f'<g transform="translate(19 {top + plot_h/2}) rotate(-90)">')
    text(0, 0, y_title, "middle")
    svg.append("</g>")
    step = plot_w / len(labels)
    centers = [left + step * (i + .5) for i in range(len(labels))]
    for x, label in zip(centers, labels, strict=True):
        text(x, top + plot_h + 21, label, "middle", 12)
    text(left + plot_w / 2, top + plot_h + 45, x_title, "middle")
    for index, (name, values) in enumerate(series):
        color = COLORS[index]
        points = []
        for x, value in zip(centers, values, strict=True):
            bar_h = plot_h * value / max_value
            y = top + plot_h - bar_h
            if style == "line":
                points.append((x, y))
            else:
                bar_w = step * .72 / len(series)
                x += (index - len(series) / 2) * bar_w
                svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{bar_h:.1f}" fill="{color}"><title>{escape(name)} {value:.3f}</title></rect>')
        if points:
            line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{line}"/>')
            for (x, y), value in zip(points, values, strict=True):
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"><title>{escape(name)} {value:.3f}</title></circle>')
        x = left + index * 153
        svg.append(f'<rect x="{x}" y="422" width="14" height="11" fill="{color}"/>')
        text(x + 20, 432, name, size=12)
    text(left, 456, "Source: results/a2-memory-v2/summary.json · proxy labels", size=11, color="#546376")
    svg.append("</svg>")
    (ROOT / filename).write_text("\n".join(svg) + "\n", encoding="utf-8")


def main():
    archive = load("summary.json")
    rankings = archive["ranking"]["groups"]
    compressed = archive["compression"]["groups"]
    variants = ["R0", "R1", "R2", "R3", "R4"]
    chart("ranking-quality.svg", "Memory ranking quality", "Ranking mode · fixed E0",
          "Macro metric (0–1)", variants,
          [(metric, [rankings[f"validation/E0/{r}"][metric] for r in variants])
           for metric in ("recall@1", "recall@3", "ndcg@3")], ceiling=1)
    chart("ranking-latency.svg", "Warm memory query latency", "Ranking mode · fixed E0",
          "Latency (ms)", variants,
          [("Mean", [rankings[f"validation/E0/{r}"]["latency_ms"] for r in variants]),
           ("P95", [rankings[f"validation/E0/{r}"]["p95_ms"] for r in variants])])
    modes = archive["selection"]["compression_modes"]
    budgets = [900, 1800, 3600]
    chart("budget-retention.svg", "Relevant experience delivered by budget", "Context cap (UTF-8 bytes)",
          "Macro delivery recall (0–1)", budgets,
          [(mode, [compressed[f"validation/{mode}/{budget}"]["memory_delivery_recall"]
                   for budget in budgets]) for mode in modes], style="line", ceiling=1)
    chart("compression-delivery.svg", "Context size and experience delivery", "Compression variant · 1800 bytes",
          "Fraction of reference (0–1.25)", modes,
          [("Bytes / raw bytes", [1 - compressed[f"validation/{mode}/1800"]["byte_reduction"] for mode in modes]),
           ("Delivered / selected", [compressed[f"validation/{mode}/1800"]["delivery_fraction"] for mode in modes])],
          ceiling=1.25)
    stages = ("relevant", "recalled_relevant", "injected_relevant")
    chart("delivery-loss.svg", "Relevant experience by stage", "Stage",
          "Relevant episode-query pairs (count)", ["PIT candidate pool", "Selected Top-3", "Delivered"],
          [(mode, [compressed[f"validation/{mode}/1800"][key] for key in stages])
           for mode in ("C0", "C1", "C3", "C4")])


if __name__ == "__main__":
    main()
