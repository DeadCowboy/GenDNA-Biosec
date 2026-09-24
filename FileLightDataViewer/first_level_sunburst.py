#!/usr/bin/env python3
"""
First Level Sunburst Diagram
============================

A simplified sunburst chart showing only the first level of the hierarchy.
Optimized for presentation-quality images with large, readable labels.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for better image quality
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
import csv
from pathlib import Path


class Node:
    """A single node in the label hierarchy."""

    __slots__ = ("name", "path", "self_size", "children", "total")

    def __init__(self, name, path):
        self.name = name  # this node's own segment, e.g. "3"
        self.path = path  # full dotted path, e.g. "1.2.3"
        self.self_size = 0.0  # size attached directly to this label
        self.children = {}  # name -> Node
        self.total = 0.0  # self_size + sum(children totals)

    def get_child(self, name):
        child = self.children.get(name)
        if child is None:
            child_path = f"{self.path}.{name}" if self.path else name
            child = Node(name, child_path)
            self.children[name] = child
        return child

    def compute_totals(self):
        total = self.self_size
        for child in self.children.values():
            total += child.compute_totals()
        self.total = total
        return total

    def sorted_children(self):
        # Sort numerically/alphabetically
        def sort_key(item):
            name = item[0]
            try:
                return (0, float(name))
            except ValueError:
                return (1, name)

        return [child for _, child in sorted(self.children.items(), key=sort_key)]


def build_tree(label_sizes):
    """label_sizes: dict of {"1.2.3": size, ...} -> root Node."""
    root = Node(name="", path="")
    for label, size in label_sizes.items():
        parts = str(label).split(".")
        node = root
        for part in parts:
            node = node.get_child(part)
        node.self_size += float(size)
    root.compute_totals()
    return root


def load_from_csv(path):
    """Load a two-column CSV file: label,size (no header, or a header row
    that doesn't parse as a number is skipped automatically)."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            label, size = row[0].strip(), row[1].strip()
            try:
                data[label] = float(size)
            except ValueError:
                continue  # probably a header row
    return data


def load_data():
    """Return a dict of {label: size}. Load from CSV file."""
    path = Path(__file__).parent / "swissProtEnzyme_sizes.csv"
    return load_from_csv(path)


def make_color(base_hue, depth=1):
    """Create colors for the wedges - just first level so depth is always 1."""
    lightness = 0.45  # Consistent lightness for all wedges
    r, g, b = matplotlib.colors.hsv_to_rgb((base_hue, 0.7, 1.0))
    # Blend toward white for better visibility
    r = r + (1 - r) * lightness
    g = g + (1 - g) * lightness
    b = b + (1 - b) * lightness
    return (r, g, b)


def create_first_level_sunburst(label_sizes, output_path="first_level.png"):
    """Create a sunburst diagram showing only the first level of the hierarchy."""
    tree = build_tree(label_sizes)

    # Create figure with high resolution for presentation
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    # Remove margins and set proper limits
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Set up the rings
    INNER_RADIUS = 0.3
    OUTER_RADIUS = INNER_RADIUS + 0.8  # Total radius is 1.1

    # Get first level children
    first_level_children = tree.sorted_children()
    if not first_level_children:
        print("No first-level children found")
        return

    n_children = len(first_level_children)

    # Calculate total size for percentage calculations
    total_size = sum(child.total for child in first_level_children)

    # Draw wedges for first level only
    theta = 90  # Start from top (12 o'clock)

    for i, child in enumerate(first_level_children):
        # Calculate wedge angle
        frac = child.total / total_size
        span = frac * 360

        if span <= 0:
            continue

        # Create color
        hue = i / n_children
        color = make_color(hue)

        # Create wedge with proper parameters
        wedge = Wedge((0, 0), OUTER_RADIUS, theta, theta + span, width=OUTER_RADIUS - INNER_RADIUS, facecolor=color, edgecolor="white", linewidth=1.2)
        ax.add_patch(wedge)

        # Add large, readable label if wedge is wide enough
        if span > 5:  # Lower threshold to show more labels
            mid_angle = theta + span / 2
            import math

            rad = math.radians(mid_angle)
            label_r = INNER_RADIUS + (OUTER_RADIUS - INNER_RADIUS) * 0.6  # Position label closer to center

            # Position label at center of wedge
            x, y = label_r * math.cos(rad), label_r * math.sin(rad)

            # Add label with percentage and absolute count
            percentage = 100 * child.total / total_size
            label_text = f"{child.name}\n{percentage:.1f}%\n({child.total:,.0f})"

            # Add text with large font size for presentation, centered and straight
            ax.text(
                x,
                y,
                label_text,
                ha="center",  # Center horizontally
                va="center",  # Center vertically
                fontsize=20,  # Large font for presentation
                fontweight="bold",
                rotation=0,  # No rotation - keep text straight
                color="black",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3, boxstyle="round,pad=0.3"),
            )

        theta += span

    # Add title with increased font size
    plt.title("First Level EC Group Distribution", fontsize=28, pad=30, fontweight="bold")

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    print(f"Sunburst saved to {output_path}")

    plt.close(fig)


def main():
    label_sizes = load_data()
    if not label_sizes:
        raise SystemExit("No data loaded -- edit load_data() to supply labels and sizes.")

    create_first_level_sunburst(label_sizes)
    print(f"Generated sunburst diagram with {len(label_sizes)} leaf nodes")


if __name__ == "__main__":
    main()
