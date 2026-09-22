#!/usr/bin/env python3
"""
Sunburst Hierarchy Viewer
==========================

An interactive, Filelight-style radial pie chart ("sunburst chart") for
visualizing the size of hierarchically labeled data.

Labels look like:  "1.2.3.4"  ->  class 1 > subclass 2 > sub-subclass 3 > item 4

Each ring of the chart is one level of the hierarchy. The size (angular
width) of a wedge is proportional to the total size of everything beneath
it, exactly like Filelight/Disk-usage-analyzers show folder sizes.

Controls
--------
- Left click a wedge   : zoom in (make that wedge's subtree fill the circle)
- Left click the center hole, or press the "Back" button : zoom out one level
- Hover over a wedge    : see its full path and size in the status bar
- "Reset" button        : go back to the top-level view

How to supply your own data
----------------------------
Edit the `load_data()` function near the bottom of this file. It just needs
to return a dict of {label: size}, e.g.:

    {
        "1.1.1": 120,
        "1.1.2": 340,
        "1.2.1": 80,
        "2.1.1": 900,
    }

Sizes only need to be given for the *leaf* labels; parent group sizes are
computed automatically by summing their children (just like folder sizes
on disk). If you also want to attach a size directly to an intermediate
label (e.g. "1.2" itself has some size in addition to its children "1.2.1"
etc.), that is supported too -- it's added as the node's own "self size".

You can also load data from a two-column CSV file (label,size) using the
`load_from_csv()` helper.

Requirements: Python 3, matplotlib, tkinter (both ship with most standard
Python installations).
"""

import csv
import random
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors

# --------------------------------------------------------------------------
# Tree construction
# --------------------------------------------------------------------------


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
        # Largest first tends to look nicer, but alphabetical/numeric is
        # more predictable for browsing -- sort numerically/alphabetically.
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


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


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


def generate_sample_data(seed=42):
    """Generates some random hierarchical demo data so the app is runnable
    out of the box. Replace load_data() below with your real data."""
    rng = random.Random(seed)
    data = {}
    n_top = 5
    for i in range(1, n_top + 1):
        n_sub = rng.randint(2, 4)
        for j in range(1, n_sub + 1):
            n_subsub = rng.randint(1, 4)
            for k in range(1, n_subsub + 1):
                if rng.random() < 0.6:
                    n_leaf = rng.randint(1, 3)
                    for l in range(1, n_leaf + 1):
                        label = f"{i}.{j}.{k}.{l}"
                        data[label] = rng.randint(10, 500)
                else:
                    label = f"{i}.{j}.{k}"
                    data[label] = rng.randint(10, 500)
    return data


def load_data():
    """Return a dict of {label: size}. Two ready-made options are shown below."""
    path = Path(__file__).parent / "swissProtEnzyme_sizes.csv"
    return load_from_csv(path)


# --------------------------------------------------------------------------
# Color handling
# --------------------------------------------------------------------------


def make_color(base_hue, depth, max_depth):
    """Shades a base hue lighter as depth increases, Filelight-style."""
    lightness = 0.35 + 0.5 * (depth / max(max_depth, 1))
    lightness = min(lightness, 0.88)
    r, g, b = matplotlib.colors.hsv_to_rgb((base_hue, 0.65, 1.0))
    # blend toward white by `lightness` to get a lighter shade per ring
    r = r + (1 - r) * lightness
    g = g + (1 - g) * lightness
    b = b + (1 - b) * lightness
    return (r, g, b)


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------


class SunburstApp:
    RING_WIDTH = 1.0
    MAX_RINGS_SHOWN = 5  # rings visible at once from the current root

    def __init__(self, root_tk, tree_root):
        self.root_tk = root_tk
        self.data_root = tree_root
        self.view_stack = [tree_root]  # last element = current center node
        self.wedge_info = []  # list of (wedge_patch, node, depth)

        self._build_ui()
        self._assign_top_level_hues()
        self.draw()

    # ---- UI scaffolding ----

    def _build_ui(self):
        self.root_tk.title("Sunburst Hierarchy Viewer")
        self.root_tk.geometry("900x800")

        toolbar = ttk.Frame(self.root_tk)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.back_btn = ttk.Button(toolbar, text="\u2190 Back", command=self.zoom_out)
        self.back_btn.pack(side=tk.LEFT)

        self.reset_btn = ttk.Button(toolbar, text="Reset", command=self.zoom_reset)
        self.reset_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.breadcrumb_var = tk.StringVar(value="/")
        breadcrumb = ttk.Label(toolbar, textvariable=self.breadcrumb_var, font=("TkDefaultFont", 11, "bold"))
        breadcrumb.pack(side=tk.LEFT, padx=16)

        self.fig = plt.Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root_tk)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Hover over a segment to see details")
        status_bar = ttk.Label(self.root_tk, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def _assign_top_level_hues(self):
        children = self.data_root.sorted_children()
        n = max(len(children), 1)
        self.top_level_hue = {}
        for idx, child in enumerate(children):
            self.top_level_hue[child.path] = idx / n

    def _hue_for(self, node):
        # Walk up from node to find which top-level branch it belongs to
        parts = node.path.split(".") if node.path else []
        if not parts:
            return 0.0
        top_path = parts[0]
        return self.top_level_hue.get(top_path, 0.0)

    # ---- Drawing ----

    @property
    def current_root(self):
        return self.view_stack[-1]

    def draw(self):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.wedge_info = []

        root = self.current_root
        max_extent = 1 + self.MAX_RINGS_SHOWN
        limit = max_extent * self.RING_WIDTH
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        # center hub (click to go back)
        hub_radius = self.RING_WIDTH * 0.9
        hub = Wedge((0, 0), hub_radius, 0, 360, facecolor="#dddddd", edgecolor="white")
        self.ax.add_patch(hub)
        label = root.name if root.name else "root"
        self.ax.text(0, 0, label, ha="center", va="center", fontsize=10, weight="bold")
        self.wedge_info.append((hub, root, 0))

        if root.total > 0:
            self._draw_children(root, theta_start=90, theta_span=360, depth=1, inner_r=self.RING_WIDTH)

        self.breadcrumb_var.set("/" + "/".join(n.name for n in self.view_stack if n.name) or "/")
        self.canvas.draw_idle()

    def _draw_children(self, node, theta_start, theta_span, depth, inner_r):
        if depth > self.MAX_RINGS_SHOWN:
            return
        children = node.sorted_children()
        total = node.total if node.total > 0 else 1
        theta = theta_start
        outer_r = inner_r + self.RING_WIDTH
        for child in children:
            frac = child.total / total
            span = frac * theta_span
            if span <= 0:
                continue
            hue = self._hue_for(child)
            color = make_color(hue, depth, self.MAX_RINGS_SHOWN)
            wedge = Wedge(
                (0, 0),
                outer_r,
                theta,
                theta + span,
                width=self.RING_WIDTH,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )
            self.ax.add_patch(wedge)
            self.wedge_info.append((wedge, child, depth))

            # label if the wedge is wide enough to read
            if span > 6:
                mid_angle = theta + span / 2
                import math

                rad = math.radians(mid_angle)
                label_r = inner_r + self.RING_WIDTH / 2
                x, y = label_r * math.cos(rad), label_r * math.sin(rad)
                rot = mid_angle - 90 if 90 < mid_angle < 270 else mid_angle + 90
                self.ax.text(x, y, child.name, ha="center", va="center", fontsize=7, rotation=rot if -90 <= (mid_angle % 360) - 180 else mid_angle, rotation_mode="anchor")

            self._draw_children(child, theta, span, depth + 1, outer_r)
            theta += span

    # ---- Interaction ----

    def _wedge_at(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        import math

        r = math.hypot(event.xdata, event.ydata)
        angle = math.degrees(math.atan2(event.ydata, event.xdata)) % 360
        for wedge, node, depth in reversed(self.wedge_info):
            theta1, theta2 = wedge.theta1, wedge.theta2
            width = wedge.width if wedge.width is not None else wedge.r
            r_inner = wedge.r - width
            r_outer = wedge.r
            if r_inner <= r <= r_outer:
                a1, a2 = theta1 % 360, theta2 % 360
                if a1 <= a2:
                    in_range = a1 <= angle <= a2
                else:
                    in_range = angle >= a1 or angle <= a2
                if in_range or depth == 0:  # depth 0 = hub, covers small r regardless of angle
                    if depth == 0 and r > r_outer:
                        continue
                    return node, depth
        return None

    def on_click(self, event):
        hit = self._wedge_at(event)
        if hit is None:
            return
        node, depth = hit
        if depth == 0:
            self.zoom_out()
            return
        if node.children:
            self.view_stack.append(node)
            self.draw()
        else:
            self.status_var.set(f"{node.path}  —  size {node.total:,.0f} (leaf, no sub-items)")

    def on_hover(self, event):
        hit = self._wedge_at(event)
        if hit is None:
            self.status_var.set("Hover over a segment to see details")
            return
        node, depth = hit
        if depth == 0:
            self.status_var.set(f"Center: {node.path or 'root'}  —  total size {node.total:,.0f}")
        else:
            pct = 100 * node.total / self.current_root.total if self.current_root.total else 0
            self.status_var.set(f"{node.path}   size={node.total:,.0f}   ({pct:.1f}% of current view)")

    def zoom_out(self):
        if len(self.view_stack) > 1:
            self.view_stack.pop()
            self.draw()

    def zoom_reset(self):
        self.view_stack = [self.data_root]
        self.draw()


def main():
    label_sizes = load_data()
    if not label_sizes:
        raise SystemExit("No data loaded -- edit load_data() to supply labels and sizes.")
    tree = build_tree(label_sizes)

    root_tk = tk.Tk()
    app = SunburstApp(root_tk, tree)
    root_tk.mainloop()


if __name__ == "__main__":
    main()
