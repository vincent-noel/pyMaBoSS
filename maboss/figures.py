"""Tools to draw figures from the MaBoSS results."""

import operator
import sys
import textwrap
import warnings
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
import matplotlib as mpl

def persistent_color(palette, state):

    reordered = " -- ".join(sorted(state.split(" -- ")))
    if state in palette:
        return palette[state]
    if reordered in palette:
        return palette[reordered]
    if state == "Others":
        color = "lightgray"
    else:
        count = len(palette)
        if "Others" not in palette:
            count += 1
        color = "C%d" % (count % 10)
    palette[state] = color
    palette[reordered] = color
    return color

def register_states_for_color(palette, collection):
    [persistent_color(palette, state) for state in collection]

def make_plot_trajectory(time_table, ax, palette, legend=True, error_table=None):
    register_states_for_color(palette, time_table.columns.values)
    color_list = [persistent_color(palette, idx) \
                    for idx in time_table.columns.values]
    time_table.plot(ax=ax, color=color_list, legend=legend, yerr=error_table)
    if legend:
        plt.legend(time_table.columns, loc=(1.1, 0))


def plot_node_prob(time_table, ax, palette, legend=True, error_table=None):
    """Plot the probability of each node being up over time."""
    
    register_states_for_color(palette, time_table.columns.values)
    color_list = [persistent_color(palette, idx) \
                    for idx in time_table.columns.values]
    if len(time_table.columns) == 0:
        print("No data to plot !", file=sys.stderr)
        
    else:        
        if ax is None:
            _, ax = plt.subplots(1,1)
        
    
        time_table.plot(ax=ax, color=color_list, legend=legend, yerr=error_table)
        if not np.any(time_table):
            plt.ylim(0, 1)
        if legend:
            plt.legend(time_table.columns, loc='upper right')
        return ax


def plot_piechart(plot_table, ax, palette, embed_labels=False, autopct=4, \
                    prob_cutoff=0.01, legend=True, nil_label=None):
    plot_line = plot_table.iloc[-1].rename("")  # Takes the last time point

    others = plot_line[plot_line <= prob_cutoff].sum()

    plot_line = plot_line[plot_line > prob_cutoff]
    plot_line.sort_values(ascending=False, inplace=True)

    if others:
        plot_line.at["Others"] = others

    register_states_for_color(palette, plot_line.index)

    plotting_labels = []
    color_list = []
    for value_index, value in enumerate(plot_line):
        state = plot_line.index.values[value_index]
        color_list.append(persistent_color(palette, state))
        if embed_labels and value >= 0.1:
            if state == "<nil>" and nil_label is not None:
                plotting_labels.append(nil_label)
            else:
                plotting_labels.append(state)
        else:
            plotting_labels.append("")

    opts = {}
    if autopct:
        cutoff = autopct if type(autopct) is not bool else 4
        opts.update(autopct=lambda p: '%1.1f%%' % p if p >= cutoff else "")
    else:
        opts.update(labeldistance=0.4)

    if others:
        plot_line = plot_line.rename({"Others": "Others (%1.2f%%)" % (others*100)})

    ax.pie(plot_line, labels=plotting_labels, radius=1.2,
           startangle=90, colors=color_list, **opts)
    ax.axis('equal')
    if legend:
        if nil_label is not None:
            ax.legend([nil_label if label == "<nil>" else label for label in plot_line.index.values], loc=(0.9, 0.2))
        else:
            ax.legend(plot_line.index.values, loc=(0.9, 0.2))


def plot_fix_point(table, ax, palette):
    """Plot a piechart representing the fixed point probability."""
    palette['no_fp'] = '#121212'
    prob_list = []
    color_list = []
    labels = []
    for i in range(len(table)):
        prob = table['Proba'][i]
        state = table['State'][i]
        prob_list.append(prob)
        color_list.append(persistent_color(palette, state))
        labels.append('FP '+str(i+1))
    prob_ns = 1 - sum(prob_list)
    if prob_ns > 0.01:
        prob_list.append(prob_ns)
        color_list.append(persistent_color(palette, "no_fp"))
        labels.append('no_fp')
    ax.pie(prob_list, labels=labels, colors=color_list)

# Colors of the transitions, from the least to the most probable (light to dark blue, as in WebMaBoSS)
OBSERVED_GRAPH_LOW_COLOR = np.array([198, 219, 239]) / 255
OBSERVED_GRAPH_HIGH_COLOR = np.array([8, 48, 107]) / 255


def _proba_color(proba):
    return tuple(OBSERVED_GRAPH_LOW_COLOR + (OBSERVED_GRAPH_HIGH_COLOR - OBSERVED_GRAPH_LOW_COLOR) * proba)


def _format_proba(proba):
    return "%.1f%%" % (proba * 100) if proba < 0.1 else "%.0f%%" % (proba * 100)


def _observed_nodes(states):
    """Return the observed nodes, from the names of the states, in order of first appearance."""
    nodes = []
    for state in states:
        for node in state.split(" -- "):
            if node != "<nil>" and node not in nodes:
                nodes.append(node)
    return nodes


def _observed_graph_layout(G, layout):
    if layout != "spring":
        try:
            import pygraphviz
            # Graphviz is told the size of the boxes, so that it keeps them apart. "<...>" would be an HTML label
            H = nx.DiGraph(graph={"overlap": "false", "sep": "+12", "start": "regular", "model": "subset"})
            H.add_nodes_from(
                (state, {"label": state.replace(" -- ", "\\n").replace("<", "(").replace(">", ")"),
                         "shape": "box", "fontsize": 14})
                for state in G
            )
            H.add_edges_from(G.edges)
            return nx.drawing.nx_agraph.graphviz_layout(H, prog=layout)
        except ImportError:
            pass
        except OSError as e:
            # Graphviz installed without this layout program
            warnings.warn("Graphviz layout %s failed, using spring layout instead: %s" % (layout, str(e).strip()))
    return nx.spring_layout(G, seed=0)


def _graph_scale(xy, offsets, width, height, gap):
    """Return the scale (pixels per data unit) and the zoom at which the graph fits in width x height pixels.

    offsets are the borders of the boxes relative to the positions xy, in pixels. We look for the largest scale
    at which the states and their boxes fit. If at this scale the boxes are closer than gap pixels,
    we zoom out instead: the boxes shrink with the distances between them.
    """
    def size(scale, offsets):
        return (
            np.max(scale * xy[:, 0] + offsets[:, 2]) - np.min(scale * xy[:, 0] + offsets[:, 0]),
            np.max(scale * xy[:, 1] + offsets[:, 3]) - np.min(scale * xy[:, 1] + offsets[:, 1]),
        )

    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    if span == 0:
        return 1.0, 1.0

    # The size of the graph grows with the scale: bisection
    low, high = 0, max(width, height) / span
    for _ in range(60):
        scale = (low + high) / 2
        graph_width, graph_height = size(scale, offsets)
        if graph_width <= width and graph_height <= height:
            low = scale
        else:
            high = scale

    # Smallest scale at which each pair of boxes is separated by the gap, horizontally or vertically
    half_sizes = (offsets[:, 2:] - offsets[:, :2]) / 2
    distances = np.abs(xy[:, np.newaxis, :] - xy[np.newaxis, :, :])
    needed = half_sizes[:, np.newaxis, :] + half_sizes[np.newaxis, :, :] + gap
    with np.errstate(divide="ignore"):
        pair_scales = np.min(needed / distances, axis=2)
    separated_scale = np.max(pair_scales[np.triu_indices(len(xy), 1)]) if len(xy) > 1 else 0

    if low >= separated_scale:
        return low, 1.0
    graph_width, graph_height = size(separated_scale, offsets)
    zoom = min(width / graph_width, height / graph_height)
    return zoom * separated_scale, zoom


def _fit_observed_graph(ax, pos, boxes, legend_height, pad=10, gap=70):
    """Rotate the layout and set the limits of the axes, with an equal aspect, so that the boxes of the states
    are not clipped and the graph is as large as possible, above the legend (legend_height pixels).
    Return the new positions, and the zoom to apply to the sizes of the texts and lines.
    """
    # Sizes the boxes of the labels
    ax.get_figure().draw_without_rendering()

    states = list(boxes)
    xy = np.array([pos[state] for state in states], dtype=float)
    anchors = ax.transData.transform(xy)
    extents = np.array([boxes[state].get_bbox_patch().get_window_extent().get_points().ravel() for state in states])
    # Borders of the boxes relative to the positions of the states, in pixels: (left, bottom, right, top)
    offsets = extents - np.hstack([anchors, anchors])

    axes_box = ax.get_window_extent()
    width = axes_box.width - 2 * pad
    height = axes_box.height - legend_height - 2 * pad

    # The orientation of the layout is arbitrary: we keep the one where the graph is the least zoomed out
    best = None
    for angle in np.radians(np.arange(0, 180, 15)):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = (xy - xy.mean(axis=0)) @ rotation.T
        scale, zoom = _graph_scale(rotated, offsets, width, height, gap)
        if best is None or (round(zoom, 3), scale) > (round(best[2], 3), best[1] * 1.001):
            best = (rotated, scale, zoom)
    xy, scale, zoom = best
    offsets = zoom * offsets

    left = np.min(scale * xy[:, 0] + offsets[:, 0])
    right = np.max(scale * xy[:, 0] + offsets[:, 2])
    lower = np.min(scale * xy[:, 1] + offsets[:, 1])
    upper = np.max(scale * xy[:, 1] + offsets[:, 3])
    # The graph is centered in the part of the axes above the legend
    bottom = (lower + upper) / 2 - (axes_box.height + legend_height) / 2
    center_x = (left + right) / 2
    ax.set_xlim((center_x - axes_box.width / 2) / scale, (center_x + axes_box.width / 2) / scale)
    ax.set_ylim(bottom / scale, (bottom + axes_box.height) / scale)
    # The limits already have the ratio of the axes: this keeps it if the figure is resized
    ax.set_aspect("equal", adjustable="box")
    return {state: tuple(xy[i]) for i, state in enumerate(states)}, zoom


def _wrap(ax, text, fontsize):
    """Wrap each line of the text to the width of the axes."""
    # Average width of a character: about half the font size
    width = int(ax.get_window_extent().width / (0.5 * fontsize * ax.get_figure().dpi / 72))
    return "\n".join(textwrap.fill(line, max(width, 20)) for line in text.split("\n"))


def _observed_graph_legend(ax, nodes, gap=6):
    """Draw the color scale and the caption at the bottom of the axes. Return their height in pixels."""
    caption = ax.text(
        0.5, 0.0,
        _wrap(ax,
              "States of %s (<nil> : none of them is active).\n"
              "Each arrow is the probability that the next change of these nodes leads to that state."
              % ", ".join(nodes), 7.5),
        transform=ax.transAxes, ha="center", va="bottom", multialignment="center", fontsize=7.5, color="grey"
    )
    ax.get_figure().draw_without_rendering()
    caption_height = caption.get_window_extent().height
    axes_height = ax.get_window_extent().height

    bar_height = 0.08 * ax.get_figure().dpi
    colorscale = ax.inset_axes([0.3, (caption_height + gap) / axes_height, 0.4, bar_height / axes_height])
    colorscale.imshow(
        np.linspace(0, 1, 256)[np.newaxis, :], aspect="auto",
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "observed_graph", [OBSERVED_GRAPH_LOW_COLOR, OBSERVED_GRAPH_HIGH_COLOR]
        )
    )
    colorscale.set_axis_off()
    colorscale.text(-0.02, 0.5, "0%", transform=colorscale.transAxes, ha="right", va="center", fontsize=8)
    colorscale.text(1.02, 0.5, "100%", transform=colorscale.transAxes, ha="left", va="center", fontsize=8)
    return caption_height + gap + bar_height + 2 * gap


def plot_observed_graph(table, ax, prune=True, show_probas=True, layout="neato"):
    """Draw the observed state transition graph.

    :param table: transition probabilities, from the states of the rows to the states of the columns
    :type table: pandas.DataFrame
    :param ax: the axes to draw on
    :param bool prune: leave out the states without any transition
    :param bool show_probas: write the probability of each transition on its arrow
    :param str layout: graphviz program used to place the states ("neato", "dot", ...),
        or "spring" for networkx's spring layout (also used when pygraphviz is not installed)
    """
    nodes = _observed_nodes(list(table.index) + list(table.columns))
    states = list(dict.fromkeys(list(table.index) + list(table.columns)))

    edges = [
        (source, target, table.loc[source, target])
        for source in table.index for target in table.columns
        if source != target and table.loc[source, target] > 0
    ]
    if prune:
        states = [state for state in states if any(state in edge[:2] for edge in edges)]

    ax.set_axis_off()
    legend_height = _observed_graph_legend(ax, nodes)

    if len(states) == 0:
        ax.text(
            0.5, 0.5, _wrap(ax, "No transition between the states of %s is that likely" % ", ".join(nodes), 10),
            transform=ax.transAxes, ha="center", va="center", multialignment="center", fontsize=10
        )
        return

    G = nx.DiGraph()
    G.add_nodes_from(states)
    G.add_weighted_edges_from(edges)
    pos = _observed_graph_layout(G, layout)

    def state_box(state, zorder, visible):
        return ax.text(
            *pos[state], state.replace(" -- ", "\n"),
            ha="center", va="center", multialignment="center", fontsize=10.5,
            color="black" if visible else "none", zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.5,
                      facecolor="white" if visible else "none", edgecolor="black" if visible else "none")
        )

    # The boxes are drawn above the arrows, so that an arrow never crosses a box. The arrows end on the borders of
    # invisible copies drawn before them: the size of a box is only updated when it is drawn.
    boxes = {state: state_box(state, 5, True) for state in states}
    clips = {state: state_box(state, 1, False) for state in states}

    pos, zoom = _fit_observed_graph(ax, pos, boxes, legend_height)
    for state in states:
        for box in (boxes[state], clips[state]):
            box.set_position(pos[state])
            box.set_fontsize(10.5 * zoom)
            box.get_bbox_patch().set_linewidth(1.5 * zoom)

    for source, target, proba in edges:
        # The two directions of a pair of states bow to opposite sides, and are at least 20 pixels apart
        rad = 0.0
        if G.has_edge(target, source):
            length = np.hypot(*(ax.transData.transform(pos[target]) - ax.transData.transform(pos[source])))
            rad = min(max(0.15, 20 * zoom / max(length, 1e-9)), 0.5)
        color = _proba_color(proba)

        ax.annotate(
            "", xy=pos[target], xytext=pos[source], zorder=3,
            arrowprops=dict(
                arrowstyle="-|>", color=color,
                linewidth=(0.75 + 3 * proba) * zoom, mutation_scale=(10 + 6 * proba) * zoom,
                patchA=clips[source].get_bbox_patch(), patchB=clips[target].get_bbox_patch(),
                shrinkA=2 * zoom, shrinkB=2 * zoom, connectionstyle="arc3,rad=%g" % rad
            )
        )

        if show_probas:
            # Middle of the curve: the aspect ratio is equal, so data coordinates are display coordinates up to a scale
            (x1, y1), (x2, y2) = pos[source], pos[target]
            ax.text(
                (x1 + x2) / 2 + 0.5 * rad * (y2 - y1), (y1 + y2) / 2 - 0.5 * rad * (x2 - x1),
                _format_proba(proba), ha="center", va="center", fontsize=7.5 * zoom, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
            )
