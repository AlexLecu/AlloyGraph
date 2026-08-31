#!/usr/bin/env python3
"""Graphical abstract for the AlloyGraph manuscript (Advanced Engineering Informatics).

The journal asks for a landscape image at exactly 2.5:1, at least 1328 x 531 px,
still readable when printed at 13 x 5 cm. The figure is therefore drawn at its
final printed size -- 13.0 x 5.2 cm -- so a point here is a point on paper and
the >= 7 pt floor can be checked directly rather than guessed at after
scaling. One label -- the novelty caption under the evaluation heading -- sits
at 6 pt by request, and is the only element below that floor.
Nothing is rescaled at typesetting.

Palette and typography follow the manuscript's own figures: the pale band /
saturated stroke scheme of `docs/paper/figures/fig1_system_architecture_big.drawio`
for the three system blocks, and the Okabe-Ito NEAR/MID/FAR colours of
`evaluation/paper_assets/make_figures.py` for the evaluation strata, so a stratum
is the same colour here as in the results figures.

Block widths are not eyeballed: `check_fit` measures every label against the box
that owns it with the real renderer and reports any overflow, so a wording change
that no longer fits fails loudly instead of silently colliding.

Outputs (into this directory):
    alloygraph_graphical_abstract.pdf     vector, TrueType text, 13.0 x 5.2 cm
    alloygraph_graphical_abstract.tiff    600 dpi LZW, 3071 x 1228 px
    alloygraph_graphical_abstract.png     600 dpi preview

Usage:
    python make_graphical_abstract.py
    python make_graphical_abstract.py --outdir /tmp --dpi 300
"""

import argparse
import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

CM = 1 / 2.54
FIG_W_CM, FIG_H_CM = 13.0, 5.2          # exactly 2.5:1
FIG_W, FIG_H = FIG_W_CM * CM, FIG_H_CM * CM

#: Drawing units, same 2.5:1 aspect, so one unit is one unit in either direction.
W, H = 250.0, 100.0
UNIT_PT = (FIG_W * 72.0) / W            # ~1.474 pt per drawing unit

# --------------------------------------------------------------------------
# Palette -- pale band + saturated stroke, as in the architecture figure
# --------------------------------------------------------------------------

ORANGE_BAND, ORANGE_EDGE, ORANGE_TEXT = "#fff5e6", "#d79b00", "#95590a"
PURPLE_BAND, PURPLE_EDGE, PURPLE_TEXT = "#f4eefb", "#9673a6", "#664480"
GREEN_BAND, GREEN_EDGE, GREEN_TEXT = "#eaf5e9", "#82b366", "#3f6b34"
BLUE_EDGE, BLUE_TEXT = "#6c8ebf", "#2f5c8f"
STRIP_BAND, STRIP_EDGE = "#f4f4f4", "#cccccc"

NOTE_BAND, NOTE_EDGE, NOTE_TEXT = "#fdf3e2", "#c98a2e", "#7a4f0a"

INK = "#1a1a1a"          # headings and card titles
BODY = "#3d3d3d"         # secondary card text
MUTED = "#6b6b6b"        # captions
FLOW = "#9a9a9a"         # inter-block arrows

#: Okabe-Ito, identical to the stratum colours in the manuscript's results
#: figures, so NEAR/MID/FAR carry the same colour throughout the paper.
NEAR_C, MID_C, FAR_C = "#0072B2", "#E69F00", "#009E73"
GREY_C = "#4D4D4D"           # the same neutral, for the cross-stratum row

# --------------------------------------------------------------------------
# Type scale. 7.0 pt is the floor at the 13 cm print size.
# --------------------------------------------------------------------------

FS_TITLE = 8.6
FS_HEAD = 7.8
FS_CARD = 7.4
FS_BODY = 7.0
FS_CAP = 7.0
FS_STRIP = 7.2
FS_TINY = 6.0            # the one label below the 7 pt floor, by request

_boxes = []              # (name, artist, (x_lo, x_hi)) for the fit report


def style():
    plt.rcParams.update({
        "pdf.fonttype": 42,          # embed TrueType: text stays selectable
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS_BODY,
        "text.color": INK,
        "figure.dpi": 300,
    })


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def band(ax, x0, y0, x1, y1, face, edge, r=2.2, lw=0.7, z=1):
    """A pale grouping band -- one of the three narrative blocks."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def card(ax, x0, y0, x1, y1, edge, face="#ffffff", r=1.4, lw=0.55, z=2):
    """A content card inside a band."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def label(ax, x, y, s, size=FS_BODY, weight="normal", color=BODY,
          ha="center", va="center", container=None, name=""):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                ha=ha, va=va, zorder=5)
    if container is not None:
        _boxes.append((name or s[:26], t, container))
    return t


def run_text(ax, rend, x, y, parts, align="left", container=None, name=""):
    """Lay a mixed-weight/colour run of text left to right, measured not guessed.

    Used where one word carries the emphasis -- the '-30%' in a result line --
    so the coloured fragment and the plain fragment cannot collide however the
    wording changes.
    """
    inv = ax.transData.inverted()
    arts, widths = [], []
    for s, weight, color, size in parts:
        t = ax.text(0, y, s, fontsize=size, fontweight=weight, color=color,
                    ha="left", va="center", zorder=5)
        bb = t.get_window_extent(renderer=rend)
        widths.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
        arts.append(t)
    total = sum(widths)
    cur = x if align == "left" else x - total / 2
    for i, (t, w) in enumerate(zip(arts, widths)):
        t.set_x(cur)
        cur += w
        if container is not None:
            _boxes.append((f"{name}:{i}", t, container))
    return total


def flow_arrow(ax, x0, x1, y, color=FLOW, lw=1.7, head=3.4):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle=f"-|>,head_width={head / 2},head_length={head}",
        mutation_scale=1, shrinkA=0, shrinkB=0,
        linewidth=lw, color=color, zorder=3))


def thin_arrow(ax, p0, p1, color, lw=0.65, head=1.8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=f"-|>,head_width={head / 2},head_length={head}",
        mutation_scale=1, shrinkA=0, shrinkB=0,
        linewidth=lw, color=color, zorder=3))


def swatch(ax, x, y, color, w=2.4, h=2.4):
    ax.add_patch(FancyBboxPatch(
        (x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
        facecolor=color, edgecolor="none", zorder=4))


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

def block_data(ax, rend, x0, x1, y0, y1):
    """Left: the open sources -- handbook, ontology / KG, physics-informed ML."""
    band(ax, x0, y0, x1, y1, ORANGE_BAND, ORANGE_EDGE)
    cx = (x0 + x1) / 2
    label(ax, cx, y1 - 5.5, "Open data & knowledge", size=FS_HEAD,
          weight="bold", color=ORANGE_TEXT,
          container=(x0 + 1.5, x1 - 1.5), name="head:data")

    cards = (
        ("Nickel Institute handbook", ("106 alloy records,", "open datasheets")),
        ("OWL 2 DL ontology", ("RDF graph (GraphDB)", "vector index (Weaviate)")),
        ("Physics-informed ML", ("XGBoost + Random Forest",
                                 "50-80 metallurgical features")),
    )
    px0, px1 = x0 + 2.5, x1 - 2.5
    ccx = (px0 + px1) / 2
    inner = (px0 + 1.5, px1 - 1.5)
    top, ch, gap = y1 - 10.5, 17.5, 5.0
    lead = 5.7                                   # line pitch, ~8.4 pt
    for i, (title, subs) in enumerate(cards):
        cy1 = top - i * (ch + gap)
        cy0 = cy1 - ch
        card(ax, px0, cy0, px1, cy1, ORANGE_EDGE, face="#fffdf8")
        # centre the title-plus-subs run in the card whatever its line count
        ty = (cy0 + cy1) / 2 + lead * len(subs) / 2
        label(ax, ccx, ty, title, size=FS_CARD, weight="bold",
              color=INK, container=inner, name=f"data{i}:t")
        for j, sub in enumerate(subs):
            label(ax, ccx, ty - lead * (j + 1), sub, size=FS_BODY, color=BODY,
                  container=inner, name=f"data{i}:{j}")


def block_agents(ax, rend, x0, x1, y0, y1):
    """Middle: two agents reconciling three anchors, plus the designer."""
    band(ax, x0, y0, x1, y1, PURPLE_BAND, PURPLE_EDGE)
    cx = (x0 + x1) / 2
    label(ax, cx, y1 - 5.5, "Multi-agent triangulation", size=FS_HEAD,
          weight="bold", color=PURPLE_TEXT,
          container=(x0 + 1.5, x1 - 1.5), name="head:agents")
    label(ax, cx, y1 - 11.5, "CrewAI  ·  Llama 3.3-70B", size=FS_CAP,
          color=MUTED, container=(x0 + 2, x1 - 2), name="agents:sub")

    px0, px1 = x0 + 3, x1 - 3
    inner = (px0 + 1.5, px1 - 1.5)

    # three evidence anchors, coloured by where they come from in the
    # architecture figure: prediction engine (ML, physics) and knowledge layer
    anchors = (("ML", BLUE_EDGE, BLUE_TEXT),
               ("physics", GREEN_EDGE, GREEN_TEXT),
               ("KG", ORANGE_EDGE, ORANGE_TEXT))
    chip_y1, chip_y0 = y1 - 16.0, y1 - 24.0
    cgap = 2.0
    cw = ((px1 - px0) - 2 * cgap) / 3
    for i, (name, edge, txt) in enumerate(anchors):
        ax0 = px0 + i * (cw + cgap)
        card(ax, ax0, chip_y0, ax0 + cw, chip_y1, edge, face="#ffffff", r=1.2)
        label(ax, ax0 + cw / 2, (chip_y0 + chip_y1) / 2, name, size=FS_BODY,
              weight="bold", color=txt,
              container=(ax0 + 0.8, ax0 + cw - 0.8), name=f"anchor:{name}")

    # analyst / reviewer
    ay1, ay0 = chip_y0 - 7.5, chip_y0 - 23.0
    # land the three anchors on separate points of the card edge, otherwise the
    # arrowheads stack on one spot and read as a blob at print size
    for i in range(3):
        ax0 = px0 + i * (cw + cgap) + cw / 2
        thin_arrow(ax, (ax0, chip_y0 - 0.6),
                   (cx + (i - 1) * 7.0, ay1 + 0.8), PURPLE_EDGE)
    card(ax, px0, ay0, px1, ay1, PURPLE_EDGE, face="#ffffff")
    label(ax, cx, ay1 - 5.2, "Analyst  +  Reviewer", size=FS_CARD,
          weight="bold", color=INK, container=inner, name="ar:t")
    label(ax, cx, ay1 - 10.8, "cross-check three anchors", size=FS_BODY,
          color=BODY, container=inner, name="ar:a")

    # deterministic guards
    gy1, gy0 = ay0 - 3.5, ay0 - 12.0
    card(ax, px0, gy0, px1, gy1, PURPLE_EDGE, face="#f7f1fc", r=1.2)
    label(ax, cx, (gy0 + gy1) / 2, "deterministic guards", size=FS_BODY,
          weight="bold", color=PURPLE_TEXT, container=inner, name="guards")

    # designer
    dy1, dy0 = gy0 - 3.5, y0 + 3.0
    card(ax, px0, dy0, px1, dy1, PURPLE_EDGE, face="#ffffff")
    dcm = (dy0 + dy1) / 2
    label(ax, cx, dcm + 2.9, "Designer agent", size=FS_CARD, weight="bold",
          color=INK, container=inner, name="des:t")
    label(ax, cx, dcm - 2.9, "inverse design", size=FS_BODY, color=BODY,
          container=inner, name="des:a")


def block_eval(ax, rend, x0, x1, y0, y1):
    """Right: the headline finding -- what carries accuracy in each stratum."""
    band(ax, x0, y0, x1, y1, GREEN_BAND, GREEN_EDGE)
    cx = (x0 + x1) / 2
    bx0, bx1 = x0 + 3.0, x1 - 3.0
    inner = (bx0, bx1)

    run_text(ax, rend, cx, y1 - 5.5,
             [("Stratified evaluation", "bold", GREEN_TEXT, FS_HEAD),
              ("   n = 88", "normal", MUTED, FS_CAP)],
             align="center", container=(x0 + 1.5, x1 - 1.5), name="head:eval")

    # Names this panel as the outcome rather than another pipeline stage. It
    # cannot sit beside the incoming arrow -- that gutter is 6.5 units wide and
    # the label is 65 -- so it rides directly under the panel heading instead.
    label(ax, cx, y1 - 10.7, "evaluated by compositional novelty", size=FS_TINY,
          color=MUTED, container=(x0 + 1.5, x1 - 1.5), name="eval:tiny")

    # stratum bar
    by1, by0 = y1 - 14.7, y1 - 20.7
    seg = (bx1 - bx0) / 3
    for i, (nm, col) in enumerate((("NEAR", NEAR_C), ("MID", MID_C), ("FAR", FAR_C))):
        sx0 = bx0 + i * seg
        ax.add_patch(Rectangle((sx0, by0), seg, by1 - by0, facecolor=col,
                               edgecolor="#ffffff", linewidth=0.8, zorder=3))
        label(ax, sx0 + seg / 2, (by0 + by1) / 2, nm, size=FS_BODY,
              weight="bold", color="#ffffff",
              container=(sx0 + 1, sx0 + seg - 1), name=f"seg:{nm}")

    # distance axis
    flow_arrow(ax, bx0, bx1, by0 - 3.0, color="#8a8a8a", lw=0.8, head=2.4)
    label(ax, cx, by0 - 6.7, "distance to nearest training alloy", size=FS_CAP,
          color=MUTED, container=inner, name="eval:axis")

    # Mechanism attributions, one row per stratum. Every number below is the
    # committed result, not a rounded recollection:
    #   NEAR  YS MAE 107.56 -> 75.42 MPa, KG anchoring over ML+physics  (-29.9%)
    #   FAR   YS MAE 110.10 -> 89.83 MPa, full agent system over ML+physics+KG
    #         (-18.4%; five-seed mean, sigma 1.32)
    #   ALL   EM MAE 12.86 -> 8.21 GPa, physics over ML-only  (-36.2%, n=304)
    # from evaluation/prediction/results/nn_distance_stratified_results.md and
    # analyst_ablation.md. The modulus row quotes the pooled figure the paper
    # text uses; it is not stratified, because physics helps NEAR (-54%) and
    # FAR (-41%) but makes MID slightly worse (9.92 -> 10.48 GPa).
    rows = (
        (NEAR_C, "near", "KG anchoring: exact retrieval",
         [("-30% ", "bold", NEAR_C, FS_BODY),
          ("yield-strength error, known alloys", "normal", BODY, FS_BODY)]),
        (FAR_C, "far", "agent triangulation, novel alloys",
         [("-18% ", "bold", FAR_C, FS_BODY),
          ("yield-strength error", "normal", BODY, FS_BODY)]),
        (GREY_C, "all", "physics: elastic modulus",
         [("-36% ", "bold", GREY_C, FS_BODY),
          ("elastic-modulus error", "normal", BODY, FS_BODY)]),
    )
    tx = bx0 + 3.2
    row_inner = (tx, bx1)
    ry = by0 - 15.2
    for col, tag, line1, parts in rows:
        swatch(ax, bx0, ry + 2.8, col)
        label(ax, tx, ry + 2.8, line1, size=FS_BODY, weight="bold", color=INK,
              ha="left", container=row_inner, name=f"row:{tag}:1")
        run_text(ax, rend, tx, ry - 2.8, parts, align="left",
                 container=row_inner, name=f"row:{tag}:2")
        ry -= 10.0

    # the erratum that motivates stratifying in the first place
    ny1, ny0 = ry + 3.6, y0 + 2.7
    card(ax, bx0, ny0, bx1, ny1, NOTE_EDGE, face=NOTE_BAND, r=1.2)
    ncm = (ny0 + ny1) / 2
    note_inner = (bx0 + 1.5, bx1 - 1.5)
    run_text(ax, rend, cx, ncm + 2.8,
             [("13 of 88", "bold", NOTE_TEXT, FS_BODY),
              (" 'held-out' alloys were", "normal", NOTE_TEXT, FS_BODY)],
             align="center", container=note_inner, name="note:1")
    label(ax, cx, ncm - 2.8, "vendor-renamed near-duplicates", size=FS_BODY,
          color=NOTE_TEXT, container=note_inner, name="note:2")


# --------------------------------------------------------------------------
# Fit report -- every label must sit inside the box that owns it
# --------------------------------------------------------------------------

def check_fit(fig, ax):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for name, artist, (cx0, cx1) in _boxes:
        bb = artist.get_window_extent(renderer=rend)
        p0 = inv.transform((bb.x0, bb.y0))[0]
        p1 = inv.transform((bb.x1, bb.y1))[0]
        over = max(cx0 - p0, p1 - cx1)
        if over > 0.05:
            bad.append((name, round(over, 2), round(p1 - p0, 1), round(cx1 - cx0, 1)))
    if bad:
        print("  OVERFLOW (units over, text width, box width):")
        for row in sorted(bad, key=lambda r: -r[1]):
            print(f"    {row[0]:<22} +{row[1]:>5}   {row[2]:>5} / {row[3]:>5}")
    else:
        print(f"  fit: all {len(_boxes)} labels inside their containers")
    return bad


# --------------------------------------------------------------------------

def build():
    style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="#ffffff",
                           edgecolor="none", zorder=0))
    rend = fig.canvas.get_renderer()

    # title strip
    ax.add_patch(Rectangle((0, H - 10.0), W, 10.0, facecolor=STRIP_BAND,
                           edgecolor="none", zorder=1))
    ax.plot([0, W], [H - 10.0, H - 10.0], color=STRIP_EDGE, lw=0.6, zorder=2)
    label(ax, W / 2, H - 5.0,
          "AlloyGraph: Knowledge-Graph-Grounded Multi-Agent Superalloy Design",
          size=FS_TITLE, weight="bold", color=INK,
          container=(3, W - 3), name="title")

    # bottom strip
    ax.add_patch(Rectangle((0, 0), W, 8.5, facecolor=STRIP_BAND,
                           edgecolor="none", zorder=1))
    ax.plot([0, W], [8.5, 8.5], color=STRIP_EDGE, lw=0.6, zorder=2)
    run_text(ax, rend, W / 2, 4.2,
             [("Fully open:", "bold", "#3a3a3a", FS_STRIP),
              ("  data  ·  code  ·  models  ·  erratum ledger  ·  "
               "seeded evaluation harness", "normal", "#4a4a4a", FS_STRIP)],
             align="center", container=(3, W - 3), name="strip")

    # three blocks, left to right
    by0, by1 = 11.5, H - 12.5
    ax0, ax1 = 3.0, 74.0
    bx0, bx1 = 80.5, 149.5
    cx0, cx1 = 156.0, 247.0

    block_data(ax, rend, ax0, ax1, by0, by1)
    block_agents(ax, rend, bx0, bx1, by0, by1)
    block_eval(ax, rend, cx0, cx1, by0, by1)

    ymid = (by0 + by1) / 2
    flow_arrow(ax, ax1 + 0.9, bx0 - 0.9, ymid)
    flow_arrow(ax, bx1 + 0.9, cx0 - 0.9, ymid)

    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig, ax = build()
    check_fit(fig, ax)

    stem = os.path.join(args.outdir, "alloygraph_graphical_abstract")
    fig.savefig(stem + ".pdf", format="pdf")
    fig.savefig(stem + ".png", dpi=args.dpi)
    fig.savefig(stem + ".tiff", dpi=args.dpi,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"  {FIG_W_CM} x {FIG_H_CM} cm, ratio {FIG_W / FIG_H:.4f}, "
          f"raster {round(FIG_W * args.dpi)} x {round(FIG_H * args.dpi)} px "
          f"@ {args.dpi} dpi")
    print(f"  body type floor {FS_BODY} pt at print size; the "
          f"'evaluated by compositional novelty' caption is {FS_TINY} pt; "
          f"1 unit = {UNIT_PT:.3f} pt")
    for ext in ("pdf", "tiff", "png"):
        print(f"  wrote {stem}.{ext}")


if __name__ == "__main__":
    main()
