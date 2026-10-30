"""Interactive rendering of a walked traversal, in the style of a graph browser.

Nodes are circles coloured by label, relationships are arrows captioned with
their type, and the layout runs left to right by hop, so a multi-level traversal
reads as the columns it actually is. The viewer reveals the walk one hop at a
time and can replay it; clicking a node shows its stored properties.

The whole viewer is one self-contained HTML document driven by the traversal
data, so the Streamlit page stays a rendering layer and nothing here needs the
application to run in order to be tested.
"""

from __future__ import annotations

import json
from typing import Any

from retrieval_tester.context_tool import TraversalGraph

#: Colours assigned to labels in order of first appearance, the way a graph
#: browser gives each label its own colour. Chosen to stay legible on both
#: a light and a dark page.
_PALETTE = (
    "#4c8eda",
    "#57c7e3",
    "#f79767",
    "#f16667",
    "#d9c8ae",
    "#8dcc93",
    "#ecb5c9",
    "#c990c0",
    "#ffc454",
    "#a5abb6",
)
#: Ring drawn around the nodes that answered the query.
_RESULT_RING = "#16a34a"
#: Ring drawn around the node the traversal started from.
_ANCHOR_RING = "#1d4ed8"

_TEMPLATE = """
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font-family: "Source Sans Pro", system-ui, sans-serif; }
  #bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
         padding: 6px 2px 10px; font-size: 13px; }
  button { font: inherit; padding: 4px 12px; border-radius: 6px; cursor: pointer;
           border: 1px solid rgba(128,128,128,.4); background: rgba(128,128,128,.12); }
  button:disabled { opacity: .45; cursor: default; }
  #hop { flex: 1; min-width: 120px; }
  #stage { font-weight: 600; }
  #legend { display: flex; gap: 12px; flex-wrap: wrap; padding: 6px 2px; font-size: 12px; }
  .chip { display: inline-flex; align-items: center; gap: 5px; }
  .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
  #net { height: HEIGHTpx; border: 1px solid rgba(128,128,128,.25); border-radius: 8px; }
  #info { font-size: 12px; padding: 8px 2px; min-height: 2.6em; line-height: 1.5; }
  #info b { font-family: ui-monospace, monospace; }
  .muted { opacity: .65; }
</style>

<div id="bar">
  <button id="play">▶ Replay the walk</button>
  <button id="back">←</button>
  <input id="hop" type="range" min="0" max="0" value="0">
  <button id="next">→</button>
  <span id="stage"></span>
</div>
<div id="legend"></div>
<div id="net"></div>
<div id="info" class="muted">Click a node to see everything stored on it.</div>

<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<script>
const DATA = __DATA__;
const COLOURS = __COLOURS__;
const RESULT_RING = "__RESULT_RING__";
const ANCHOR_RING = "__ANCHOR_RING__";
const net = document.getElementById("net");

if (typeof vis === "undefined") {
  net.innerHTML =
    '<p style="padding:1em">The graph viewer could not load (no network access to the ' +
    'CDN). The static picture below shows the same walk.</p>';
} else {
  const legend = document.getElementById("legend");
  DATA.labels.forEach(label => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = '<span class="dot" style="background:' + COLOURS[label] + '"></span>' + label;
    legend.appendChild(chip);
  });
  ["started here", "answered the query"].forEach((text, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML =
      '<span class="dot" style="background:transparent;border:3px solid ' +
      (index ? RESULT_RING : ANCHOR_RING) + '"></span>' + text;
    legend.appendChild(chip);
  });

  const nodes = new vis.DataSet(DATA.nodes.map(node => ({
    id: node.id,
    label: node.caption.length > 16 ? node.caption.slice(0, 15) + "\\u2026" : node.caption,
    title: node.label + ": " + node.caption,
    level: node.level,
    shape: "dot",
    size: node.isResult ? 26 : 20,
    color: {
      background: COLOURS[node.label],
      border: node.isResult ? RESULT_RING : node.level === 0 ? ANCHOR_RING : COLOURS[node.label],
      highlight: { background: COLOURS[node.label], border: "#111827" },
    },
    borderWidth: node.isResult || node.level === 0 ? 4 : 1,
    font: { size: 13, face: "system-ui", multi: false, vadjust: -2 },
  })));

  const edges = new vis.DataSet(DATA.edges.map((edge, index) => ({
    id: index,
    from: edge.source,
    to: edge.target,
    label: edge.type,
    level: edge.level,
    arrows: { to: { enabled: true, scaleFactor: 0.7 } },
    color: { color: "#9aa4b2", highlight: "#111827" },
    font: { size: 11, align: "horizontal", background: "rgba(255,255,255,.75)", strokeWidth: 0 },
    smooth: { type: "dynamic" },
  })));

  const network = new vis.Network(net, { nodes: nodes, edges: edges }, {
    layout: DATA.stages.length > 1
      ? { hierarchical: { enabled: true, direction: "LR", sortMethod: "directed",
                          levelSeparation: 230, nodeSpacing: 90 } }
      : { improvedLayout: true },
    physics: DATA.stages.length > 1
      ? false
      : { stabilization: { iterations: 220 }, barnesHut: { springLength: 140 } },
    interaction: { hover: true, tooltipDelay: 120, navigationButtons: false },
    nodes: { shadow: false },
  });

  const info = document.getElementById("info");
  const byId = {};
  DATA.nodes.forEach(node => { byId[node.id] = node; });
  network.on("click", params => {
    const node = params.nodes.length ? byId[params.nodes[0]] : null;
    if (!node) {
      info.className = "muted";
      info.textContent = "Click a node to see everything stored on it.";
      return;
    }
    info.className = "";
    const rows = Object.keys(node.properties)
      .map(key => "<b>" + key + "</b>: " + node.properties[key]);
    info.innerHTML =
      "<b>(" + node.label + ")</b> at hop " + node.level +
      (node.isResult ? " \\u2014 this one answered the query" : "") + "<br>" +
      (rows.join(" \\u00b7 ") || "no properties stored");
  });

  const last = Math.max(DATA.stages.length - 1, 0);
  const slider = document.getElementById("hop");
  const stage = document.getElementById("stage");
  slider.max = String(last);
  slider.value = String(last);

  function show(upto) {
    nodes.forEach(node => nodes.update({ id: node.id, hidden: node.level > upto }));
    edges.forEach(edge => edges.update({ id: edge.id, hidden: edge.level > upto }));
    stage.textContent = "hop " + upto + " of " + last + " \\u2014 " + DATA.stages[upto];
    document.getElementById("back").disabled = upto === 0;
    document.getElementById("next").disabled = upto === last;
    network.fit({ animation: { duration: 400 } });
  }

  document.getElementById("back").onclick = () => {
    slider.value = String(Math.max(0, Number(slider.value) - 1)); show(Number(slider.value));
  };
  document.getElementById("next").onclick = () => {
    slider.value = String(Math.min(last, Number(slider.value) + 1)); show(Number(slider.value));
  };
  slider.oninput = () => show(Number(slider.value));
  document.getElementById("play").onclick = () => {
    let frame = 0;
    show(0);
    slider.value = "0";
    const timer = setInterval(() => {
      frame += 1;
      if (frame > last) { clearInterval(timer); return; }
      slider.value = String(frame);
      show(frame);
    }, 1100);
  };

  show(last);
}
</script>
"""


def render_html(graph: TraversalGraph, *, height: int = 460) -> str:
    """A self-contained interactive view of ``graph``."""
    payload: dict[str, Any] = graph.to_payload()
    colours = {
        label: _PALETTE[index % len(_PALETTE)]
        for index, label in enumerate(payload["labels"])
    }
    return (
        _TEMPLATE.replace("HEIGHT", str(height))
        .replace("__DATA__", _json(payload))
        .replace("__COLOURS__", _json(colours))
        .replace("__RESULT_RING__", _RESULT_RING)
        .replace("__ANCHOR_RING__", _ANCHOR_RING)
    )


def _json(value: Any) -> str:
    """JSON that is safe to inline inside a <script> block."""
    return json.dumps(value).replace("</", "<\\/")
