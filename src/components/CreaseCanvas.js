const MOUNTAIN = '#f59e0b';
const VALLEY = '#38bdf8';

function toSvg(x, y, dim) {
  return [x * dim, (1 - y) * dim];
}

function GhostEdges({ target, dim }) {
  if (!target) return null;
  const { vertices_coords, edges_vertices, edges_assignment } = target;
  if (!vertices_coords || !edges_vertices || !edges_assignment) return null;

  return edges_vertices.map((ev, i) => {
    const asgn = edges_assignment[i];
    if (asgn === 'B') return null;
    const [v1x, v1y] = vertices_coords[ev[0]];
    const [v2x, v2y] = vertices_coords[ev[1]];
    const [x1, y1] = toSvg(v1x, v1y, dim);
    const [x2, y2] = toSvg(v2x, v2y, dim);
    const color = asgn === 'M' ? MOUNTAIN : VALLEY;
    return (
      <line
        key={i}
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={color}
        strokeOpacity={0.25}
        strokeWidth={1.5}
        strokeDasharray="5 4"
      />
    );
  });
}

function CurrentEdges({ paperState, dim }) {
  if (!paperState || !paperState.edges) return null;
  return paperState.edges.map((edge) => {
    if (edge.assignment === 'B') return null;
    const [x1, y1] = toSvg(edge.v1[0], edge.v1[1], dim);
    const [x2, y2] = toSvg(edge.v2[0], edge.v2[1], dim);
    const color = edge.assignment === 'M' ? MOUNTAIN : VALLEY;
    return (
      <line
        key={edge.id}
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={color}
        strokeWidth={2.5}
        strokeLinecap="square"
      />
    );
  });
}

function AnchorCrosses({ paperState, dim }) {
  if (!paperState || !paperState.anchor_points) return null;
  const size = 4;
  return paperState.anchor_points.map((pt, i) => {
    const [cx, cy] = toSvg(pt[0], pt[1], dim);
    return (
      <g key={i}>
        <line
          x1={cx - size} y1={cy} x2={cx + size} y2={cy}
          stroke="#64748b" strokeWidth={1}
        />
        <line
          x1={cx} y1={cy - size} x2={cx} y2={cy + size}
          stroke="#64748b" strokeWidth={1}
        />
      </g>
    );
  });
}

export default function CreaseCanvas({ paperState, target, dim = 280, ghostOnly = false }) {
  const pad = 1;
  const size = dim;

  return (
    <svg
      className="canvas-svg"
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{ flexShrink: 0 }}
    >
      {/* Paper background */}
      <rect
        x={pad} y={pad}
        width={size - pad * 2} height={size - pad * 2}
        fill="#fafaf5"
      />

      {/* Ghost target overlay */}
      <GhostEdges target={target} dim={size} />

      {/* Current paper state */}
      {!ghostOnly && (
        <>
          <CurrentEdges paperState={paperState} dim={size} />
          <AnchorCrosses paperState={paperState} dim={size} />
        </>
      )}

      {/* Paper border */}
      <rect
        x={pad} y={pad}
        width={size - pad * 2} height={size - pad * 2}
        fill="none"
        stroke="#2a2a3a"
        strokeWidth={1}
      />
    </svg>
  );
}
