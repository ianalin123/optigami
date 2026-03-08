import { useCallback, useEffect, useRef } from 'react';

const PAPER_RGB = [250, 250, 245];
const LIGHT_DIR = normalize3([0.4, -0.45, 1.0]);
const MOUNTAIN_COLOR = 'rgba(245, 158, 11, 0.9)';
const VALLEY_COLOR = 'rgba(56, 189, 248, 0.9)';

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalize3(v) {
  const mag = Math.hypot(v[0], v[1], v[2]);
  if (mag < 1e-12) return [0, 0, 0];
  return [v[0] / mag, v[1] / mag, v[2] / mag];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function sub3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function shadePaper(intensity) {
  const lit = clamp(0.3 + 0.7 * Math.abs(intensity), 0.0, 1.0);
  const r = Math.round(PAPER_RGB[0] * lit);
  const g = Math.round(PAPER_RGB[1] * lit);
  const b = Math.round(PAPER_RGB[2] * lit);
  return `rgb(${r}, ${g}, ${b})`;
}

function strainColor(strain, intensity) {
  const t = clamp(strain / 0.15, 0, 1);
  const lit = clamp(0.3 + 0.7 * Math.abs(intensity), 0, 1);
  // Blend from paper ivory to red-orange with lighting
  const r = Math.round((250 + t * 5) * lit);
  const g = Math.round((250 - t * 200) * lit);
  const bv = Math.round((245 - t * 245) * lit);
  return `rgb(${clamp(r,0,255)}, ${clamp(g,0,255)}, ${clamp(bv,0,255)})`;
}

function projectVertex(vertex, dim) {
  let x = vertex[0] - 0.5;
  let y = vertex[1] - 0.5;
  let z = vertex[2] || 0;

  const pitch = 1.04;
  const yaw = -0.78;

  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const y1 = y * cp - z * sp;
  const z1 = y * sp + z * cp;

  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const x2 = x * cy + z1 * sy;
  const z2 = -x * sy + z1 * cy;

  const camDist = 2.8;
  const perspective = camDist / (camDist - z2);

  return {
    x: dim * 0.5 + x2 * perspective * dim * 0.82,
    y: dim * 0.52 - y1 * perspective * dim * 0.82,
    z: z2,
  };
}

function drawPaperState(ctx, paperState, dim) {
  ctx.clearRect(0, 0, dim, dim);
  ctx.fillStyle = '#121220';
  ctx.fillRect(0, 0, dim, dim);

  if (!paperState) {
    // Draw flat sheet for initial state
    const flatVerts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]];
    const flatFaces = [[0,1,2],[0,2,3]];
    renderMesh(ctx, flatVerts, flatFaces, null, dim);
    return;
  }

  const { vertices_coords, faces_vertices, strain_per_vertex, edges_vertices, edges_assignment } = paperState;

  if (!vertices_coords || !faces_vertices) {
    ctx.fillStyle = '#121220';
    ctx.fillRect(0, 0, dim, dim);
    return;
  }

  renderMesh(ctx, vertices_coords, faces_vertices, strain_per_vertex, dim);

  // Draw fold creases on top
  if (edges_vertices && edges_assignment) {
    const projected = vertices_coords.map(v => projectVertex(v, dim));
    for (let i = 0; i < edges_vertices.length; i++) {
      const asgn = edges_assignment[i];
      if (asgn !== 'M' && asgn !== 'V') continue;
      const [ai, bi] = edges_vertices[i];
      const pa = projected[ai];
      const pb = projected[bi];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.strokeStyle = asgn === 'M' ? MOUNTAIN_COLOR : VALLEY_COLOR;
      ctx.lineWidth = 2.0;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
}

function renderMesh(ctx, verts, faces, strain, dim) {
  const projected = verts.map(v => projectVertex(v, dim));

  const tris = [];
  for (const face of faces) {
    // Triangulate face (fan from first vertex)
    for (let k = 1; k < face.length - 1; k++) {
      const a = face[0], b = face[k], c = face[k + 1];
      const p0 = projected[a];
      const p1 = projected[b];
      const p2 = projected[c];
      if (!p0 || !p1 || !p2) continue;
      const avgZ = (p0.z + p1.z + p2.z) / 3;
      const v0 = verts[a], v1 = verts[b], v2 = verts[c];
      const normal = normalize3(cross3(sub3(v1, v0), sub3(v2, v0)));
      const intensity = dot3(normal, LIGHT_DIR);
      const avgStrain = strain
        ? ((strain[a] || 0) + (strain[b] || 0) + (strain[c] || 0)) / 3
        : 0;
      tris.push({ a, b, c, avgZ, intensity, avgStrain });
    }
  }

  tris.sort((x, y) => x.avgZ - y.avgZ);

  for (const tri of tris) {
    const p0 = projected[tri.a];
    const p1 = projected[tri.b];
    const p2 = projected[tri.c];

    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.closePath();

    const fillColor = tri.avgStrain > 0.005
      ? strainColor(tri.avgStrain, tri.intensity)
      : shadePaper(tri.intensity);

    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = 'rgba(42, 42, 58, 0.22)';
    ctx.lineWidth = 0.55;
    ctx.stroke();
  }
}

export default function Fold3DCanvas({
  steps,
  currentStep,
  dim = 280,
}) {
  const canvasRef = useRef(null);

  const getPaperState = useCallback(() => {
    if (!steps || steps.length === 0 || currentStep === 0) return null;
    const stepData = steps[currentStep - 1];
    return stepData ? stepData.paper_state : null;
  }, [steps, currentStep]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawPaperState(ctx, getPaperState(), dim);
  }, [getPaperState, dim]);

  return (
    <canvas
      ref={canvasRef}
      width={dim}
      height={dim}
      className="canvas-3d"
      aria-label="3D fold preview"
    />
  );
}
