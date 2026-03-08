import { useCallback, useEffect, useMemo, useRef } from 'react';

const PAPER_RGB = [250, 250, 245];
const LIGHT_DIR = normalize3([0.4, -0.45, 1.0]);
const MAX_FOLD_RAD = Math.PI * 0.92;
const SIDE_EPS = 1e-7;
const MOUNTAIN_COLOR = 'rgba(245, 158, 11, 0.95)';
const VALLEY_COLOR = 'rgba(56, 189, 248, 0.95)';

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

function buildGridMesh(resolution = 18) {
  const vertices = [];
  for (let y = 0; y <= resolution; y += 1) {
    for (let x = 0; x <= resolution; x += 1) {
      vertices.push([x / resolution, y / resolution, 0]);
    }
  }

  const triangles = [];
  const stride = resolution + 1;
  for (let y = 0; y < resolution; y += 1) {
    for (let x = 0; x < resolution; x += 1) {
      const a = y * stride + x;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      triangles.push([a, b, d]);
      triangles.push([a, d, c]);
    }
  }

  return { vertices, triangles, resolution };
}

function rotateAroundAxis(point, axisPoint, axisDir, angleRad) {
  const px = point[0] - axisPoint[0];
  const py = point[1] - axisPoint[1];
  const pz = point[2] - axisPoint[2];

  const kx = axisDir[0];
  const ky = axisDir[1];
  const kz = axisDir[2];

  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);

  const crossX = ky * pz - kz * py;
  const crossY = kz * px - kx * pz;
  const crossZ = kx * py - ky * px;

  const dot = px * kx + py * ky + pz * kz;
  const oneMinus = 1.0 - cosA;

  return [
    axisPoint[0] + px * cosA + crossX * sinA + kx * dot * oneMinus,
    axisPoint[1] + py * cosA + crossY * sinA + ky * dot * oneMinus,
    axisPoint[2] + pz * cosA + crossZ * sinA + kz * dot * oneMinus,
  ];
}

function applyFoldToVertices(vertices, fold, progress) {
  if (!fold || progress <= 0) return;
  const [x1, y1] = fold.from;
  const [x2, y2] = fold.to;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1e-8) return;

  const sideValues = [];
  let posCount = 0;
  let negCount = 0;

  for (let i = 0; i < vertices.length; i += 1) {
    const v = vertices[i];
    const side = dx * (v[1] - y1) - dy * (v[0] - x1);
    sideValues.push(side);
    if (side > SIDE_EPS) posCount += 1;
    else if (side < -SIDE_EPS) negCount += 1;
  }

  let rotatePositive = posCount <= negCount;
  if (posCount === 0 && negCount > 0) rotatePositive = false;
  if (negCount === 0 && posCount > 0) rotatePositive = true;
  if (posCount === 0 && negCount === 0) return;

  const sign = fold.assignment === 'V' ? 1 : -1;
  const angle = sign * MAX_FOLD_RAD * progress;
  const axisPoint = [x1, y1, 0];
  const axisDir = [dx / len, dy / len, 0];

  for (let i = 0; i < vertices.length; i += 1) {
    const side = sideValues[i];
    const shouldRotate = rotatePositive ? side > SIDE_EPS : side < -SIDE_EPS;
    if (!shouldRotate) continue;
    vertices[i] = rotateAroundAxis(vertices[i], axisPoint, axisDir, angle);
  }
}

function projectVertex(vertex, dim) {
  let x = vertex[0] - 0.5;
  let y = vertex[1] - 0.5;
  let z = vertex[2];

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

function foldProgresses(stepValue, foldCount, mode, totalSteps) {
  const values = new Array(foldCount).fill(0);
  if (foldCount === 0) return values;

  if (mode === 'final') {
    const startCollapse = Math.max(totalSteps - 1, 0);
    const collapse = clamp(stepValue - startCollapse, 0, 1);
    for (let i = 0; i < foldCount; i += 1) values[i] = collapse;
    return values;
  }

  for (let i = 0; i < foldCount; i += 1) {
    if (stepValue >= i + 1) values[i] = 1;
    else if (stepValue > i) values[i] = clamp(stepValue - i, 0, 1);
  }
  return values;
}

function stepEasing(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2;
}

export default function Fold3DCanvas({
  steps,
  currentStep,
  totalSteps,
  mode = 'progressive',
  dim = 280,
}) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const animatedStepRef = useRef(currentStep);

  const folds = useMemo(
    () => (steps || [])
      .map((s) => s.fold)
      .filter(Boolean)
      .map((fold) => ({
        from: [Number(fold.from_point[0]), Number(fold.from_point[1])],
        to: [Number(fold.to_point[0]), Number(fold.to_point[1])],
        assignment: fold.assignment === 'M' ? 'M' : 'V',
      })),
    [steps],
  );

  const mesh = useMemo(() => buildGridMesh(18), []);

  const draw = useCallback((stepValue) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, dim, dim);
    ctx.fillStyle = '#121220';
    ctx.fillRect(0, 0, dim, dim);

    const vertices = mesh.vertices.map((v) => [v[0], v[1], v[2]]);
    const progress = foldProgresses(stepValue, folds.length, mode, totalSteps);

    for (let i = 0; i < folds.length; i += 1) {
      if (progress[i] <= 0) continue;
      applyFoldToVertices(vertices, folds[i], progress[i]);
    }

    const projected = vertices.map((v) => projectVertex(v, dim));

    const tris = mesh.triangles.map((tri) => {
      const p0 = projected[tri[0]];
      const p1 = projected[tri[1]];
      const p2 = projected[tri[2]];
      const avgZ = (p0.z + p1.z + p2.z) / 3;

      const v0 = vertices[tri[0]];
      const v1 = vertices[tri[1]];
      const v2 = vertices[tri[2]];
      const normal = normalize3(cross3(sub3(v1, v0), sub3(v2, v0)));
      const intensity = dot3(normal, LIGHT_DIR);

      return {
        tri,
        avgZ,
        shade: shadePaper(intensity),
      };
    });

    tris.sort((a, b) => a.avgZ - b.avgZ);

    for (const triInfo of tris) {
      const [a, b, c] = triInfo.tri;
      const p0 = projected[a];
      const p1 = projected[b];
      const p2 = projected[c];

      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.closePath();
      ctx.fillStyle = triInfo.shade;
      ctx.fill();
      ctx.strokeStyle = 'rgba(42, 42, 58, 0.22)';
      ctx.lineWidth = 0.55;
      ctx.stroke();
    }

    const res = mesh.resolution;
    const stride = res + 1;
    const pointToIndex = (pt) => {
      const ix = clamp(Math.round(pt[0] * res), 0, res);
      const iy = clamp(Math.round(pt[1] * res), 0, res);
      return iy * stride + ix;
    };

    for (let i = 0; i < folds.length; i += 1) {
      if (progress[i] <= 0.02) continue;
      const fold = folds[i];
      const aIdx = pointToIndex(fold.from);
      const bIdx = pointToIndex(fold.to);
      const pa = projected[aIdx];
      const pb = projected[bIdx];

      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.strokeStyle = fold.assignment === 'M' ? MOUNTAIN_COLOR : VALLEY_COLOR;
      ctx.globalAlpha = clamp(0.35 + 0.65 * progress[i], 0, 1);
      ctx.lineWidth = 2.15;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [dim, folds, mesh, mode, totalSteps]);

  useEffect(() => {
    draw(animatedStepRef.current);
  }, [draw]);

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    const startValue = animatedStepRef.current;
    const endValue = currentStep;
    const durationMs = 420;
    const startAt = performance.now();

    const tick = (now) => {
      const t = clamp((now - startAt) / durationMs, 0, 1);
      const eased = stepEasing(t);
      const value = startValue + (endValue - startValue) * eased;
      animatedStepRef.current = value;
      draw(value);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [currentStep, draw]);

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
