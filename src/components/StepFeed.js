import { useEffect, useRef } from 'react';

function compactnessDelta(step, prevStep) {
  if (!step || !step.metrics) return null;
  const curr = step.metrics.compactness;
  if (curr == null) return null;
  if (prevStep && prevStep.metrics && prevStep.metrics.compactness != null) {
    return curr - prevStep.metrics.compactness;
  }
  return curr;
}

function foldAssignment(fold) {
  if (!fold) return null;
  const t = fold.type || '';
  if (t === 'valley') return 'V';
  if (t === 'mountain') return 'M';
  if (t === 'pleat') return 'P';
  if (t === 'crimp') return 'C';
  return t.charAt(0).toUpperCase() || null;
}

function foldLabel(fold) {
  if (!fold) return '';
  const type = fold.type || 'fold';
  const angle = fold.angle != null ? ` ${fold.angle}°` : '';
  return `${type.toUpperCase()} FOLD${angle}`;
}

export default function StepFeed({ steps, currentStep }) {
  const feedRef = useRef(null);
  const activeRef = useRef(null);

  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [currentStep]);

  if (!steps || steps.length === 0) {
    return (
      <div className="step-feed">
        <div style={{ padding: '16px', color: 'var(--text-dim)', fontFamily: 'var(--font-display)', fontSize: '11px' }}>
          NO STEPS LOADED
        </div>
      </div>
    );
  }

  return (
    <div className="step-feed" ref={feedRef}>
      {steps.map((step, idx) => {
        const stepNum = idx + 1;
        const isActive = currentStep === stepNum;
        const delta = compactnessDelta(step, idx > 0 ? steps[idx - 1] : null);
        const asgn = foldAssignment(step.fold);
        const label = foldLabel(step.fold);
        const m = step.metrics || {};

        return (
          <div
            key={stepNum}
            className={`step-entry${isActive ? ' active' : ''}`}
            ref={isActive ? activeRef : null}
          >
            <div className="step-entry-top">
              <span className="step-num">#{stepNum}</span>
              <span className="step-instruction">{label}</span>
              {asgn && (
                <span className={`assign-badge ${asgn}`}>{asgn}</span>
              )}
            </div>
            {delta !== null && (
              <div className="step-reward-delta">
                {'\u0394'} compact:{' '}
                <span className={delta >= 0 ? 'delta-positive' : 'delta-negative'}>
                  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
                </span>
                {m.max_strain != null && (
                  <span style={{ color: 'var(--text-dim)' }}>
                    {' '}| strain: {m.max_strain.toFixed(4)}
                    {m.is_valid != null && (
                      <span> | {m.is_valid ? '✓' : '✗'}</span>
                    )}
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
