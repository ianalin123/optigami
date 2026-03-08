import { useEffect, useRef } from 'react';

function rewardDelta(step, prevStep) {
  if (!step || !step.reward) return null;
  const curr = step.reward.total;
  if (prevStep && prevStep.reward) {
    return curr - prevStep.reward.total;
  }
  return curr;
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
        const delta = rewardDelta(step, idx > 0 ? steps[idx - 1] : null);
        const asgn = step.fold ? step.fold.assignment : null;
        const instruction = step.fold ? step.fold.instruction : (step.prompt || '');

        return (
          <div
            key={stepNum}
            className={`step-entry${isActive ? ' active' : ''}`}
            ref={isActive ? activeRef : null}
          >
            <div className="step-entry-top">
              <span className="step-num">#{stepNum}</span>
              <span className="step-instruction">{instruction}</span>
              {asgn && (
                <span className={`assign-badge ${asgn}`}>{asgn}</span>
              )}
            </div>
            {delta !== null && (
              <div className="step-reward-delta">
                {'\u0394'} total:{' '}
                <span className={delta >= 0 ? 'delta-positive' : 'delta-negative'}>
                  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
                </span>
                {step.reward && (
                  <span style={{ color: 'var(--text-dim)' }}>
                    {' '}| progress: {step.reward.progress.toFixed(2)}
                    {' '}| economy: {step.reward.economy.toFixed(2)}
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
