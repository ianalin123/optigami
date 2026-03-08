export default function PlayerControls({
  playing,
  onPlay,
  onPause,
  onNext,
  onPrev,
  onReset,
  currentStep,
  totalSteps,
  disabled,
}) {
  const atStart = currentStep === 0;
  const atEnd = currentStep >= totalSteps;

  return (
    <div className="player-controls">
      <button
        className="ctrl-btn"
        onClick={onReset}
        disabled={disabled || atStart}
        title="Reset to start"
      >
        ⏮ RST
      </button>
      <button
        className="ctrl-btn"
        onClick={onPrev}
        disabled={disabled || atStart}
        title="Previous step"
      >
        ◀ PREV
      </button>
      <span className="ctrl-step-display">
        {disabled ? '—/—' : `${currentStep} / ${totalSteps}`}
      </span>
      <button
        className="ctrl-btn"
        onClick={onNext}
        disabled={disabled || atEnd}
        title="Next step"
      >
        NEXT ▶
      </button>
      <button
        className={`ctrl-btn play`}
        onClick={playing ? onPause : onPlay}
        disabled={disabled || (!playing && atEnd)}
        title={playing ? 'Pause' : 'Play'}
      >
        {playing ? '⏸ PAUSE' : '▶▶ PLAY'}
      </button>
    </div>
  );
}
