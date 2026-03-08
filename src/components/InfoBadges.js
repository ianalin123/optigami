function BoolVal({ value }) {
  if (value === null || value === undefined) {
    return <span className="info-val dim">—</span>;
  }
  return (
    <span className={`info-val ${value ? 'bool-true' : 'bool-false'}`}>
      {value ? 'TRUE' : 'FALSE'}
    </span>
  );
}

function TextVal({ value, dim = false }) {
  if (value === null || value === undefined) {
    return <span className="info-val dim">—</span>;
  }
  return (
    <span className={`info-val${dim ? ' dim' : ''}`}>
      {String(value).toUpperCase()}
    </span>
  );
}

function NumVal({ value }) {
  if (value === null || value === undefined) {
    return <span className="info-val dim">—</span>;
  }
  return <span className="info-val">{value}</span>;
}

export default function InfoBadges({ metrics, paperState, targetDef }) {
  const numLayers = paperState?.num_layers ?? metrics?.num_layers ?? null;
  const foldCount = metrics?.fold_count ?? paperState?.fold_count ?? null;

  return (
    <div className="info-badges">
      <div className="info-row">
        <span className="info-key">fold_count</span>
        <NumVal value={foldCount} />
      </div>
      <div className="info-row">
        <span className="info-key">num_layers</span>
        <NumVal value={numLayers} />
      </div>
      <div className="info-row">
        <span className="info-key">is_valid</span>
        <BoolVal value={metrics ? metrics.is_valid : null} />
      </div>
      <div className="info-row">
        <span className="info-key">strain_exceeded</span>
        <BoolVal value={metrics ? metrics.strain_exceeded : null} />
      </div>
      <div className="info-row">
        <span className="info-key">is_deployable</span>
        <BoolVal value={metrics ? metrics.is_deployable : null} />
      </div>
      {targetDef && (
        <>
          <div className="info-row">
            <span className="info-key">level</span>
            <span className="info-val">LVL {targetDef.level}</span>
          </div>
          <div className="info-row">
            <span className="info-key">material</span>
            <TextVal value={targetDef.material} dim={true} />
          </div>
          <div className="info-row">
            <span className="info-key">task</span>
            <span className="info-val" style={{ fontSize: '10px', textAlign: 'right', maxWidth: '140px', wordBreak: 'break-word' }}>
              {(targetDef.name || '').replace(/_/g, ' ').toUpperCase()}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
