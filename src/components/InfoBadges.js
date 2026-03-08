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

export default function InfoBadges({ info, targetDef }) {
  return (
    <div className="info-badges">
      <div className="info-row">
        <span className="info-key">n_creases</span>
        <NumVal value={info ? info.n_creases : (targetDef ? targetDef.n_creases : null)} />
      </div>
      <div className="info-row">
        <span className="info-key">interior_verts</span>
        <NumVal value={info ? info.n_interior_vertices : null} />
      </div>
      <div className="info-row">
        <span className="info-key">local_fold</span>
        <BoolVal value={info ? info.local_foldability : null} />
      </div>
      <div className="info-row">
        <span className="info-key">blb_sat</span>
        <BoolVal value={info ? info.blb_satisfied : null} />
      </div>
      <div className="info-row">
        <span className="info-key">global_fold</span>
        <TextVal
          value={info ? info.global_foldability : null}
          dim={true}
        />
      </div>
      {targetDef && (
        <>
          <div className="info-row">
            <span className="info-key">level</span>
            <span className="info-val">LVL {targetDef.level}</span>
          </div>
          <div className="info-row">
            <span className="info-key">target</span>
            <span className="info-val" style={{ fontSize: '10px', textAlign: 'right', maxWidth: '140px', wordBreak: 'break-word' }}>
              {targetDef.name.replace(/_/g, ' ').toUpperCase()}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
