const METRIC_FIELDS = [
  {
    key: 'compactness',
    label: 'compactness',
    color: 'var(--progress)',
    normalize: (v) => Math.min(Math.max(v || 0, 0), 1),
    format: (v) => (v != null ? v.toFixed(3) : '—'),
  },
  {
    key: 'max_strain',
    label: 'max strain',
    color: 'var(--validity)',
    // Show as inverted bar: low strain = small bar (good)
    normalize: (v) => Math.min((v || 0) / 0.2, 1),
    format: (v) => (v != null ? v.toFixed(4) : '—'),
    inverted: true,
  },
  {
    key: 'kawasaki_violations',
    label: 'kawasaki',
    color: 'var(--validity)',
    normalize: (v) => Math.min((v || 0) / 5, 1),
    format: (v) => (v != null ? String(v) : '—'),
    inverted: true,
  },
  {
    key: 'maekawa_violations',
    label: 'maekawa',
    color: 'var(--validity)',
    normalize: (v) => Math.min((v || 0) / 5, 1),
    format: (v) => (v != null ? String(v) : '—'),
    inverted: true,
  },
  {
    key: 'fits_target_box',
    label: 'fits box',
    color: 'var(--progress)',
    normalize: (v) => (v ? 1 : 0),
    format: (v) => (v == null ? '—' : v ? 'YES' : 'NO'),
  },
  {
    key: 'is_deployable',
    label: 'deployable',
    color: 'var(--progress)',
    normalize: (v) => (v ? 1 : 0),
    format: (v) => (v == null ? '—' : v ? 'YES' : 'NO'),
  },
];

function RewardRow({ label, color, pct, formattedValue, isDash, inverted }) {
  const barColor = inverted && pct > 0 ? 'var(--validity)' : color;
  return (
    <div className="reward-row">
      <span className="reward-label">{label}</span>
      <div className="reward-track">
        <div
          className="reward-bar"
          style={{ width: `${isDash ? 0 : pct}%`, background: barColor }}
        />
      </div>
      <span className={`reward-value${isDash ? ' dim' : ''}`}>
        {formattedValue}
      </span>
    </div>
  );
}

export default function RewardPanel({ metrics }) {
  return (
    <div className="reward-panel">
      {METRIC_FIELDS.map(({ key, label, color, normalize, format, inverted }) => {
        const raw = metrics ? metrics[key] : undefined;
        const isDash = raw === null || raw === undefined;
        const pct = isDash ? 0 : normalize(raw) * 100;
        return (
          <RewardRow
            key={key}
            label={label}
            color={color}
            pct={pct}
            formattedValue={isDash ? '—' : format(raw)}
            isDash={isDash}
            inverted={!!inverted}
          />
        );
      })}
    </div>
  );
}
