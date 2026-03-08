const REWARD_FIELDS = [
  { key: 'kawasaki',   label: 'kawasaki',  color: 'var(--validity)' },
  { key: 'maekawa',   label: 'maekawa',   color: 'var(--validity)' },
  { key: 'blb',       label: 'blb',       color: 'var(--validity)' },
  { key: 'progress',  label: 'progress',  color: 'var(--progress)' },
  { key: 'economy',   label: 'economy',   color: 'var(--economy)' },
];

const TOTAL_FIELD = { key: 'total', label: 'total', color: 'var(--text-primary)' };

function RewardRow({ label, color, value }) {
  const isDash = value === null || value === undefined;
  const pct = isDash ? 0 : Math.min(Math.max(value, 0), 1) * 100;

  return (
    <div className="reward-row">
      <span className="reward-label">{label}</span>
      <div className="reward-track">
        <div
          className="reward-bar"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className={`reward-value${isDash ? ' dim' : ''}`}>
        {isDash ? '—' : value.toFixed(2)}
      </span>
    </div>
  );
}

export default function RewardPanel({ reward }) {
  return (
    <div className="reward-panel">
      {REWARD_FIELDS.map(({ key, label, color }) => (
        <RewardRow
          key={key}
          label={label}
          color={color}
          value={reward ? reward[key] : null}
        />
      ))}
      <div className="reward-divider" />
      <RewardRow
        label={TOTAL_FIELD.label}
        color={TOTAL_FIELD.color}
        value={reward ? reward[TOTAL_FIELD.key] : null}
      />
    </div>
  );
}
