function groupByLevel(targets) {
  const levels = {};
  Object.values(targets).forEach(t => {
    if (!levels[t.level]) levels[t.level] = [];
    levels[t.level].push(t);
  });
  return levels;
}

export default function TargetSelector({ targets, selected, onChange }) {
  const levels = groupByLevel(targets);
  const sortedLevels = Object.keys(levels).sort((a, b) => Number(a) - Number(b));

  return (
    <div className="target-selector">
      <span className="target-selector-label">TARGET</span>
      <select
        className="target-select"
        value={selected}
        onChange={e => onChange(e.target.value)}
      >
        {sortedLevels.length === 0 ? (
          <option value="">LOADING...</option>
        ) : (
          sortedLevels.map(level => (
            <optgroup key={level} label={`── LEVEL ${level}`}>
              {levels[level].map(t => (
                <option key={t.name} value={t.name}>
                  {t.name.replace(/_/g, ' ').toUpperCase()}
                </option>
              ))}
            </optgroup>
          ))
        )}
      </select>
    </div>
  );
}
