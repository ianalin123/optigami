import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import CreaseCanvas from './components/CreaseCanvas';
import RewardPanel from './components/RewardPanel';
import StepFeed from './components/StepFeed';
import InfoBadges from './components/InfoBadges';
import TargetSelector from './components/TargetSelector';
import PlayerControls from './components/PlayerControls';

const API_BASE = 'http://localhost:8000';

function App() {
  const [targets, setTargets] = useState({});
  const [selectedTarget, setSelectedTarget] = useState('half_horizontal');
  const [episode, setEpisode] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [apiStatus, setApiStatus] = useState('connecting'); // 'connecting' | 'ok' | 'err'
  const [episodeLoading, setEpisodeLoading] = useState(false);
  const intervalRef = useRef(null);

  const fetchTargets = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/targets`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTargets(data);
      setApiStatus('ok');
    } catch {
      setApiStatus('err');
    }
  }, []);

  const fetchDemoEpisode = useCallback(async (targetName) => {
    setEpisodeLoading(true);
    setPlaying(false);
    setCurrentStep(0);
    try {
      const res = await fetch(`${API_BASE}/episode/demo?target=${targetName}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setEpisode(data);
      setApiStatus('ok');
    } catch {
      setEpisode(null);
      setApiStatus('err');
    } finally {
      setEpisodeLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTargets();
  }, [fetchTargets]);

  useEffect(() => {
    fetchDemoEpisode(selectedTarget);
  }, [selectedTarget, fetchDemoEpisode]);

  const totalSteps = episode ? episode.steps.length : 0;

  // currentStep is 1-indexed for display (0 = "empty paper before any folds")
  // steps array is 0-indexed: steps[0] = result of fold 1
  const activeStepData = episode && currentStep > 0 ? episode.steps[currentStep - 1] : null;

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= totalSteps) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1500);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, totalSteps]);

  const handlePlay = () => {
    if (currentStep >= totalSteps) setCurrentStep(0);
    setPlaying(true);
  };
  const handlePause = () => setPlaying(false);
  const handleNext = () => {
    setPlaying(false);
    setCurrentStep(prev => Math.min(prev + 1, totalSteps));
  };
  const handlePrev = () => {
    setPlaying(false);
    setCurrentStep(prev => Math.max(prev - 1, 0));
  };
  const handleReset = () => {
    setPlaying(false);
    setCurrentStep(0);
  };

  const targetDef = targets[selectedTarget] || null;
  const targetFold = episode ? episode.target : null;

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">
          OPTI<span className="title-accent">GAMI</span> RL
        </span>
        <div className="header-sep" />
        <TargetSelector
          targets={targets}
          selected={selectedTarget}
          onChange={name => setSelectedTarget(name)}
        />
        <div className="header-sep" />
        <PlayerControls
          playing={playing}
          onPlay={handlePlay}
          onPause={handlePause}
          onNext={handleNext}
          onPrev={handlePrev}
          onReset={handleReset}
          currentStep={currentStep}
          totalSteps={totalSteps}
          disabled={!episode || episodeLoading}
        />
        <div className="header-right">
          <div className="api-status">
            <span className={`api-status-dot ${apiStatus === 'ok' ? 'ok' : apiStatus === 'err' ? 'err' : ''}`} />
            <span>{apiStatus === 'ok' ? 'API OK' : apiStatus === 'err' ? 'API ERR' : 'CONNECTING'}</span>
          </div>
        </div>
      </header>

      <div className="app-body">
        <div className="app-left">
          <div className="canvas-row">
            <div className="canvas-wrap">
              <span className="canvas-label">
                TARGET — {targetDef ? targetDef.name.replace(/_/g, ' ').toUpperCase() : '—'}
              </span>
              <CreaseCanvas
                paperState={null}
                target={targetFold}
                label="TARGET"
                dim={280}
                ghostOnly={true}
              />
            </div>
            <div className="canvas-wrap">
              <span className="canvas-label">
                {currentStep === 0 ? 'INITIAL STATE' : `STEP ${currentStep} / ${totalSteps}`}
              </span>
              <CreaseCanvas
                paperState={activeStepData ? activeStepData.paper_state : null}
                target={targetFold}
                label={currentStep === 0 ? 'INITIAL' : `STEP ${currentStep}`}
                dim={280}
                ghostOnly={false}
              />
            </div>
          </div>

          <div className="step-feed-section">
            <div className="section-header">FOLD SEQUENCE</div>
            {episodeLoading ? (
              <div className="episode-loading">
                <div className="pulse-dot" />
                FETCHING EPISODE...
              </div>
            ) : (
              <StepFeed
                steps={episode ? episode.steps : []}
                currentStep={currentStep}
              />
            )}
          </div>
        </div>

        <div className="app-right">
          <div className="section-header">REWARD DECOMPOSITION</div>
          <RewardPanel reward={activeStepData ? activeStepData.reward : null} />
          <div className="section-header">EPISODE INFO</div>
          <InfoBadges info={activeStepData ? activeStepData.info : null} targetDef={targetDef} />
        </div>
      </div>
    </div>
  );
}

export default App;
