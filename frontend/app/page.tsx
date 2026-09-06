'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Bike,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  Eye,
  LoaderCircle,
  MapPinned,
  Navigation,
  Radio,
  Route,
  Sparkles,
  TriangleAlert,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

type PolicyId = 'balanced' | 'strict';

type Scenario = {
  id: string;
  label: string;
  decision_time: string;
  district: string;
  note: string;
};

type Policy = {
  id: PolicyId;
  label: string;
  threshold: number;
  description: string;
};

type OptionsResponse = {
  policies: Policy[];
  scenarios: Scenario[];
  default: {
    decision_time: string;
    district: string;
    policy: PolicyId;
    action_limit: number;
  };
};

type Candidate = {
  station_id: number;
  station_name: string;
  district: string;
  latitude: number | null;
  longitude: number | null;
  red_duration_display: string;
  status: 'scored' | 'insufficient_history';
  risk_probability: number | null;
  risk_rank: number | null;
  is_alert: boolean;
  in_action_list: boolean;
  route_order: number | null;
  distance_from_previous_km: number | null;
};

type PredictionResponse = {
  decision_time: string;
  target_time: string;
  district: string;
  action_limit: number;
  policy: {
    id: PolicyId;
    label: string;
    threshold: number;
  };
  summary: {
    current_empty: number;
    scored_current_empty: number;
    unscored_current_empty: number;
    alerts_before_limit: number;
    action_count: number;
    max_risk_probability: number | null;
    longest_red_duration: string;
  };
  candidates: Candidate[];
  actions: Candidate[];
  route_method: string;
};

type RevealResponse = {
  target_time: string;
  results: Array<{ station_id: number; still_empty: boolean | null }>;
  summary: {
    evaluated_actions: number;
    hits: number;
    precision: number | null;
  };
  scoring_note: string;
};

type RunInput = {
  scenarioId?: string;
  policy?: PolicyId;
  actionLimit?: number;
};

function formatLocalTime(value: string) {
  return new Intl.DateTimeFormat('zh-TW', {
    timeZone: 'Asia/Taipei',
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(new Date(value));
}

function formatFullTime(value: string) {
  return new Intl.DateTimeFormat('zh-TW', {
    timeZone: 'Asia/Taipei',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(new Date(value));
}

function riskText(value: number | null) {
  return value === null ? '未評分' : `${(value * 100).toFixed(1)}%`;
}

function compactDuration(value: string | undefined) {
  if (!value) return '—';
  if (value.includes('90分鐘以上')) return '≥90分';
  const minutes = value.match(/(\d+)分鐘/)?.[1];
  return minutes ? `${minutes}分` : value;
}

function projectCandidates(candidates: Candidate[]) {
  const located = candidates.filter(
    (station): station is Candidate & { latitude: number; longitude: number } =>
      station.latitude !== null && station.longitude !== null,
  );
  if (!located.length) return new Map<number, { x: number; y: number }>();

  const longitudes = located.map((station) => station.longitude);
  const latitudes = located.map((station) => station.latitude);
  const minLongitude = Math.min(...longitudes);
  const maxLongitude = Math.max(...longitudes);
  const minLatitude = Math.min(...latitudes);
  const maxLatitude = Math.max(...latitudes);
  const longitudeSpan = Math.max(maxLongitude - minLongitude, 0.002);
  const latitudeSpan = Math.max(maxLatitude - minLatitude, 0.002);

  return new Map(
    located.map((station) => [
      station.station_id,
      {
        x: 9 + ((station.longitude - minLongitude) / longitudeSpan) * 82,
        y: 8 + ((maxLatitude - station.latitude) / latitudeSpan) * 62,
      },
    ]),
  );
}

function outcomeFor(reveal: RevealResponse | null, stationId: number) {
  return reveal?.results.find((item) => item.station_id === stationId)?.still_empty ?? null;
}

function markerClass(station: Candidate, outcome: boolean | null) {
  if (station.in_action_list && outcome === true) return 'point still-empty';
  if (station.in_action_list && outcome === false) return 'point recovered';
  if (station.in_action_list) return 'point action';
  if (station.status === 'insufficient_history') return 'point unknown';
  if (station.is_alert) return 'point overflow-alert';
  return 'point below-threshold';
}

function RiskMap({
  prediction,
  reveal,
  selectedStationId,
}: {
  prediction: PredictionResponse;
  reveal: RevealResponse | null;
  selectedStationId: number | null;
}) {
  const positions = useMemo(() => projectCandidates(prediction.candidates), [prediction.candidates]);
  const routePoints = prediction.actions
    .slice()
    .sort((left, right) => (left.route_order ?? 99) - (right.route_order ?? 99))
    .map((station) => positions.get(station.station_id))
    .filter((position): position is { x: number; y: number } => Boolean(position))
    .map((position) => `${position.x},${position.y}`)
    .join(' ');
  const selected = prediction.candidates.find((station) => station.station_id === selectedStationId);

  return (
    <div className="map-shell" aria-label={`${prediction.district}缺車風險站點與巡補順序示意圖`}>
      <div className="map-meta">
        <div>
          <span className="eyebrow">區域風險圖</span>
          <strong>{prediction.district} · {formatLocalTime(prediction.decision_time)} 快照</strong>
        </div>
        <span className="live-pill"><Radio size={14} /> 歷史回放</span>
      </div>

      <div className="map-stage">
        <svg className="risk-map" viewBox="0 0 100 78" aria-label="站點座標與風險導向巡補順序">
          <defs>
            <linearGradient id="map-glow" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="#132c3a" />
              <stop offset="100%" stopColor="#071922" />
            </linearGradient>
            <filter id="point-glow" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="1.2" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <rect width="100" height="78" rx="3" fill="url(#map-glow)" />
          <path className="river" d="M-4 64 C18 54, 25 72, 48 62 S75 48, 105 53" />
          <g className="road-grid">
            <path d="M3 17 C22 25, 38 10, 97 28" />
            <path d="M8 43 C34 30, 56 45, 95 38" />
            <path d="M22 2 C27 22, 23 46, 35 76" />
            <path d="M58 1 C51 26, 72 44, 67 77" />
            <path d="M86 5 C72 22, 91 48, 81 74" />
          </g>
          <text x="7" y="74" className="map-label">實際站點經緯度 · 示意底圖</text>
          {routePoints && <polyline points={routePoints} className="route-halo" />}
          {routePoints && <polyline points={routePoints} className="route-line" />}
          {prediction.candidates.map((station) => {
            const position = positions.get(station.station_id);
            if (!position) return null;
            const outcome = outcomeFor(reveal, station.station_id);
            const selectedClass = station.station_id === selectedStationId ? ' selected' : '';
            return (
              <g
                key={station.station_id}
                className="station-marker"
                transform={`translate(${position.x} ${position.y})`}
                aria-label={`${station.station_name}，風險${riskText(station.risk_probability)}`}
              >
                <title>{station.station_name}：{riskText(station.risk_probability)}</title>
                <circle
                  r={station.in_action_list ? 3.4 : station.is_alert ? 2.25 : 1.75}
                  className={`${markerClass(station, outcome)}${selectedClass}`}
                  filter={station.in_action_list ? 'url(#point-glow)' : undefined}
                />
                {station.in_action_list && <text y="1.15" textAnchor="middle">{station.route_order}</text>}
              </g>
            );
          })}
        </svg>
        {selected && (
          <div className="map-popover">
            <span>{selected.status === 'scored' ? `風險排序 #${selected.risk_rank}` : '歷史長度不足'}</span>
            <strong>{selected.station_name}</strong>
            <small>{selected.red_duration_display} · {riskText(selected.risk_probability)}</small>
          </div>
        )}
      </div>

      <div className="map-legend">
        <span><i className="legend-dot action" />行動清單</span>
        {prediction.summary.alerts_before_limit > prediction.summary.action_count && (
          <span><i className="legend-dot overflow-alert" />過門檻但清單額滿</span>
        )}
        <span><i className="legend-dot below-threshold" />30分鐘風險未達門檻</span>
        <span><i className="legend-dot unknown" />資料不足</span>
        {reveal && <span><i className="legend-dot still-empty" />仍缺車</span>}
        {reveal && <span><i className="legend-dot recovered" />已恢復</span>}
        <span className="route-note"><Route size={14} />風險優先、鄰近串接；非實際派車指令</span>
      </div>
    </div>
  );
}

export default function Home() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [scenarioId, setScenarioId] = useState('representative');
  const [policy, setPolicy] = useState<PolicyId>('balanced');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [reveal, setReveal] = useState<RevealResponse | null>(null);
  const [selectedStationId, setSelectedStationId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [revealing, setRevealing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runPrediction = useCallback(async (input: RunInput = {}) => {
    const activeOptions = options;
    if (!activeOptions) throw new Error('歷史情境仍在載入');
    const requestedScenarioId = input.scenarioId ?? scenarioId;
    const scenario = activeOptions.scenarios.find((item) => item.id === requestedScenarioId);
    if (!scenario) throw new Error('找不到指定的歷史情境');
    const requestedPolicy = input.policy ?? policy;
    const actionLimit = input.actionLimit ?? 10;

    setLoading(true);
    setError(null);
    setReveal(null);
    setSelectedStationId(null);
    setScenarioId(requestedScenarioId);
    setPolicy(requestedPolicy);
    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision_time: scenario.decision_time,
          district: scenario.district,
          policy: requestedPolicy,
          action_limit: actionLimit,
        }),
      });
      const payload = await response.json() as PredictionResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? '預測服務沒有回應');
      setPrediction(payload as PredictionResponse);
      return {
        scenario: scenario.label,
        policy: requestedPolicy,
        actionCount: (payload as PredictionResponse).summary.action_count,
      };
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : '無法連線預測服務';
      setError(message);
      throw requestError;
    } finally {
      setLoading(false);
    }
  }, [options, policy, scenarioId]);

  const revealOutcome = useCallback(async () => {
    if (!prediction || !prediction.actions.length) throw new Error('目前沒有可揭曉的行動清單');
    setRevealing(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/reveal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision_time: prediction.decision_time,
          station_ids: prediction.actions.map((station) => station.station_id),
        }),
      });
      const payload = await response.json() as RevealResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? '無法揭曉歷史結果');
      setReveal(payload as RevealResponse);
      return (payload as RevealResponse).summary;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '無法揭曉歷史結果');
      throw requestError;
    } finally {
      setRevealing(false);
    }
  }, [prediction]);

  const runPredictionRef = useRef(runPrediction);
  const revealOutcomeRef = useRef(revealOutcome);
  useEffect(() => {
    runPredictionRef.current = runPrediction;
    revealOutcomeRef.current = revealOutcome;
  }, [revealOutcome, runPrediction]);

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/options`);
        const payload = await response.json() as OptionsResponse & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? '無法載入歷史情境');
        if (!active) return;
        const loaded = payload as OptionsResponse;
        setOptions(loaded);
        setScenarioId(loaded.scenarios[0]?.id ?? 'representative');
        setPolicy(loaded.default.policy);

        const scenario = loaded.scenarios[0];
        const predictionResponse = await fetch(`${API_BASE_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            decision_time: scenario.decision_time,
            district: scenario.district,
            policy: loaded.default.policy,
            action_limit: loaded.default.action_limit,
          }),
        });
        const predictionPayload = await predictionResponse.json() as PredictionResponse & { error?: string };
        if (!predictionResponse.ok) throw new Error(predictionPayload.error ?? '預測服務沒有回應');
        if (active) setPrediction(predictionPayload as PredictionResponse);
      } catch (requestError) {
        if (active) setError(requestError instanceof Error ? requestError.message : '無法啟動Demo');
      } finally {
        if (active) setLoading(false);
      }
    }
    void initialize();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool || !options) return;
    const lifecycle = new AbortController();

    void Promise.resolve(context.registerTool({
      name: 'run_youbike_prediction',
      title: '執行 YouBike 持續缺車預測',
      description: '選擇歷史情境與警示政策，執行同一個畫面上的30分鐘持續缺車預測。',
      inputSchema: {
        type: 'object',
        properties: {
          scenarioId: { type: 'string', enum: options.scenarios.map((scenario) => scenario.id) },
          policy: { type: 'string', enum: ['balanced', 'strict'] },
          actionLimit: { type: 'integer', minimum: 1, maximum: 25 },
        },
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async (input: unknown) => {
        const value = (input ?? {}) as RunInput;
        if (value.policy && !['balanced', 'strict'].includes(value.policy)) throw new Error('不支援的警示政策');
        return runPredictionRef.current(value);
      },
    }, { signal: lifecycle.signal })).catch(() => undefined);

    void Promise.resolve(context.registerTool({
      name: 'reveal_youbike_outcome',
      title: '揭曉歷史結果',
      description: '揭曉目前畫面行動清單在30分鐘後是否仍為0車，並更新可見結果。',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async () => revealOutcomeRef.current(),
    }, { signal: lifecycle.signal })).catch(() => undefined);

    return () => lifecycle.abort();
  }, [options]);

  const activeScenario = options?.scenarios.find((item) => item.id === scenarioId);
  const activePolicy = options?.policies.find((item) => item.id === policy);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Bike size={22} /></span>
          <div>
            <strong>YouBike 持續缺車預警</strong>
            <span>新北市歷史回放 · 30分鐘動態決策</span>
          </div>
        </div>
        <div className={`system-state ${error ? 'offline' : !options ? 'checking' : ''}`}>
          <span /> {error ? '服務需要檢查' : options ? '模型服務正常' : '正在連線模型'}
        </div>
      </header>

      <section className="control-strip" aria-label="回放條件">
        <div className="control-intro">
          <span className="eyebrow">決策時點</span>
          <div>
            <Clock3 size={18} />
            <strong>{prediction ? formatFullTime(prediction.decision_time) : '載入歷史快照'}</strong>
            <ChevronRight size={16} />
            <span>{prediction ? `預測 ${formatLocalTime(prediction.target_time)}` : 't + 30分鐘'}</span>
          </div>
        </div>
        <div className="control-fields">
          <div className="control-field">
            <span>歷史情境</span>
            <Select
              value={scenarioId}
              onValueChange={(value) => {
                if (!value) return;
                setScenarioId(value);
                setPrediction(null);
                setReveal(null);
              }}
              disabled={!options || loading}
            >
              <SelectTrigger className="control-select">
                <SelectValue>{activeScenario?.label ?? '選擇歷史情境'}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {options?.scenarios.map((scenario) => (
                  <SelectItem value={scenario.id} key={scenario.id}>{scenario.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="control-field">
            <span>警示政策</span>
            <Select
              value={policy}
              onValueChange={(value) => {
                if (!value) return;
                setPolicy(value as PolicyId);
                setPrediction(null);
                setReveal(null);
              }}
              disabled={!options || loading}
            >
              <SelectTrigger className="control-select small">
                <SelectValue>{activePolicy ? `${activePolicy.label} ${(activePolicy.threshold * 100).toFixed(1)}%` : '選擇政策'}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {options?.policies.map((item) => (
                  <SelectItem value={item.id} key={item.id}>
                    {item.label} {(item.threshold * 100).toFixed(1)}%
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button
            className="run-button"
            size="lg"
            disabled={!options || loading}
            onClick={() => void runPrediction().catch(() => undefined)}
          >
            {loading ? <LoaderCircle size={17} className="spin" /> : <Sparkles size={17} />}
            {loading ? '計算中' : '執行預測'}
          </Button>
        </div>
      </section>

      {error && (
        <section className="error-banner" role="alert">
          <TriangleAlert size={17} />
          <div><strong>Demo 尚未連上模型服務</strong><span>{error}</span></div>
        </section>
      )}

      <section className="scenario-note">
        <span>{activeScenario?.note ?? '使用六月未參與訓練的資料進行歷史回放。'}</span>
        <small>{activePolicy?.description}</small>
      </section>

      <section className="metric-row" aria-label="預測摘要">
        <article className="metric-card">
          <span className="metric-icon red"><AlertTriangle size={18} /></span>
          <div>
            <span>可評估缺車站</span>
            <strong>{prediction?.summary.scored_current_empty ?? '—'}</strong>
            <small>{prediction ? `另有${prediction.summary.unscored_current_empty}站資料不足` : '當下0車才進入模型'}</small>
          </div>
        </article>
        <article className="metric-card accent">
          <span className="metric-icon cyan"><Navigation size={18} /></span>
          <div>
            <span>建議關注</span>
            <strong>{prediction?.summary.action_count ?? '—'}</strong>
            <small>{prediction ? `${prediction.summary.alerts_before_limit}站過門檻，最多顯示${prediction.action_limit}站` : '先過門檻，再套行動上限'}</small>
          </div>
        </article>
        <article className="metric-card">
          <span className="metric-icon amber"><CircleDot size={18} /></span>
          <div>
            <span>最高持續風險</span>
            <strong>{riskText(prediction?.summary.max_risk_probability ?? null)}</strong>
            <small>校正後的30分鐘持續機率</small>
          </div>
        </article>
        <article className={`metric-card ${reveal ? 'revealed' : ''}`}>
          <span className="metric-icon blue">{reveal ? <CheckCircle2 size={18} /> : <Clock3 size={18} />}</span>
          <div>
            <span>{reveal ? '本案例逐站命中' : '最長已持續'}</span>
            <strong>{reveal ? `${reveal.summary.hits}/${reveal.summary.evaluated_actions}` : compactDuration(prediction?.summary.longest_red_duration)}</strong>
            <small>{reveal?.summary.precision !== null && reveal ? `本次命中率 ${(reveal.summary.precision * 100).toFixed(1)}%` : '每30分鐘重新評估'}</small>
          </div>
        </article>
      </section>

      {prediction ? (
        <section className="workspace">
          <RiskMap
            prediction={prediction}
            reveal={reveal}
            selectedStationId={selectedStationId}
          />

          <aside className="action-panel">
            <div className="panel-heading">
              <div><span className="eyebrow">行動清單</span><h2>優先巡補順序</h2></div>
              <span className="count-badge">{prediction.summary.action_count} 站</span>
            </div>
            <div className="station-list">
              {prediction.actions.length ? prediction.actions.map((station) => {
                const outcome = outcomeFor(reveal, station.station_id);
                return (
                  <button
                    className={`station-row ${selectedStationId === station.station_id ? 'active' : ''}`}
                    key={station.station_id}
                    onClick={() => setSelectedStationId(station.station_id)}
                  >
                    <span className="route-number">{station.route_order}</span>
                    <span className="station-copy">
                      <strong>{station.station_name}</strong>
                      <span>
                        {station.red_duration_display}
                        {station.distance_from_previous_km ? ` · 距前站${station.distance_from_previous_km}km` : ''}
                      </span>
                    </span>
                    <span className="risk-copy">
                      {outcome === true && <AlertTriangle size={15} className="outcome-still-empty" />}
                      {outcome === false && <CheckCircle2 size={15} className="outcome-recovered" />}
                      <strong>{riskText(station.risk_probability)}</strong>
                      <span>{outcome === true ? '仍為0車' : outcome === false ? '已恢復有車' : '持續風險'}</span>
                    </span>
                  </button>
                );
              }) : (
                <div className="empty-actions">
                  <CheckCircle2 size={28} />
                  <strong>沒有站點超過門檻</strong>
                  <span>系統不會為了湊滿名額加入低風險站。</span>
                </div>
              )}
            </div>
            <div className="panel-footer">
              <Button
                variant="outline"
                className="reveal-button"
                disabled={!prediction.actions.length || revealing || Boolean(reveal)}
                onClick={() => void revealOutcome().catch(() => undefined)}
              >
                {revealing ? <LoaderCircle size={16} className="spin" /> : <Eye size={16} />}
                {reveal ? '已揭曉30分鐘後結果' : `揭曉 ${formatLocalTime(prediction.target_time)} 結果`}
              </Button>
              <span>每站各算一次，不採整段事件灌水</span>
            </div>
          </aside>
        </section>
      ) : (
        <section className="workspace-placeholder">
          {loading ? <LoaderCircle size={30} className="spin" /> : <MapPinned size={30} />}
          <strong>{loading ? '正在讀取模型與六月快照' : '選好情境後執行預測'}</strong>
          <span>風險分數由凍結模型即時計算；站點與真值來自六月歷史資料。</span>
        </section>
      )}

      <section className="explain-bar">
        <div className="explain-icon"><MapPinned size={20} /></div>
        <div>
          <strong>此刻已經亮紅燈，模型回答的是：</strong>
          <span>哪些站30分鐘後仍可能維持0車，值得優先保留營運注意力。</span>
        </div>
        <div className="method-chip"><CheckCircle2 size={15} />LightGBM · 68項凍結特徵</div>
      </section>
    </main>
  );
}
