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
  RotateCcw,
  Route,
  Sparkles,
  Truck,
  TriangleAlert,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
// Static Amplify hosting loads each system independently; no server-side RSC routing.
/* eslint-disable @next/next/no-html-link-for-pages */
import RoadMap, { type MapStation } from './road-map';

import FleetMap from './fleet-map';
import { fleetTargets, type FleetProgress } from './fleet-routing';
import type { SupplyContext, SupplyPolicy } from './supply-routing';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

// The demo Lambda has concurrency one. Navigation can overlap a prior explanation.
async function fetchModel(url: string, init?: RequestInit): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, init);
    if (![429, 503].includes(response.status) || attempt >= 3) return response;
    await new Promise(resolve => setTimeout(resolve, 1100 * (attempt + 1)));
  }
}

const SHOW_DISPATCH_SIMULATION = true;


type PolicyId = 'balanced' | 'strict';
type ModeId = 'empty' | 'full_dock';
type MobileView = 'map' | 'actions' | 'insight';

type ModeOption = {
  id: ModeId;
  label: string;
  description?: string;
  short_label?: string;
  event_noun?: string;
  positive_outcome?: string;
};

type ModeValue = ModeId | ModeOption;

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
  mode?: ModeValue;
  modes?: ModeOption[];
  policies: Policy[];
  scenarios: Scenario[];
  default: {
    mode?: ModeId;
    decision_time: string;
    district: string;
    policy: PolicyId;
    action_limit: number;
  };
};

type Candidate = MapStation;

type PredictionResponse = {
  supply?: SupplyContext | null;
  mode: ModeValue;
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
    current_red?: number;
    scored_current_red?: number;
    unscored_current_red?: number;
    current_empty?: number;
    scored_current_empty?: number;
    unscored_current_empty?: number;
    current_full_dock?: number;
    scored_current_full_dock?: number;
    unscored_current_full_dock?: number;
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
  mode: ModeValue;
  target_time: string;
  results: Array<{
    station_id: number;
    still_red?: boolean | null;
    still_empty?: boolean | null;
    still_full_dock?: boolean | null;
  }>;
  summary: {
    evaluated_actions: number;
    hits: number;
    precision: number | null;
  };
  scoring_note: string;
};

type ShapFactor = {
  feature: string;
  label: string;
  value_display: string;
  contribution: number;
  direction: 'increase' | 'decrease';
};

type ExplainResponse = {
  mode: ModeValue;
  decision_time: string;
  target_time: string;
  station: {
    station_id: number;
    station_name: string;
    district: string;
    risk_probability: number;
  };
  shap: {
    base_value_raw: number;
    raw_score: number;
    sum_error: number;
    factors: ShapFactor[];
    disclaimer: string;
  };
  operational_summary: {
    provider: 'amazon_bedrock' | 'template';
    text: string;
    fallback_reason?: string | null;
  };
};

type RunInput = {
  mode?: ModeId;
  scenarioId?: string;
  policy?: PolicyId;
  actionLimit?: number;
};

const DEFAULT_MODES: ModeOption[] = [
  { id: 'empty', label: '缺車持續（主要）', description: '預測目前0車站點30分鐘後是否仍無車可借。' },
  { id: 'full_dock', label: '滿柱持續（輔助）', description: '預測目前0空位站點30分鐘後是否仍無位可還。' },
];

const MODE_COPY = {
  empty: {
    title: 'YouBike 持續缺車預警',
    subtitle: '新北市歷史回放 · 30分鐘動態決策',
    eventNoun: '缺車',
    currentState: '0車',
    candidateLabel: '可評估缺車站',
    triggerNote: '當下0車才進入模型',
    positiveOutcome: '仍為0車',
    positiveLegend: '仍缺車',
    recoveredOutcome: '已恢復有車',
    routeTitle: '優先關注站點',
    question: '哪些站30分鐘後仍可能維持0車，值得優先保留營運注意力。',
  },
  full_dock: {
    title: 'YouBike 持續滿柱預警',
    subtitle: '輔助模式 · 30分鐘無位可還風險',
    eventNoun: '滿柱',
    currentState: '0空位',
    candidateLabel: '可評估滿柱站',
    triggerNote: '當下0空位才進入模型',
    positiveOutcome: '仍為0空位',
    positiveLegend: '仍滿柱',
    recoveredOutcome: '已恢復空位',
    routeTitle: '優先調度順序',
    question: '哪些站30分鐘後仍可能維持0空位，值得列入輔助調度清單。',
  },
} satisfies Record<ModeId, Record<string, string>>;

function modeIdOf(value: ModeValue | undefined, fallback: ModeId = 'empty'): ModeId {
  if (typeof value === 'string') return value === 'full_dock' ? 'full_dock' : 'empty';
  return value?.id === 'full_dock' ? 'full_dock' : fallback;
}

function scoredCount(summary: PredictionResponse['summary'], mode: ModeId) {
  return summary.scored_current_red
    ?? (mode === 'full_dock' ? summary.scored_current_full_dock : summary.scored_current_empty)
    ?? 0;
}

function unscoredCount(summary: PredictionResponse['summary'], mode: ModeId) {
  return summary.unscored_current_red
    ?? (mode === 'full_dock' ? summary.unscored_current_full_dock : summary.unscored_current_empty)
    ?? 0;
}

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

function outcomeFor(reveal: RevealResponse | null, stationId: number, mode: ModeId) {
  const result = reveal?.results.find((item) => item.station_id === stationId);
  if (!result) return null;
  if (result.still_red !== undefined) return result.still_red;
  return mode === 'full_dock'
    ? result.still_full_dock ?? null
    : result.still_empty ?? null;
}

function RiskMap({ prediction, reveal, dispatchMode, targets, vehicles, supplyPolicy, handlingMinutes, vehicleCapacity, revision, selectedStationId, onSelectStation, onProgress }: {
  prediction: PredictionResponse; reveal: RevealResponse | null; dispatchMode: boolean;
  targets: Candidate[]; vehicles: number; supplyPolicy: SupplyPolicy; handlingMinutes: number; vehicleCapacity: number; revision: number;
  selectedStationId: number | null; onSelectStation: (id: number) => void; onProgress: (p: FleetProgress | null) => void;
}) {
  const mode = modeIdOf(prediction.mode), copy = MODE_COPY[mode];
  const outcomes = useMemo(() => new Map(prediction.candidates.map(station => [station.station_id, outcomeFor(reveal, station.station_id, mode)])), [prediction.candidates, reveal, mode]);
  const outcomeRecord = useMemo(() => Object.fromEntries(outcomes), [outcomes]);
  const selected = prediction.candidates.find(s => s.station_id === selectedStationId);
  const targetKey = targets.map(s => s.station_id).join(',');
  return <div id="mobile-map-panel" className="map-shell" aria-label={`${prediction.district}${copy.eventNoun}風險站點與道路路線`}>
    <div className="map-meta"><div><span className="eyebrow">{dispatchMode ? '多車取送規劃' : '道路與站點風險圖'}</span><strong>{prediction.district} · {formatLocalTime(prediction.decision_time)} 快照</strong></div><span className="live-pill"><Radio size={14} />歷史回放</span></div>
    {dispatchMode && prediction.supply ? <FleetMap key={`${prediction.decision_time}:${prediction.district}:${targetKey}:${vehicles}:${supplyPolicy}:${handlingMinutes}:${vehicleCapacity}:${revision}`} context={prediction.supply} stations={prediction.candidates} targets={targets} vehicles={vehicles} policy={supplyPolicy} handlingMinutes={handlingMinutes} vehicleCapacity={vehicleCapacity} selectedStationId={selectedStationId} onSelectStation={onSelectStation} outcomes={outcomeRecord} onProgress={onProgress} />
      : <RoadMap stations={prediction.candidates} actions={prediction.actions} selectedStationId={selectedStationId} onSelectStation={onSelectStation} outcomes={outcomes} simulatedStationIds={new Set<number>()} revealed={Boolean(reveal)} />}
    {selected && <div className="road-selection"><strong>{selected.station_name}</strong><span>風險 {riskText(selected.risk_probability)} · {selected.red_duration_display}</span></div>}
    <div className="map-legend"><span><i className="legend-dot action" />{dispatchMode ? '已規劃取送' : 'Top 10 行動清單'}</span><span><i className="legend-dot overflow-alert" />其他過門檻站</span><span><i className="legend-dot below-threshold" />未達門檻</span><span><i className="legend-dot unknown" />資料不足</span>{reveal && <><span><i className="legend-dot still-empty" />{copy.positiveLegend}</span><span><i className="legend-dot recovered" />{copy.recoveredOutcome}</span></>}</div>
  </div>;
}

function StationInsight({
  station,
  explanation,
  loading,
  error,
}: {
  station: Candidate | null;
  explanation: ExplainResponse | null;
  loading: boolean;
  error: string | null;
}) {
  const factors = explanation?.shap.factors ?? [];
  const largestContribution = Math.max(
    ...factors.map((factor) => Math.abs(factor.contribution)),
    0.0001,
  );
  const isBedrock = explanation?.operational_summary.provider === 'amazon_bedrock';

  return (
    <section id="mobile-insight-panel" className="insight-panel" aria-label="站點風險解釋與營運摘要">
      <div className="insight-heading">
        <div><span className="eyebrow">站點風險解釋</span><h2>{station?.station_name ?? '選擇一個站點'}</h2></div>
        {station?.risk_probability !== null && station && (
          <strong className="insight-risk">{riskText(station.risk_probability)}</strong>
        )}
      </div>

      <div className="insight-body">
        {!station && <div className="insight-empty">從行動清單或地圖選擇站點，即可查看模型原因。</div>}
        {station && loading && (
          <div className="insight-empty"><LoaderCircle size={22} className="spin" />正在整理模型原因</div>
        )}
        {station && error && !loading && (
          <div className="insight-error"><TriangleAlert size={16} />{error}</div>
        )}
        {station && explanation && !loading && (
          <>
            <div className="shap-title">
              <strong>模型原因｜SHAP</strong>
              <span>紅色推高風險，綠色降低風險</span>
            </div>
            <div className="factor-list">
              {factors.map((factor) => (
                <div className="factor-row" key={`${factor.feature}-${factor.direction}`}>
                  <div>
                    <strong>{factor.label}</strong>
                    <span>{factor.value_display}</span>
                  </div>
                  <div className="factor-meter" aria-label={`${factor.label}貢獻${factor.contribution}`}>
                    <i
                      className={factor.direction === 'increase' ? 'increase' : 'decrease'}
                      style={{ width: `${Math.max(12, Math.abs(factor.contribution) / largestContribution * 100)}%` }}
                    />
                  </div>
                  <b className={factor.direction === 'increase' ? 'increase' : 'decrease'}>
                    {factor.contribution > 0 ? '+' : ''}{factor.contribution.toFixed(2)}
                  </b>
                </div>
              ))}
            </div>
            <small className="shap-note">{explanation.shap.disclaimer}</small>

            <article className="ai-summary">
              <div>
                <span className="ai-mark"><Sparkles size={15} /></span>
                <strong>{isBedrock ? 'Amazon Bedrock 營運摘要' : '模型原因摘要'}</strong>
                <em>{isBedrock ? 'AI 整理' : '本機備援'}</em>
              </div>
              <p>{explanation.operational_summary.text}</p>
              {!isBedrock && (
                <small>尚未連接 Bedrock 時使用固定模板；預測與 SHAP 仍為真實模型結果。</small>
              )}
            </article>
          </>
        )}
      </div>

      <div className="model-flow" aria-label="模型處理流程">
        <span>LightGBM</span><i>→</i><span>SHAP</span><i>→</i>
        <span className={isBedrock ? 'active' : ''}>{isBedrock ? 'Amazon Bedrock' : '文字備援'}</span>
      </div>
    </section>
  );
}

export default function Dashboard({ dispatchMode = false }: { dispatchMode?: boolean }) {
  const [mode, setMode] = useState<ModeId>('empty');
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [scenarioId, setScenarioId] = useState('representative');
  const [policy, setPolicy] = useState<PolicyId>('balanced');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [reveal, setReveal] = useState<RevealResponse | null>(null);
  const [selectedStationId, setSelectedStationId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [revealing, setRevealing] = useState(false);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mobileView, setMobileView] = useState<MobileView>('map');
  const simulationEnabled = dispatchMode;
  const [dispatchProgress, setDispatchProgress] = useState<FleetProgress | null>(null);
  const [vehicles, setVehicles] = useState(3);
  const [candidateLimit, setCandidateLimit] = useState(20);
  const [includeLowRisk, setIncludeLowRisk] = useState(false);
  const [supplyPolicy, setSupplyPolicy] = useState<SupplyPolicy>('fixed5');
  const [handlingMinutes, setHandlingMinutes] = useState(3);
  const [vehicleCapacity, setVehicleCapacity] = useState(20);
  const [revision, setRevision] = useState(0);
  const selectedStationIdRef = useRef<number | null>(null);
  const predictionEpoch = useRef(0);
  const resetSimulation = useCallback(() => { setDispatchProgress(null); setRevision(x => x + 1); }, []);

  const selectStation = useCallback((stationId: number | null) => {
    if (selectedStationIdRef.current === stationId) return;
    selectedStationIdRef.current = stationId;
    setSelectedStationId(stationId);
    setExplanation(null);
    setExplainError(null);
    setExplainLoading(stationId !== null);
  }, []);

  const performPrediction = useCallback(async (
    requestedMode: ModeId,
    activeOptions: OptionsResponse,
    input: RunInput = {},
  ) => {
    const epoch = ++predictionEpoch.current;
    const requestedScenarioId = input.scenarioId ?? activeOptions.scenarios[0]?.id;
    const scenario = activeOptions.scenarios.find((item) => item.id === requestedScenarioId);
    if (!scenario) throw new Error('找不到指定的歷史情境');
    const requestedPolicy = input.policy ?? activeOptions.default.policy;
    const actionLimit = input.actionLimit ?? activeOptions.default.action_limit ?? 10;

    setLoading(true);
    setError(null);
    setReveal(null);
    setExplanation(null);
    setExplainError(null);
    setMobileView('map');
    resetSimulation();
    setIncludeLowRisk(false);
    selectStation(null);
    setScenarioId(requestedScenarioId);
    setPolicy(requestedPolicy);
    try {
      const response = await fetchModel(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: requestedMode,
          decision_time: scenario.decision_time,
          district: scenario.district,
          policy: requestedPolicy,
          action_limit: actionLimit,
        }),
      });
      const payload = await response.json() as PredictionResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? '預測服務沒有回應');
      if (epoch !== predictionEpoch.current) return;
      const loadedPrediction = payload as PredictionResponse;
      const responseMode = modeIdOf(loadedPrediction.mode, requestedMode);
      if (responseMode !== requestedMode) throw new Error('模型服務回傳了不同的預測模式');
      setMode(responseMode);
      setPrediction(loadedPrediction);
      selectStation(loadedPrediction.actions[0]?.station_id ?? null);
      return {
        mode: responseMode,
        scenario: scenario.label,
        policy: requestedPolicy,
        actionCount: (payload as PredictionResponse).summary.action_count,
      };
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : '無法連線預測服務';
      if (epoch === predictionEpoch.current) setError(message);
      throw requestError;
    } finally {
      if (epoch === predictionEpoch.current) setLoading(false);
    }
  }, [resetSimulation, selectStation]);

  const loadModeAndPredict = useCallback(async (
    requestedMode: ModeId,
    input: RunInput = {},
  ) => {
    try {
      const response = await fetchModel(
        `${API_BASE_URL}/api/options?mode=${encodeURIComponent(requestedMode)}`,
      );
      const payload = await response.json() as OptionsResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? '無法載入歷史情境');
      const loaded = payload as OptionsResponse;
      const responseMode = modeIdOf(loaded.mode, requestedMode);
      if (responseMode !== requestedMode) throw new Error('模型服務回傳了不同的模式設定');
      if (!loaded.scenarios.length) throw new Error('此模式目前沒有可回放的歷史情境');

      const requestedScenarioId = input.scenarioId ?? loaded.scenarios[0].id;
      const requestedPolicy = input.policy ?? loaded.default.policy;
      setMode(responseMode);
      setOptions(loaded);
      setScenarioId(requestedScenarioId);
      setPolicy(requestedPolicy);
      return await performPrediction(responseMode, loaded, {
        ...input,
        scenarioId: requestedScenarioId,
        policy: requestedPolicy,
      });
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : '無法切換預測模式';
      setError(message);
      throw requestError;
    } finally {
      setLoading(false);
    }
  }, [performPrediction]);

  const runPrediction = useCallback(async (input: RunInput = {}) => {
    const requestedMode = input.mode ?? mode;
    const activeOptions = options;
    if (!activeOptions || requestedMode !== mode) {
      return loadModeAndPredict(requestedMode, input);
    }
    return performPrediction(requestedMode, activeOptions, {
      ...input,
      scenarioId: input.scenarioId ?? scenarioId,
      policy: input.policy ?? policy,
    });
  }, [loadModeAndPredict, mode, options, performPrediction, policy, scenarioId]);

  const revealOutcome = useCallback(async () => {
    if (!prediction) return;
    const epoch = predictionEpoch.current;
    const ids = (dispatchMode ? fleetTargets(prediction.supply, prediction.candidates, 30, true) : prediction.actions).map(s => s.station_id);
    if (!ids.length) return;
    setRevealing(true); setError(null);
    try {
      const all: RevealResponse['results'] = [];
      let combined: RevealResponse | null = null;
      for (let offset = 0; offset < ids.length; offset += 25) {
        if (offset) await new Promise(resolve => setTimeout(resolve, 1100));
        const response = await fetch(`${API_BASE_URL}/api/reveal`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: modeIdOf(prediction.mode, mode), decision_time: prediction.decision_time, station_ids: ids.slice(offset, offset + 25) }),
        });
        const payload = await response.json() as RevealResponse & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? '無法揭曉歷史結果');
        if (modeIdOf(payload.mode, mode) !== modeIdOf(prediction.mode, mode) || payload.target_time !== prediction.target_time) throw new Error('揭曉結果與目前情境不一致');
        all.push(...payload.results); combined = payload;
      }
      const evaluated = all.filter(s => typeof (s.still_red ?? s.still_empty ?? s.still_full_dock) === 'boolean').length;
      const hits = all.filter(s => (s.still_red ?? s.still_empty ?? s.still_full_dock) === true).length;
      const result: RevealResponse = { ...combined!, results: all, summary: { evaluated_actions: evaluated, hits, precision: evaluated ? hits / evaluated : null } };
      if (epoch !== predictionEpoch.current) return;
      setReveal(result); return result.summary;
    } catch (e) { if (epoch === predictionEpoch.current) setError(e instanceof Error ? e.message : '無法揭曉歷史結果'); throw e; }
    finally { setRevealing(false); }
  }, [mode, prediction, dispatchMode]);

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
        const response = await fetchModel(`${API_BASE_URL}/api/options?mode=empty`);
        const payload = await response.json() as OptionsResponse & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? '無法載入歷史情境');
        if (!active) return;
        const loaded = payload as OptionsResponse;
        if (!loaded.scenarios.length) throw new Error('缺車模式目前沒有可回放的歷史情境');
        setMode('empty');
        setOptions(loaded);
        setScenarioId(loaded.scenarios[0].id);
        setPolicy(loaded.default.policy);
        await performPrediction('empty', loaded, {
          scenarioId: loaded.scenarios[0].id,
          policy: loaded.default.policy,
        });
      } catch (requestError) {
        if (active) setError(requestError instanceof Error ? requestError.message : '無法啟動Demo');
      } finally {
        if (active) setLoading(false);
      }
    }
    void initialize();
    return () => { active = false; };
  }, [performPrediction]);

  useEffect(() => {
    if (!prediction || selectedStationId === null) return;
    const lifecycle = new AbortController();

    void fetch(`${API_BASE_URL}/api/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: modeIdOf(prediction.mode, mode),
        decision_time: prediction.decision_time,
        district: prediction.district,
        station_id: selectedStationId,
        policy: prediction.policy.id,
        action_limit: prediction.action_limit,
      }),
      signal: lifecycle.signal,
    })
      .then(async (response) => {
        const payload = await response.json() as ExplainResponse & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? '無法載入模型原因');
        setExplanation(payload as ExplainResponse);
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === 'AbortError') return;
        setExplainError(requestError instanceof Error ? requestError.message : '無法載入模型原因');
      })
      .finally(() => {
        if (!lifecycle.signal.aborted) setExplainLoading(false);
      });

    return () => lifecycle.abort();
  }, [mode, prediction, selectedStationId]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool || !options) return;
    const lifecycle = new AbortController();

    void Promise.resolve(context.registerTool({
      name: 'run_youbike_prediction',
      title: '執行 YouBike 失衡持續預測',
      description: '選擇缺車或滿柱模式、歷史情境與警示政策，執行30分鐘持續風險預測。',
      inputSchema: {
        type: 'object',
        properties: {
          mode: { type: 'string', enum: ['empty', 'full_dock'] },
          scenarioId: { type: 'string', enum: options.scenarios.map((scenario) => scenario.id) },
          policy: { type: 'string', enum: ['balanced', 'strict'] },
          actionLimit: { type: 'integer', minimum: 1, maximum: 25 },
        },
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async (input: unknown) => {
        const value = (input ?? {}) as RunInput;
        if (value.mode && !['empty', 'full_dock'].includes(value.mode)) throw new Error('不支援的預測模式');
        if (value.policy && !['balanced', 'strict'].includes(value.policy)) throw new Error('不支援的警示政策');
        return runPredictionRef.current(value);
      },
    }, { signal: lifecycle.signal })).catch(() => undefined);

    void Promise.resolve(context.registerTool({
      name: 'reveal_youbike_outcome',
      title: '揭曉歷史結果',
      description: '揭曉目前畫面行動清單在30分鐘後是否仍維持所選失衡狀態，並更新可見結果。',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: async () => revealOutcomeRef.current(),
    }, { signal: lifecycle.signal })).catch(() => undefined);

    return () => lifecycle.abort();
  }, [options]);

  const activeScenario = options?.scenarios.find((item) => item.id === scenarioId);
  const activePolicy = options?.policies.find((item) => item.id === policy);
  const activeMode = modeIdOf(prediction?.mode, mode);
  const copy = MODE_COPY[activeMode];
  const modeOptions = options?.modes?.length ? options.modes : DEFAULT_MODES;
  const activeModeOption = modeOptions.find((item) => item.id === activeMode);
  const selectedStation =
    prediction?.candidates.find((station) => station.station_id === selectedStationId) ?? null;
  const orderedActionStations = useMemo(
    () => (prediction?.actions ?? [])
      .slice()
      .sort((left, right) => (left.route_order ?? 99) - (right.route_order ?? 99)),
    [prediction],
  );
  const targets = useMemo(() => fleetTargets(prediction?.supply, prediction?.candidates ?? [], candidateLimit, includeLowRisk), [prediction, candidateLimit, includeLowRisk]);
  const dispatchActive = Boolean(dispatchMode && prediction?.supply);
  const plannedIds = dispatchProgress?.plan.servedIds ?? [];
  const assignedIds = dispatchProgress?.started ? plannedIds : [];
  const assignedCount = assignedIds.length;
  const potentialImproved = reveal ? assignedIds.filter(id => outcomeFor(reveal, id, activeMode) === true).length : 0;
  const alreadyRecovered = reveal ? assignedIds.filter(id => outcomeFor(reveal, id, activeMode) === false).length : 0;
  const unknownSimulationOutcomes = assignedCount - potentialImproved - alreadyRecovered;
  const improvementReady = Boolean(dispatchProgress?.started && reveal && assignedCount && !unknownSimulationOutcomes);
  const improvementPercent = assignedCount ? `${Number((100 * potentialImproved / assignedCount).toFixed(1))}%` : '—';
  const displayedActions = dispatchMode ? targets : orderedActionStations;
  const labels = new Map(dispatchProgress?.plan.routes.flatMap(r => r.steps.map((s, i) => [s.targetId, `${r.vehicleId}-${i + 1}`] as const)) ?? []);
  const improvementHighlight = <div className="improvement-highlight" aria-label="派送命中率">
    <span className="improvement-label">30 分鐘派送命中率</span>
    <strong className="improvement-value">{improvementReady ? improvementPercent : '—'}</strong>
    <span className="improvement-count">{dispatchProgress?.started ? `已派送 ${assignedCount} 站中，${potentialImproved} 站歷史上仍缺車（${potentialImproved}/${assignedCount}）` : '確認路線並開始運送後計算'}</span>
    <small>{unknownSimulationOutcomes && reveal ? `${unknownSimulationOutcomes} 站答案未知，暫不顯示完整命中率。` : '命中代表找對持續缺車站，不等於實際改善成效。'}</small>
  </div>;


  return (
    <main className="app-shell" data-system={dispatchMode ? 'dispatch' : 'warning'}>
      <nav className="system-nav" aria-label="系統切換"><a href="/" aria-current={!dispatchMode ? 'page' : undefined}>預警系統</a><a href="/dispatch/" aria-current={dispatchMode ? 'page' : undefined}>派車系統</a></nav>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Bike size={22} /></span>
          <div>
            <strong>{dispatchMode ? 'YouBike 多車派送規劃' : copy.title}</strong>
            <span>{copy.subtitle}</span>
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
            <span>預測模式</span>
            <Select
              value={mode}
              onValueChange={(value) => {
                if (!value || value === mode) return;
                setMode(value as ModeId);
                setLoading(true);
                setError(null);
                setOptions(null);
                setPrediction(null);
                setReveal(null);
                setExplanation(null);
                setExplainError(null);
                setMobileView('map');
                selectStation(null);
                void loadModeAndPredict(value as ModeId).catch(() => undefined);
              }}
              disabled={loading || revealing}
            >
              <SelectTrigger className="control-select mode">
                <SelectValue>
                  {modeOptions.find((item) => item.id === mode)?.label ?? copy.eventNoun}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {modeOptions.map((item) => (
                  <SelectItem value={item.id} key={item.id}>{item.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="control-field">
            <span>歷史情境</span>
            <Select
              value={scenarioId}
              onValueChange={(value) => {
                if (!value) return;
                predictionEpoch.current++; resetSimulation();
                setScenarioId(value);
                setPrediction(null);
                setReveal(null);
                setMobileView('map');
              }}
              disabled={!options || loading || revealing}
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
                predictionEpoch.current++; resetSimulation();
                setPolicy(value as PolicyId);
                setPrediction(null);
                setReveal(null);
                setMobileView('map');
              }}
              disabled={!options || loading || revealing}
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
            disabled={!options || loading || revealing}
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
        <small>{[activeModeOption?.description, activePolicy?.description].filter(Boolean).join(' · ')}</small>
      </section>

      <section className="metric-row" aria-label="預測摘要">
        <article className="metric-card">
          <span className="metric-icon red"><AlertTriangle size={18} /></span>
          <div>
            <span>{copy.candidateLabel}</span>
            <strong>{prediction ? scoredCount(prediction.summary, activeMode) : '—'}</strong>
            <small>{prediction ? `另有${unscoredCount(prediction.summary, activeMode)}站資料不足` : copy.triggerNote}</small>
          </div>
        </article>
        <article className="metric-card accent">
          <span className="metric-icon cyan"><Navigation size={18} /></span>
          <div>
            <span>{dispatchMode ? '已安排／納入考慮站數' : '建議關注'}</span>
            <strong>{dispatchMode ? `${plannedIds.length}/${targets.length}` : prediction?.summary.action_count ?? '—'}</strong>
            <small>{prediction ? dispatchMode ? `${vehicles} 位派車員上限 · 每條路線含作業最多 30 分鐘` : `${prediction.summary.alerts_before_limit}站過門檻，最多顯示${prediction.action_limit}站` : '先過門檻，再套行動上限'}</small>
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
            <span>{reveal ? simulationEnabled ? '30分鐘派送命中率' : '本案例逐站命中' : '最長已持續'}</span>
            <strong>{reveal ? simulationEnabled ? improvementReady ? improvementPercent : '—' : `${reveal.summary.hits}/${reveal.summary.evaluated_actions}` : compactDuration(prediction?.summary.longest_red_duration)}</strong>
            <small>{reveal && dispatchMode ? `已派送 ${assignedCount} 站 · 歷史仍缺車 ${potentialImproved} 站` : reveal?.summary.precision !== null && reveal ? `本次命中率 ${(reveal.summary.precision * 100).toFixed(1)}%` : '每30分鐘重新評估'}</small>
          </div>
        </article>
      </section>

      {prediction ? (
        <>
          <div className="mobile-view-tabs" role="tablist" aria-label="手機版資訊切換">
            <button
              type="button"
              role="tab"
              aria-selected={mobileView === 'map'}
              aria-controls="mobile-map-panel"
              className={mobileView === 'map' ? 'active' : ''}
              onClick={() => setMobileView('map')}
            >
              <MapPinned size={15} />地圖
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mobileView === 'actions'}
              aria-controls="mobile-actions-panel"
              className={mobileView === 'actions' ? 'active' : ''}
              onClick={() => setMobileView('actions')}
            >
              <Route size={15} />行動清單
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mobileView === 'insight'}
              aria-controls="mobile-insight-panel"
              className={mobileView === 'insight' ? 'active' : ''}
              onClick={() => setMobileView('insight')}
            >
              <Sparkles size={15} />模型解釋
            </button>
          </div>

          <section className="workspace" data-mobile-view={mobileView}>
            <RiskMap
              prediction={prediction}
              reveal={reveal}
              dispatchMode={dispatchMode} targets={targets} vehicles={vehicles} supplyPolicy={supplyPolicy} handlingMinutes={handlingMinutes} vehicleCapacity={vehicleCapacity} revision={revision}
              selectedStationId={selectedStationId}
              onSelectStation={selectStation}
              onProgress={setDispatchProgress}
            />

            <div className="side-stack">
              <aside id="mobile-actions-panel" className="action-panel">
              <div className="panel-heading">
                <div><span className="eyebrow">行動清單</span><h2>{dispatchMode ? '候選站・風險優先' : copy.routeTitle}</h2></div>
                <span className="count-badge">{displayedActions.length} 站</span>
              </div>
              <div className="station-list">
                {displayedActions.length ? displayedActions.map((station, index) => {
                  const outcome = outcomeFor(reveal, station.station_id, activeMode);
                  return (
                    <button
                      className={`station-row ${selectedStationId === station.station_id ? 'active' : ''}`}
                      key={station.station_id}
                      onClick={() => {
                        selectStation(station.station_id);
                        setMobileView(dispatchActive ? 'map' : 'insight');
                      }}
                    >
                      <span className="route-number">{dispatchMode ? labels.get(station.station_id) ?? index + 1 : station.route_order}</span>
                      <span className="station-copy">
                        <strong>{station.station_name}</strong>
                        <span>
                          {station.red_duration_display}
                          {dispatchMode ? labels.has(station.station_id) ? ' · 已安排' : ' · 尚未安排' : ''}
                        </span>
                      </span>
                      <span className="risk-copy">
                        {outcome === true && <AlertTriangle size={15} className="outcome-still-empty" />}
                        {outcome === false && <CheckCircle2 size={15} className="outcome-recovered" />}
                        <strong>{riskText(station.risk_probability)}</strong>
                        <span>{outcome === true ? copy.positiveOutcome : outcome === false ? copy.recoveredOutcome : '持續風險'}</span>
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
              {SHOW_DISPATCH_SIMULATION && dispatchMode && (
                <section className="dispatch-simulator active" aria-label="車隊派送設定">
                  <div className="dispatch-simulator-heading"><span className="dispatch-icon"><Truck size={16} /></span><div><strong>車隊派送設定</strong><small>風險優先 greedy · 每站 Top 3 供車候選</small></div><Button variant="outline" className="simulation-toggle" onClick={resetSimulation}><RotateCcw size={14} />重設規劃</Button></div>
                  <div className="dispatch-simulator-body fleet-settings">
                    <label>派車員／車輛數<select id="fleet-worker-select" value={vehicles} onChange={e => { setVehicles(Number(e.target.value)); resetSimulation(); }}>{[1,2,3,4,5].map(n => <option key={n} value={n}>{n} 位</option>)}</select></label>
                    <label>納入規劃的候選站數<select id="fleet-target-select" value={candidateLimit} onChange={e => { setCandidateLimit(Number(e.target.value)); resetSimulation(); }}>{[10,20,30].map(n => <option key={n} value={n}>前 {n} 站（依風險）</option>)}</select></label>
                    <label className="fleet-low-risk-toggle"><input id="fleet-include-low-risk" type="checkbox" checked={includeLowRisk} onChange={e => { setIncludeLowRisk(e.target.checked); resetSimulation(); }} />納入低風險站（未達警示門檻）</label>
                    <label>補車與保留庫存<select id="fleet-stock-select" value={supplyPolicy} onChange={e => { setSupplyPolicy(e.target.value as SupplyPolicy); resetSimulation(); }}><option value="fixed5">至少 5 台</option><option value="hybrid20">5 台與容量 20% 取較大值</option></select></label>
                    <label>每目標作業時間（分鐘）<input id="fleet-handling-input" type="number" min="0" max="60" step="1" value={handlingMinutes} onChange={e => { setHandlingMinutes(Math.max(0, Math.min(60, Number(e.target.value) || 0))); resetSimulation(); }} /></label>
                    <label>每車載量（台）<select id="fleet-capacity-select" value={vehicleCapacity} onChange={e => { setVehicleCapacity(Number(e.target.value)); resetSimulation(); }}>{[5,10,15,20,30,40,50].map(n => <option key={n} value={n}>{n} 台</option>)}</select></label>
                    <small>{includeLowRisk ? '已納入低於警示門檻的備選站。' : `只納入達到目前警示門檻 ${(prediction.policy.threshold * 100).toFixed(1)}% 的站。`}本次最多考慮 {candidateLimit} 站，實際符合 {targets.length} 站；不為湊滿站數加入其他站。{vehicles} 位派送員再依庫存、載量與每車 30 分鐘限制，安排可完成的站。</small>
                    {!dispatchActive && <output>此情境沒有可用庫存快照，請切換至 6/29 09:00 缺車情境。</output>}
                    {!reveal ? <p className="simulation-pending">先按地圖上方「自動安排取送」，確認路線後按「開始運送」。目前已規劃 {plannedIds.length} 站，已啟動 {assignedCount} 站。</p> : <div className="simulation-result">{improvementHighlight}<p>{alreadyRecovered} 站歷史上已恢復；{unknownSimulationOutcomes} 站結果未知。未派送站不列入命中率分母。</p><p>若準時補車後不再失衡，這次最多可能改善 {potentialImproved} 站；這是情境假設，歷史答案保持不變。</p></div>}
                    <small className="simulation-disclaimer">每車空車從第一個供車站出發；相同供車站的連續取送可在載量內合併。每目標預設 3 分鐘含分攤取車、裝卸作業。未計車庫出發、同站排隊、即時交通與途中新借還。</small>
                  </div>
                </section>
              )}
              <div className="panel-footer">
                <Button
                  variant="outline"
                  className="reveal-button"
                  disabled={!(dispatchMode ? targets.length : prediction.actions.length) || loading || revealing || Boolean(reveal)}
                  onClick={() => void revealOutcome().catch(() => undefined)}
                >
                  {revealing ? <LoaderCircle size={16} className="spin" /> : <Eye size={16} />}
                  {reveal ? '已揭曉30分鐘後結果' : `揭曉 ${formatLocalTime(prediction.target_time)} 結果`}
                </Button>
                <span>每站各算一次，不採整段事件灌水</span>
              </div>
              </aside>

              <StationInsight
                station={selectedStation}
                explanation={explanation}
                loading={explainLoading}
                error={explainError}
              />
            </div>
          </section>
        </>
      ) : (
        <section className="workspace-placeholder">
          {loading ? <LoaderCircle size={30} className="spin" /> : <MapPinned size={30} />}
          <strong>{loading ? `正在讀取${copy.eventNoun}模型與六月快照` : '選好情境後執行預測'}</strong>
          <span>風險分數由凍結模型即時計算；站點與真值來自六月歷史資料。</span>
        </section>
      )}

      <section className="explain-bar">
        <div className="explain-icon"><MapPinned size={20} /></div>
        <div>
          <strong>此刻已經亮紅燈，模型回答的是：</strong>
          <span>{copy.question}</span>
        </div>
        <div className="method-chip"><CheckCircle2 size={15} />LightGBM · 68項凍結特徵</div>
      </section>
    </main>
  );
}
