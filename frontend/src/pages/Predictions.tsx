import { useState, useEffect, useMemo } from 'react';
import { TrendingUp, Satellite, Clock, AlertCircle, Activity, Target, BarChart3 } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader } from '@/services/dataLoader';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts';

// Interfaces for data types
interface PredictionRow {
  sample_id: number;
  horizon_label: string;
  horizon_minutes: number;
  timestamp_current: string;
  timestamp_predicted: string;
  x_error_pred: number;
  y_error_pred: number;
  z_error_pred: number;
  satclockerror_pred: number;
}

interface ActualRow {
  utc_time: string;
  'x_error (m)': number;
  'y_error (m)': number;
  'z_error (m)': number;
  'satclockerror (m)': number;
}

interface ChartDataPoint {
  timestamp: string;
  displayTime: string;
  actual: number | null;
  predicted: number | null;
  uncertainty_upper?: number;
  uncertainty_lower?: number;
}

const ERROR_VARIABLES = [
  { key: 'x_error', label: 'X Error', color: '#0ea5e9' },
  { key: 'y_error', label: 'Y Error', color: '#8b5cf6' },
  { key: 'z_error', label: 'Z Error', color: '#10b981' },
  { key: 'satclockerror', label: 'Clock Error', color: '#f59e0b' },
] as const;

const HORIZONS = ['15min', '30min', '45min', '1h', '2h', '3h', '6h', '12h', '24h'];

export default function PredictionsPage() {
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [selectedVariable, setSelectedVariable] = useState<string>('x_error');
  const [selectedHorizon, setSelectedHorizon] = useState<string>('1h');
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [actuals, setActuals] = useState<ActualRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load data
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        
        // Load predictions CSV
        const predData = await dataLoader.loadCSV<PredictionRow>(
          `/data/predictions/${satellite}_Day8_Predictions.csv`
        );
        setPredictions(predData);
        
        // Load actuals CSV
        const actualData = await dataLoader.loadCSV<ActualRow>(
          `/data/processed/${satellite}_clean_15min.csv`
        );
        setActuals(actualData);
      } catch (err) {
        setError('Failed to load prediction data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [satellite]);

  // Process data for the chart
  const chartData = useMemo(() => {
    if (!predictions.length || !actuals.length) return [];

    // Get predictions for the selected horizon
    const horizonPredictions = predictions.filter(p => p.horizon_label === selectedHorizon);
    
    // Create a map of timestamp -> predicted value
    const predMap = new Map<string, number>();
    horizonPredictions.forEach(p => {
      const key = `${selectedVariable}_pred` as keyof PredictionRow;
      predMap.set(p.timestamp_predicted, p[key] as number);
    });

    // Create chart data points
    const data: ChartDataPoint[] = [];
    const variableKey = selectedVariable === 'satclockerror' 
      ? 'satclockerror (m)' 
      : `${selectedVariable} (m)`;

    actuals.forEach(actual => {
      const timestamp = actual.utc_time;
      const predicted = predMap.get(timestamp);
      const actualValue = actual[variableKey as keyof ActualRow] as number;
      
      // Only include points where we have data
      if (predicted !== undefined || actualValue !== undefined) {
        const displayTime = new Date(timestamp).toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
          hour12: false
        });
        
        data.push({
          timestamp,
          displayTime,
          actual: actualValue ?? null,
          predicted: predicted ?? null,
          uncertainty_upper: predicted !== undefined ? predicted + 0.3 : undefined,
          uncertainty_lower: predicted !== undefined ? predicted - 0.3 : undefined,
        });
      }
    });

    return data.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [predictions, actuals, selectedVariable, selectedHorizon]);

  // Calculate accuracy metrics
  const metrics = useMemo(() => {
    const validPoints = chartData.filter(d => d.actual !== null && d.predicted !== null);
    if (validPoints.length === 0) return null;

    const errors = validPoints.map(d => (d.actual as number) - (d.predicted as number));
    const absErrors = errors.map(e => Math.abs(e));
    const squaredErrors = errors.map(e => e * e);

    const mae = absErrors.reduce((a, b) => a + b, 0) / absErrors.length;
    const rmse = Math.sqrt(squaredErrors.reduce((a, b) => a + b, 0) / squaredErrors.length);
    const bias = errors.reduce((a, b) => a + b, 0) / errors.length;
    const correlation = calculateCorrelation(validPoints);

    return { mae, rmse, bias, correlation, count: validPoints.length };
  }, [chartData]);

  function calculateCorrelation(data: ChartDataPoint[]): number {
    const n = data.length;
    if (n < 2) return 0;
    
    const actuals = data.map(d => d.actual as number);
    const preds = data.map(d => d.predicted as number);
    
    const meanA = actuals.reduce((a, b) => a + b, 0) / n;
    const meanP = preds.reduce((a, b) => a + b, 0) / n;
    
    let num = 0, denA = 0, denP = 0;
    for (let i = 0; i < n; i++) {
      const diffA = actuals[i] - meanA;
      const diffP = preds[i] - meanP;
      num += diffA * diffP;
      denA += diffA * diffA;
      denP += diffP * diffP;
    }
    
    return denA * denP > 0 ? num / Math.sqrt(denA * denP) : 0;
  }

  const selectedVarInfo = ERROR_VARIABLES.find(v => v.key === selectedVariable)!;

  return (
    <PageLayout
      title="Day-8 Predictions"
      description="Actual vs Predicted comparison for the 8th day forecast"
    >
      {/* Control Panel */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <div className="grid md:grid-cols-3 gap-6">
          {/* Satellite Selector */}
          <div>
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
              Satellite Type
            </label>
            <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
              {(['MEO', 'GEO'] as const).map((sat) => (
                <button
                  key={sat}
                  onClick={() => setSatellite(sat)}
                  className={`flex-1 px-4 py-2.5 rounded-md text-sm font-medium transition-all ${
                    satellite === sat
                      ? 'bg-white dark:bg-slate-700 text-orbit-600 shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  <Satellite className="w-4 h-4 inline mr-2" />
                  {sat}
                </button>
              ))}
            </div>
          </div>

          {/* Variable Selector */}
          <div>
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
              Error Variable
            </label>
            <select
              value={selectedVariable}
              onChange={(e) => setSelectedVariable(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-100 dark:bg-slate-800 border-0 rounded-lg text-sm font-medium text-slate-900 dark:text-white focus:ring-2 focus:ring-orbit-500"
            >
              {ERROR_VARIABLES.map((v) => (
                <option key={v.key} value={v.key}>
                  {v.label}
                </option>
              ))}
            </select>
          </div>

          {/* Horizon Selector */}
          <div>
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
              Prediction Horizon
            </label>
            <select
              value={selectedHorizon}
              onChange={(e) => setSelectedHorizon(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-100 dark:bg-slate-800 border-0 rounded-lg text-sm font-medium text-slate-900 dark:text-white focus:ring-2 focus:ring-orbit-500"
            >
              {HORIZONS.map((h) => (
                <option key={h} value={h}>
                  {h} ahead
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin w-12 h-12 border-4 border-orbit-500 border-t-transparent rounded-full mx-auto mb-4" />
            <p className="text-slate-500">Loading prediction data...</p>
          </div>
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center h-96">
          <AlertCircle className="w-16 h-16 text-amber-500 mb-4" />
          <p className="text-lg text-slate-500">{error}</p>
        </div>
      ) : (
        <>
          {/* Metrics Cards */}
          {metrics && (
            <div className="grid md:grid-cols-4 gap-4 mb-8">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl p-5 border border-blue-200 dark:border-blue-800">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-5 h-5 text-blue-600" />
                  <span className="text-sm font-medium text-blue-700 dark:text-blue-300">RMSE</span>
                </div>
                <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                  {metrics.rmse.toFixed(4)} m
                </div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-xl p-5 border border-purple-200 dark:border-purple-800">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-purple-600" />
                  <span className="text-sm font-medium text-purple-700 dark:text-purple-300">MAE</span>
                </div>
                <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                  {metrics.mae.toFixed(4)} m
                </div>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-xl p-5 border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  <span className="text-sm font-medium text-green-700 dark:text-green-300">Correlation</span>
                </div>
                <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                  {(metrics.correlation * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 rounded-xl p-5 border border-amber-200 dark:border-amber-800">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-amber-600" />
                  <span className="text-sm font-medium text-amber-700 dark:text-amber-300">Data Points</span>
                </div>
                <div className="text-2xl font-bold text-amber-900 dark:text-amber-100">
                  {metrics.count}
                </div>
              </div>
            </div>
          )}

          {/* Main Chart - Actual vs Predicted */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                  Actual vs Predicted: {selectedVarInfo.label}
                </h3>
                <p className="text-sm text-slate-500 mt-1">
                  {satellite} Satellite | {selectedHorizon} Horizon | Day 8 Forecast
                </p>
              </div>
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-blue-500 rounded"></div>
                  <span className="text-slate-600 dark:text-slate-400">Actual</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-orange-500 rounded"></div>
                  <span className="text-slate-600 dark:text-slate-400">Predicted</span>
                </div>
              </div>
            </div>
            
            <div className="h-[500px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <defs>
                    <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f97316" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#f97316" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="displayTime" 
                    tick={{ fontSize: 11 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }} 
                    tickFormatter={(v) => v.toFixed(2)}
                    label={{ 
                      value: `${selectedVarInfo.label} (m)`, 
                      angle: -90, 
                      position: 'insideLeft',
                      style: { textAnchor: 'middle', fontSize: 12 }
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e2e8f0',
                      borderRadius: '12px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      padding: '12px 16px',
                    }}
                    labelStyle={{ fontWeight: 'bold', marginBottom: '8px' }}
                    formatter={(value, name) => {
                      if (value === null || value === undefined) return ['N/A', name];
                      const numValue = typeof value === 'number' ? value : parseFloat(String(value));
                      return [`${numValue.toFixed(4)} m`, name === 'actual' ? 'Actual' : 'Predicted'];
                    }}
                    labelFormatter={(label) => `Time: ${label}`}
                  />
                  <Legend 
                    verticalAlign="top"
                    height={36}
                    formatter={(value) => value === 'actual' ? 'Actual Value' : 'Predicted Value'}
                  />
                  <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                  
                  {/* Uncertainty band */}
                  <Area
                    type="monotone"
                    dataKey="uncertainty_upper"
                    stroke="none"
                    fill="url(#uncertaintyGradient)"
                    connectNulls={false}
                  />
                  
                  {/* Actual line - Blue */}
                  <Line 
                    type="monotone" 
                    dataKey="actual" 
                    name="actual"
                    stroke="#3b82f6" 
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                    connectNulls={false}
                  />
                  
                  {/* Predicted line - Orange */}
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    name="predicted"
                    stroke="#f97316" 
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 6, fill: '#f97316', stroke: '#fff', strokeWidth: 2 }}
                    connectNulls={false}
                    strokeDasharray="5 5"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Caption */}
            <div className="mt-6 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                <span className="font-semibold text-slate-800 dark:text-slate-200">Interpretation: </span>
                This plot compares the model's prediction for the 8th day with actual values. 
                Close overlap indicates that the model has learned GNSS error dynamics rather than memorizing the dataset.
                The dashed orange line shows predicted values, while the solid blue line shows actual measurements.
              </p>
            </div>
          </div>

          {/* Multi-variable overview */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              All Variables Overview ({selectedHorizon} horizon)
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              {ERROR_VARIABLES.map((variable) => {
                const varKey = variable.key === 'satclockerror' 
                  ? 'satclockerror (m)' 
                  : `${variable.key} (m)`;
                const horizonPreds = predictions.filter(p => p.horizon_label === selectedHorizon);
                const predMap = new Map<string, number>();
                horizonPreds.forEach(p => {
                  const key = `${variable.key}_pred` as keyof PredictionRow;
                  predMap.set(p.timestamp_predicted, p[key] as number);
                });

                const miniData = actuals
                  .filter(a => predMap.has(a.utc_time))
                  .slice(0, 50)
                  .map(a => ({
                    actual: a[varKey as keyof ActualRow] as number,
                    predicted: predMap.get(a.utc_time) as number,
                  }));

                return (
                  <div 
                    key={variable.key} 
                    className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                      selectedVariable === variable.key 
                        ? 'border-orbit-500 bg-orbit-50 dark:bg-orbit-900/20' 
                        : 'border-slate-200 dark:border-slate-700 hover:border-slate-300'
                    }`}
                    onClick={() => setSelectedVariable(variable.key)}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-medium text-slate-900 dark:text-white">
                        {variable.label}
                      </span>
                      <span 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: variable.color }}
                      />
                    </div>
                    <div className="h-24">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={miniData}>
                          <Line 
                            type="monotone" 
                            dataKey="actual" 
                            stroke="#3b82f6" 
                            strokeWidth={1.5}
                            dot={false}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="predicted" 
                            stroke="#f97316" 
                            strokeWidth={1.5}
                            dot={false}
                            strokeDasharray="3 3"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </PageLayout>
  );
}
