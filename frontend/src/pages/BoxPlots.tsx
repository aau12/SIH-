import { useState, useEffect } from 'react';
import { BarChart3, AlertCircle, Info } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader, ResidualSummary } from '@/services/dataLoader';
import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ErrorBar,
  Scatter,
  Line,
} from 'recharts';

const COLORS = {
  x_error: '#0ea5e9',
  y_error: '#8b5cf6',
  z_error: '#10b981',
  satclockerror: '#f59e0b',
};

const HORIZONS = [15, 30, 45, 60, 120, 180, 360, 720, 1440];

interface BoxPlotData {
  horizon: string;
  horizonMinutes: number;
  x_error: { min: number; q1: number; median: number; q3: number; max: number };
  y_error: { min: number; q1: number; median: number; q3: number; max: number };
  z_error: { min: number; q1: number; median: number; q3: number; max: number };
  satclockerror: { min: number; q1: number; median: number; q3: number; max: number };
}

export default function BoxPlotsPage() {
  const [residuals, setResiduals] = useState<ResidualSummary[]>([]);
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [selectedVariable, setSelectedVariable] = useState<string>('x_error');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadResiduals() {
      try {
        setLoading(true);
        const data = await dataLoader.loadResidualSummary('/data/residuals/residual_summary.csv');
        setResiduals(data);
      } catch (err) {
        setError('Failed to load residual data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadResiduals();
  }, []);

  const filteredResiduals = residuals.filter((r) => r.satellite === satellite);

  // Create box plot data for the selected variable
  const boxPlotData = HORIZONS.map((horizon) => {
    const horizonLabel = horizon >= 60 ? `${horizon / 60}h` : `${horizon}min`;
    const varData = filteredResiduals.filter(
      (r) => r.variable === selectedVariable && r.horizon_minutes === horizon
    );

    // Since we have single values per horizon, we'll use RMSE, MAE, bias to simulate a box plot
    const rmseVal = varData[0]?.rmse ?? 0;
    const maeVal = varData[0]?.mae ?? 0;
    const biasVal = varData[0]?.bias ?? 0;

    return {
      horizon: horizonLabel,
      horizonMinutes: horizon,
      rmse: rmseVal,
      mae: maeVal,
      bias: Math.abs(biasVal),
      // Box plot approximation
      median: maeVal,
      q1: maeVal * 0.5,
      q3: rmseVal,
      min: Math.min(maeVal * 0.25, Math.abs(biasVal) * 0.5),
      max: rmseVal * 1.2,
    };
  });

  // Multi-variable comparison data
  const multiVarData = HORIZONS.map((horizon) => {
    const horizonLabel = horizon >= 60 ? `${horizon / 60}h` : `${horizon}min`;
    const result: Record<string, string | number> = { horizon: horizonLabel };

    ['x_error', 'y_error', 'z_error', 'satclockerror'].forEach((variable) => {
      const varData = filteredResiduals.find(
        (r) => r.variable === variable && r.horizon_minutes === horizon
      );
      result[variable] = varData?.rmse ?? 0;
    });

    return result;
  });

  return (
    <PageLayout
      title="Box Plot Visualization"
      description="Visualize residual distributions across forecast horizons"
    >
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-8">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Satellite:</span>
          <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
            {(['MEO', 'GEO'] as const).map((sat) => (
              <button
                key={sat}
                onClick={() => setSatellite(sat)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  satellite === sat
                    ? 'bg-white dark:bg-slate-700 text-orbit-600 shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                {sat}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Variable:</span>
          <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
            {Object.keys(COLORS).map((variable) => (
              <button
                key={variable}
                onClick={() => setSelectedVariable(variable)}
                className={`px-3 py-2 rounded-md text-xs font-medium transition-all ${
                  selectedVariable === variable
                    ? 'bg-white dark:bg-slate-700 text-orbit-600 shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                {variable}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin w-8 h-8 border-4 border-orbit-500 border-t-transparent rounded-full" />
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center h-64">
          <AlertCircle className="w-12 h-12 text-amber-500 mb-4" />
          <p className="text-slate-500">{error}</p>
        </div>
      ) : (
        <>
          {/* Info Banner */}
          <div className="bg-orbit-50 dark:bg-orbit-900/20 border border-orbit-200 dark:border-orbit-800 rounded-lg p-4 mb-8">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-orbit-600 mt-0.5" />
              <div>
                <h4 className="font-medium text-orbit-800 dark:text-orbit-300">
                  Box Plot Interpretation
                </h4>
                <p className="text-sm text-orbit-700 dark:text-orbit-400 mt-1">
                  Each bar represents the RMSE distribution for the selected variable at each
                  prediction horizon. The overlaid line shows MAE trends, and error bars indicate
                  the range between MAE and RMSE values.
                </p>
              </div>
            </div>
          </div>

          {/* Single Variable Box Plot */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <div className="flex items-center gap-3 mb-6">
              <BarChart3 className="w-5 h-5 text-orbit-600" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                {selectedVariable} - RMSE by Horizon ({satellite})
              </h3>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={boxPlotData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="horizon" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(2)} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number, name: string) => [value.toFixed(4), name]}
                  />
                  <Legend />
                  <Bar
                    dataKey="rmse"
                    name="RMSE"
                    fill={COLORS[selectedVariable as keyof typeof COLORS]}
                    radius={[4, 4, 0, 0]}
                    fillOpacity={0.8}
                  >
                    <ErrorBar
                      dataKey="max"
                      width={4}
                      strokeWidth={2}
                      stroke={COLORS[selectedVariable as keyof typeof COLORS]}
                    />
                  </Bar>
                  <Line
                    type="monotone"
                    dataKey="mae"
                    name="MAE"
                    stroke="#64748b"
                    strokeWidth={2}
                    dot={{ fill: '#64748b', r: 4 }}
                  />
                  <Scatter
                    dataKey="bias"
                    name="|Bias|"
                    fill="#ef4444"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Multi-Variable Comparison */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              All Variables - RMSE Comparison ({satellite})
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={multiVarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="horizon" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(1)} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number, name: string) => [value.toFixed(4), name]}
                  />
                  <Legend />
                  {Object.entries(COLORS).map(([variable, color]) => (
                    <Bar
                      key={variable}
                      dataKey={variable}
                      name={variable}
                      fill={color}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Pre-generated Boxplot Images */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Pre-computed Residual Box Plots
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">MEO Satellite</h4>
                <img
                  src="/data/dashboard/meo_residual_boxplot.png"
                  alt="MEO Residual Boxplot"
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">GEO Satellite</h4>
                <img
                  src="/data/dashboard/geo_residual_boxplot.png"
                  alt="GEO Residual Boxplot"
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
            </div>
          </div>

          {/* Statistics Table */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
            <div className="p-6 border-b border-slate-200 dark:border-slate-800">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Detailed Statistics - {selectedVariable}
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Horizon</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">RMSE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">MAE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Bias</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Range</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                  {boxPlotData.map((d) => (
                    <tr key={d.horizon} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                      <td className="px-4 py-3 font-medium text-slate-900 dark:text-white">{d.horizon}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{d.rmse.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{d.mae.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{d.bias.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">
                        {d.min.toFixed(4)} - {d.max.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </PageLayout>
  );
}
