import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Activity, BarChart3, TrendingDown, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader, ResidualSummary, ShapiroResult } from '@/services/dataLoader';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from 'recharts';

const COLORS = ['#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b'];
const VARIABLES = ['x_error', 'y_error', 'z_error', 'satclockerror'];

export default function ResidualsPage() {
  const [residuals, setResiduals] = useState<ResidualSummary[]>([]);
  const [shapiroResults, setShapiroResults] = useState<ShapiroResult[]>([]);
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [residualData, shapiroData] = await Promise.all([
          dataLoader.loadResidualSummary('/data/residuals/residual_summary.csv'),
          dataLoader.loadShapiroResults(`/data/${satellite}_shapiro.csv`)
        ]);
        setResiduals(residualData);
        setShapiroResults(shapiroData);
      } catch (err) {
        setError('Failed to load data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [satellite]);

  const filteredResiduals = residuals.filter((r) => r.satellite === satellite);

  const rmseByVariable = VARIABLES.map((variable) => {
    const varResiduals = filteredResiduals.filter((r) => r.variable === variable);
    const avgRmse = varResiduals.reduce((sum, r) => sum + r.rmse, 0) / (varResiduals.length || 1);
    return { variable, rmse: avgRmse };
  });

  const rmseByHorizon = [15, 30, 45, 60, 120, 180, 360, 720, 1440].map((horizon) => {
    const horizonData: Record<string, number | string> = { horizon: `${horizon}min` };
    VARIABLES.forEach((variable) => {
      const r = filteredResiduals.find((res) => res.variable === variable && res.horizon_minutes === horizon);
      horizonData[variable] = r?.rmse ?? 0;
    });
    return horizonData;
  });

  return (
    <PageLayout
      title="Residual Analysis"
      description="Analyze model residuals and error distributions"
    >
      {/* Navigation */}
      <div className="flex items-center gap-4 mb-8">
        <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
          {(['MEO', 'GEO'] as const).map((sat) => (
            <button
              key={sat}
              onClick={() => setSatellite(sat)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                satellite === sat
                  ? 'bg-white dark:bg-slate-700 text-orbit-600 shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              {sat}
            </button>
          ))}
        </div>
        <div className="ml-auto">
          <Link
            to="/residuals/boxplots"
            className="flex items-center gap-2 px-4 py-2 bg-orbit-500 hover:bg-orbit-600 text-white rounded-lg transition-colors"
          >
            <BarChart3 className="w-4 h-4" />
            View Box Plots
          </Link>
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
          {/* Summary Cards */}
          <div className="grid md:grid-cols-4 gap-4 mb-8">
            {rmseByVariable.map((item, idx) => (
              <div
                key={item.variable}
                className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6"
              >
                <div className="flex items-center gap-3 mb-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: COLORS[idx] }}
                  />
                  <span className="text-sm text-slate-500 font-mono">{item.variable}</span>
                </div>
                <div className="text-2xl font-bold text-slate-900 dark:text-white">
                  {item.rmse.toFixed(4)}
                </div>
                <div className="text-xs text-slate-500 mt-1">Avg RMSE (meters)</div>
              </div>
            ))}
          </div>

          {/* RMSE by Horizon Chart */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              RMSE by Prediction Horizon
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rmseByHorizon}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="horizon" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(1)} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => value.toFixed(4)}
                  />
                  <Legend />
                  {VARIABLES.map((variable, idx) => (
                    <Bar
                      key={variable}
                      dataKey={variable}
                      name={variable}
                      fill={COLORS[idx]}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Shapiro-Wilk Normality Test Results */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
              Shapiro-Wilk Normality Test ({satellite})
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
              Tests if residuals follow a normal distribution (p &gt; 0.05 = Normal). Sample size: 50 per horizon.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Horizon</th>
                    <th className="px-4 py-3 text-center font-medium text-slate-700 dark:text-slate-300">X Error</th>
                    <th className="px-4 py-3 text-center font-medium text-slate-700 dark:text-slate-300">Y Error</th>
                    <th className="px-4 py-3 text-center font-medium text-slate-700 dark:text-slate-300">Z Error</th>
                    <th className="px-4 py-3 text-center font-medium text-slate-700 dark:text-slate-300">Clock Error</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                  {shapiroResults.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                      <td className="px-4 py-3 font-medium text-slate-900 dark:text-white">{row.horizon_label}</td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          {row.normal_x_error === 'Yes' ? (
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-500" />
                          )}
                          <span className={`text-xs ${row.normal_x_error === 'Yes' ? 'text-green-600' : 'text-red-600'}`}>
                            p={row.p_x_error?.toFixed(3) || 'N/A'}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          {row.normal_y_error === 'Yes' ? (
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-500" />
                          )}
                          <span className={`text-xs ${row.normal_y_error === 'Yes' ? 'text-green-600' : 'text-red-600'}`}>
                            p={row.p_y_error?.toFixed(3) || 'N/A'}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          {row.normal_z_error === 'Yes' ? (
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-500" />
                          )}
                          <span className={`text-xs ${row.normal_z_error === 'Yes' ? 'text-green-600' : 'text-red-600'}`}>
                            p={row.p_z_error?.toFixed(3) || 'N/A'}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          {row.normal_satclockerror === 'Yes' ? (
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-500" />
                          )}
                          <span className={`text-xs ${row.normal_satclockerror === 'Yes' ? 'text-green-600' : 'text-red-600'}`}>
                            p={row.p_satclockerror?.toFixed(3) || 'N/A'}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 flex items-center gap-6 text-sm text-slate-500">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span>Normal (p &gt; 0.05)</span>
              </div>
              <div className="flex items-center gap-2">
                <XCircle className="w-4 h-4 text-red-500" />
                <span>Non-Normal (p ≤ 0.05)</span>
              </div>
            </div>
          </div>

          {/* Residual Boxplot from Dashboard */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Residual Box Plot ({satellite})
            </h3>
            <img
              src={`/data/dashboard/${satellite.toLowerCase()}_residual_boxplot.png`}
              alt={`${satellite} Residual Boxplot`}
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>

          {/* Drift Plots */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Drift Analysis ({satellite})
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">X Error Drift</h4>
                <img
                  src={`/data/residuals/drift_${satellite.toLowerCase()}_x_error.png`}
                  alt={`${satellite} X Error Drift`}
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Y Error Drift</h4>
                <img
                  src={`/data/residuals/drift_${satellite.toLowerCase()}_y_error.png`}
                  alt={`${satellite} Y Error Drift`}
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Z Error Drift</h4>
                <img
                  src={`/data/residuals/drift_${satellite.toLowerCase()}_z_error.png`}
                  alt={`${satellite} Z Error Drift`}
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Clock Error Drift</h4>
                <img
                  src={`/data/residuals/drift_${satellite.toLowerCase()}_satclockerror.png`}
                  alt={`${satellite} Clock Error Drift`}
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
            </div>
          </div>

          {/* Residual Table */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
            <div className="p-6 border-b border-slate-200 dark:border-slate-800">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Residual Summary ({satellite})
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Variable</th>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Horizon</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">RMSE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">MAE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Bias</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                  {filteredResiduals.slice(0, 20).map((r, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                      <td className="px-4 py-3 font-mono text-slate-900 dark:text-white">{r.variable}</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">{r.horizon_minutes}min</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{r.rmse.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{r.mae.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{r.bias.toFixed(4)}</td>
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
