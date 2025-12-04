import { useState, useEffect } from 'react';
import { Zap, TrendingUp, Timer, AlertCircle } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader, ModelMetrics } from '@/services/dataLoader';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export default function ModelLightGBMPage() {
  const [meoMetrics, setMeoMetrics] = useState<ModelMetrics | null>(null);
  const [geoMetrics, setGeoMetrics] = useState<ModelMetrics | null>(null);
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMetrics() {
      try {
        setLoading(true);
        setError(null);
        
        // Explicit file paths - must load from these exact JSONs, no fallbacks
        const meoPath = '/data/models/metrics/lightgbm_meo_metrics.json';
        const geoPath = '/data/models/metrics/lightgbm_geo_metrics.json';
        
        console.log('[ModelLightGBM] Loading metrics from:', {
          meoPath,
          geoPath,
          timestamp: new Date().toISOString(),
        });
        
        // Load both metrics using loadModelMetrics (JSON only, no CSV fallback)
        const [meo, geo] = await Promise.all([
          dataLoader.loadModelMetrics(meoPath),
          dataLoader.loadModelMetrics(geoPath),
        ]);
        
        // Verify data structure and log sample values
        console.log('[ModelLightGBM] MEO metrics loaded:', {
          variables: Object.keys(meo),
          sampleHorizon_x_error_15min: meo['x_error (m)']?.['15min'],
          fullMeoData: meo,
        });
        
        console.log('[ModelLightGBM] GEO metrics loaded:', {
          variables: Object.keys(geo),
          sampleHorizon_x_error_15min: geo['x_error (m)']?.['15min'],
        });
        
        // Strict validation - must have expected structure
        if (!meo || !meo['x_error (m)'] || !meo['x_error (m)']['15min']) {
          throw new Error('MEO metrics JSON has invalid structure or is missing required fields');
        }
        
        if (!geo || !geo['x_error (m)'] || !geo['x_error (m)']['15min']) {
          throw new Error('GEO metrics JSON has invalid structure or is missing required fields');
        }
        
        setMeoMetrics(meo);
        setGeoMetrics(geo);
        
        console.log('[ModelLightGBM] Metrics successfully loaded and validated');
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        setError(`Failed to load LightGBM metrics JSON: ${errorMsg}`);
        console.error('[ModelLightGBM] Load error:', err);
        // Set null to ensure chart breaks if JSON is missing
        setMeoMetrics(null);
        setGeoMetrics(null);
      } finally {
        setLoading(false);
      }
    }
    loadMetrics();
  }, []);

  const metrics = satellite === 'MEO' ? meoMetrics : geoMetrics;
  const horizons = ['15min', '30min', '45min', '1h', '2h', '3h', '6h', '12h', '24h'];
  const variables = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)'];

  const chartData = horizons.map((horizon) => {
    const result: Record<string, string | number> = { horizon };
    variables.forEach((variable) => {
      const varMetrics = metrics?.[variable]?.[horizon];
      const valRmse = varMetrics?.val_rmse ?? 0;
      result[variable.split(' ')[0]] = valRmse;
      
      // Log chart values for debugging
      if (horizon === '15min' || horizon === '24h') {
        console.log(`[ModelLightGBM Chart] ${satellite} ${horizon} ${variable}: val_rmse=${valRmse}`);
      }
    });
    return result;
  });
  
  // Log full chart data for first render
  if (chartData.length > 0 && chartData[0].horizon === '15min') {
    console.log('[ModelLightGBM Chart] Full chart data:', chartData);
  }

  return (
    <PageLayout
      title="LightGBM Model"
      description="Gradient Boosting Decision Tree model performance and metrics"
    >
      {/* Model Info */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-orbit-600" />
            </div>
            <span className="font-medium text-slate-900 dark:text-white">Model Type</span>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses
            tree-based learning algorithms.
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-orbit-600" />
            </div>
            <span className="font-medium text-slate-900 dark:text-white">Key Features</span>
          </div>
          <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
            <li>• Fast training speed</li>
            <li>• Low memory usage</li>
            <li>• Handles large datasets</li>
          </ul>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
              <Timer className="w-5 h-5 text-orbit-600" />
            </div>
            <span className="font-medium text-slate-900 dark:text-white">Training</span>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Separate models trained for each prediction horizon and error variable.
          </p>
        </div>
      </div>

      {/* Satellite Selector */}
      <div className="flex items-center gap-4 mb-8">
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
          {/* Validation RMSE Chart */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Validation RMSE by Horizon ({satellite})
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="horizon" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(2)} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => value.toFixed(4)}
                  />
                  <Legend />
                  <Bar dataKey="x_error" name="X Error" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="y_error" name="Y Error" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="z_error" name="Z Error" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="satclockerror" name="Clock Error" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* RMSE vs Horizon Plot */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              RMSE vs Horizon ({satellite})
            </h3>
            <img
              src={`/data/dashboard/${satellite.toLowerCase()}_rmse_vs_horizon.png`}
              alt={`${satellite} RMSE vs Horizon`}
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>

          {/* Detailed Metrics Table */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
            <div className="p-6 border-b border-slate-200 dark:border-slate-800">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Detailed Metrics - {satellite}
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Variable</th>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Horizon</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Train RMSE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Val RMSE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Train MAE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Val MAE</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Time (s)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                  {variables.slice(0, 2).flatMap((variable) =>
                    horizons.slice(0, 5).map((horizon) => {
                      const m = metrics?.[variable]?.[horizon];
                      return (
                        <tr key={`${variable}-${horizon}`} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                          <td className="px-4 py-3 font-mono text-slate-900 dark:text-white">{variable.split(' ')[0]}</td>
                          <td className="px-4 py-3 text-slate-600 dark:text-slate-400">{horizon}</td>
                          <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{m?.train_rmse?.toFixed(4) ?? '-'}</td>
                          <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{m?.val_rmse?.toFixed(4) ?? '-'}</td>
                          <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{m?.train_mae?.toFixed(4) ?? '-'}</td>
                          <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{m?.val_mae?.toFixed(4) ?? '-'}</td>
                          <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{m?.training_time?.toFixed(3) ?? '-'}</td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </PageLayout>
  );
}
