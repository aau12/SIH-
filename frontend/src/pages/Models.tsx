import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Brain, Zap, Timer, Target, ChevronRight } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader, ModelMetrics } from '@/services/dataLoader';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const models = [
  {
    name: 'LightGBM',
    description: 'Gradient boosting framework using tree-based learning algorithms',
    path: '/models/lightgbm',
    icon: <Zap className="w-6 h-6" />,
    features: ['Fast training', 'Low memory usage', 'High accuracy', 'Feature importance'],
  },
  {
    name: 'LSTM',
    description: 'Long Short-Term Memory networks for sequence prediction',
    path: '/models/lstm',
    icon: <Brain className="w-6 h-6" />,
    features: ['Temporal patterns', 'Long dependencies', 'Sequential learning', 'Deep learning'],
  },
];

export default function ModelsPage() {
  const [meoMetrics, setMeoMetrics] = useState<ModelMetrics | null>(null);
  const [geoMetrics, setGeoMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const [meo, geo] = await Promise.all([
          dataLoader.loadModelMetrics('/data/models/metrics/lightgbm_meo_metrics.json'),
          dataLoader.loadModelMetrics('/data/models/metrics/lightgbm_geo_metrics.json'),
        ]);
        setMeoMetrics(meo);
        setGeoMetrics(geo);
      } catch (error) {
        console.error('Failed to load metrics:', error);
      } finally {
        setLoading(false);
      }
    }
    loadMetrics();
  }, []);

  const horizons = ['15min', '30min', '45min', '1h', '2h', '3h', '6h', '12h', '24h'];

  const chartData = horizons.map((horizon) => {
    const meoVal = meoMetrics?.['x_error (m)']?.[horizon]?.val_rmse ?? 0;
    const geoVal = geoMetrics?.['x_error (m)']?.[horizon]?.val_rmse ?? 0;
    return {
      horizon,
      meo: meoVal,
      geo: geoVal,
    };
  });

  return (
    <PageLayout
      title="Model Overview"
      description="Explore the machine learning models used for orbit error prediction"
    >
      {/* Model Cards */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {models.map((model) => (
          <Link
            key={model.name}
            to={model.path}
            className="block bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 hover:border-orbit-300 dark:hover:border-orbit-700 hover:shadow-lg transition-all group"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center text-orbit-600 group-hover:bg-orbit-500 group-hover:text-white transition-colors">
                  {model.icon}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                    {model.name}
                  </h3>
                  <p className="text-sm text-slate-500">{model.description}</p>
                </div>
              </div>
              <ChevronRight className="w-5 h-5 text-slate-400 group-hover:text-orbit-500 transition-colors" />
            </div>
            <div className="flex flex-wrap gap-2">
              {model.features.map((feature) => (
                <span
                  key={feature}
                  className="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 rounded-full text-xs"
                >
                  {feature}
                </span>
              ))}
            </div>
          </Link>
        ))}
      </div>

      {/* Performance Comparison */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
          RMSE by Prediction Horizon (X-Error)
        </h3>

        {loading ? (
          <div className="h-80 flex items-center justify-center">
            <div className="animate-spin w-8 h-8 border-4 border-orbit-500 border-t-transparent rounded-full" />
          </div>
        ) : (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
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
                <Line
                  type="monotone"
                  dataKey="meo"
                  name="MEO"
                  stroke="#0ea5e9"
                  strokeWidth={2}
                  dot={{ fill: '#0ea5e9', r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="geo"
                  name="GEO"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={{ fill: '#8b5cf6', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* RMSE Heatmaps */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
          RMSE Heatmaps
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">MEO Satellite</h4>
            <img
              src="/data/dashboard/meo_rmse_heatmap.png"
              alt="MEO RMSE Heatmap"
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>
          <div>
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">GEO Satellite</h4>
            <img
              src="/data/dashboard/geo_rmse_heatmap.png"
              alt="GEO RMSE Heatmap"
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>
        </div>
      </div>

      {/* MAE vs Horizon Plots */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
          MAE vs Prediction Horizon
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">MEO Satellite</h4>
            <img
              src="/data/dashboard/meo_mae_vs_horizon.png"
              alt="MEO MAE vs Horizon"
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>
          <div>
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">GEO Satellite</h4>
            <img
              src="/data/dashboard/geo_mae_vs_horizon.png"
              alt="GEO MAE vs Horizon"
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Timer className="w-5 h-5 text-orbit-500" />
            <h4 className="font-medium text-slate-900 dark:text-white">Training Time</h4>
          </div>
          <div className="text-3xl font-bold text-slate-900 dark:text-white">&lt;5s</div>
          <p className="text-sm text-slate-500 mt-1">Average per horizon</p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-5 h-5 text-orbit-500" />
            <h4 className="font-medium text-slate-900 dark:text-white">Best RMSE</h4>
          </div>
          <div className="text-3xl font-bold text-slate-900 dark:text-white">0.05m</div>
          <p className="text-sm text-slate-500 mt-1">15-minute horizon</p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-5 h-5 text-orbit-500" />
            <h4 className="font-medium text-slate-900 dark:text-white">Predictions</h4>
          </div>
          <div className="text-3xl font-bold text-slate-900 dark:text-white">36</div>
          <p className="text-sm text-slate-500 mt-1">Per satellite per run</p>
        </div>
      </div>
    </PageLayout>
  );
}
