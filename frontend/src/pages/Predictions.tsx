import { useState, useEffect } from 'react';
import { TrendingUp, Satellite, Clock, AlertCircle } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader, PredictionData } from '@/services/dataLoader';
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

export default function PredictionsPage() {
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [predictions, setPredictions] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadPredictions() {
      try {
        setLoading(true);
        setError(null);
        const data = await dataLoader.loadPredictions(
          `/data/predictions/${satellite}_Day8_Predictions.json`
        );
        setPredictions(data);
      } catch (err) {
        setError('Failed to load predictions');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadPredictions();
  }, [satellite]);

  const chartData = predictions?.predictions.map((p) => ({
    horizon: p.horizon_label,
    x_error: p.x_error_pred,
    y_error: p.y_error_pred,
    z_error: p.z_error_pred,
    clock_error: p.satclockerror_pred,
  })) ?? [];

  return (
    <PageLayout
      title="Predictions"
      description="View multi-horizon satellite orbit error predictions"
    >
      {/* Satellite Selector */}
      <div className="flex items-center gap-4 mb-8">
        <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
          Select Satellite:
        </span>
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
          {/* Prediction Info */}
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="flex items-center gap-3 mb-2">
                <Satellite className="w-5 h-5 text-orbit-500" />
                <span className="text-sm text-slate-500">Satellite Type</span>
              </div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {predictions?.satellite_type}
              </div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="flex items-center gap-3 mb-2">
                <TrendingUp className="w-5 h-5 text-orbit-500" />
                <span className="text-sm text-slate-500">Predictions</span>
              </div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {predictions?.prediction_count}
              </div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="flex items-center gap-3 mb-2">
                <Clock className="w-5 h-5 text-orbit-500" />
                <span className="text-sm text-slate-500">Horizons</span>
              </div>
              <div className="text-sm font-medium text-slate-900 dark:text-white">
                {predictions?.horizons.join(', ')}
              </div>
            </div>
          </div>

          {/* Prediction Chart */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Predicted Errors by Horizon
            </h3>
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
                  <Line type="monotone" dataKey="x_error" name="X Error" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="y_error" name="Y Error" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="z_error" name="Z Error" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="clock_error" name="Clock Error" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Prediction Plot Image */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Day 8 Predictions Visualization ({satellite})
            </h3>
            <img
              src={`/data/predictions/${satellite}_Day8_Predictions.png`}
              alt={`${satellite} Day 8 Predictions`}
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
            />
          </div>

          {/* Prediction Table */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
            <div className="p-6 border-b border-slate-200 dark:border-slate-800">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Prediction Details
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Horizon</th>
                    <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">Timestamp</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">X Error (m)</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Y Error (m)</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Z Error (m)</th>
                    <th className="px-4 py-3 text-right font-medium text-slate-700 dark:text-slate-300">Clock Error (m)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                  {predictions?.predictions.map((p) => (
                    <tr key={p.horizon_label} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                      <td className="px-4 py-3 font-medium text-slate-900 dark:text-white">{p.horizon_label}</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">{p.timestamp_predicted}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{p.x_error_pred.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{p.y_error_pred.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{p.z_error_pred.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-600 dark:text-slate-400">{p.satclockerror_pred.toFixed(4)}</td>
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
