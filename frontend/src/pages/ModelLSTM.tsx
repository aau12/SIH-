import { useState, useEffect } from 'react';
import { Brain, Layers, Activity, AlertCircle } from 'lucide-react';
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
} from 'recharts';

interface LSTMHistory {
  train_loss: number[];
  val_loss: number[];
  epoch_time: number[];
}

export default function ModelLSTMPage() {
  const [meoHistory, setMeoHistory] = useState<LSTMHistory | null>(null);
  const [geoHistory, setGeoHistory] = useState<LSTMHistory | null>(null);
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadHistory() {
      try {
        setLoading(true);
        const [meo, geo] = await Promise.all([
          dataLoader.loadJSON<LSTMHistory>('/data/models/metrics/lstm_meo_history.json'),
          dataLoader.loadJSON<LSTMHistory>('/data/models/metrics/lstm_geo_history.json'),
        ]);
        setMeoHistory(meo);
        setGeoHistory(geo);
      } catch (err) {
        setError('Failed to load LSTM training history');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadHistory();
  }, []);

  const history = satellite === 'MEO' ? meoHistory : geoHistory;

  const chartData = history?.train_loss.map((loss: number, idx: number) => ({
    epoch: idx + 1,
    train_loss: loss,
    val_loss: history.val_loss[idx],
  })) ?? [];

  return (
    <PageLayout
      title="LSTM Model"
      description="Long Short-Term Memory network for sequential orbit error prediction"
    >
      {/* Model Info */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-orbit-600" />
            </div>
            <span className="font-medium text-slate-900 dark:text-white">Architecture</span>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            LSTM (Long Short-Term Memory) is a recurrent neural network architecture designed for
            learning long-term dependencies in sequential data.
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
              <Layers className="w-5 h-5 text-orbit-600" />
            </div>
            <span className="font-medium text-slate-900 dark:text-white">Key Features</span>
          </div>
          <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
            <li>• Captures temporal patterns</li>
            <li>• Handles variable-length sequences</li>
            <li>• Memory cell for long-term dependencies</li>
          </ul>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-orbit-600" />
            </div>
            <span className="font-medium text-slate-900 dark:text-white">Training</span>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Trained with early stopping and learning rate scheduling for optimal convergence.
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
          {/* Training History Chart */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Training History ({satellite})
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 12 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(3)} label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => value.toFixed(6)}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" name="Training Loss" stroke="#0ea5e9" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_loss" name="Validation Loss" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Architecture */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Model Architecture
            </h3>
            <div className="grid md:grid-cols-4 gap-4">
              {[
                { layer: 'Input', details: 'Sequence of features' },
                { layer: 'LSTM Layer 1', details: '64 units, return_sequences=True' },
                { layer: 'LSTM Layer 2', details: '32 units' },
                { layer: 'Dense Output', details: '4 outputs (x, y, z, clock)' },
              ].map((item, idx) => (
                <div key={item.layer} className="relative">
                  <div className="p-4 bg-orbit-50 dark:bg-orbit-900/20 border border-orbit-200 dark:border-orbit-800 rounded-lg">
                    <div className="text-sm font-medium text-orbit-700 dark:text-orbit-300">{item.layer}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">{item.details}</div>
                  </div>
                  {idx < 3 && (
                    <div className="hidden md:block absolute top-1/2 -right-2 transform translate-x-full -translate-y-1/2 text-orbit-400">
                      →
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Training Statistics */}
          <div className="grid md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="text-sm text-slate-500 mb-1">Total Epochs</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{chartData.length}</div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="text-sm text-slate-500 mb-1">Final Train Loss</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {chartData[chartData.length - 1]?.train_loss.toFixed(4) ?? '-'}
              </div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="text-sm text-slate-500 mb-1">Final Val Loss</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {chartData[chartData.length - 1]?.val_loss.toFixed(4) ?? '-'}
              </div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <div className="text-sm text-slate-500 mb-1">Best Val Loss</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {Math.min(...(history?.val_loss ?? [0])).toFixed(4)}
              </div>
            </div>
          </div>

          {/* Training Plot Images */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
              Training Visualization ({satellite})
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">Standard LSTM</h4>
                <img
                  src={`/data/models/plots/lstm_${satellite.toLowerCase()}_training.png`}
                  alt={`LSTM ${satellite} Training`}
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">Improved LSTM</h4>
                <img
                  src={`/data/models/plots/lstm_improved_${satellite.toLowerCase()}_training.png`}
                  alt={`Improved LSTM ${satellite} Training`}
                  className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                />
              </div>
            </div>
          </div>
        </>
      )}
    </PageLayout>
  );
}
