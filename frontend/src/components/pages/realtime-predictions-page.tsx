import { useState, useEffect } from 'react';
import { Activity, Satellite, TrendingUp, AlertCircle, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface Prediction {
  satellite: string;
  timestamp_current: string;
  timestamp_predicted: string;
  horizon_label: string;
  horizon_minutes: number;
  x_error_pred: number;
  y_error_pred: number;
  z_error_pred: number;
  satclockerror_pred: number;
}

export function RealtimePredictionsPage() {
  const [satellite, setSatellite] = useState<'MEO' | 'GEO'>('MEO');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:8000/predict/${satellite}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPredictions(data);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
      console.error('Prediction fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [satellite]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchPredictions();
    }, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, satellite]);

  const chartData = predictions.map(pred => ({
    horizon: pred.horizon_label,
    minutes: pred.horizon_minutes,
    x_error: Math.abs(pred.x_error_pred),
    y_error: Math.abs(pred.y_error_pred),
    z_error: Math.abs(pred.z_error_pred),
    total_error: Math.sqrt(
      pred.x_error_pred ** 2 + 
      pred.y_error_pred ** 2 + 
      pred.z_error_pred ** 2
    )
  }));

  const currentPrediction = predictions[0];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Real-time Predictions</h2>
          <p className="text-gray-600">Live GNSS orbit error forecasting</p>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-100 text-green-700 border border-green-200'
                : 'bg-gray-100 text-gray-700 border border-gray-200'
            }`}
          >
            <Activity className={`w-4 h-4 inline mr-2 ${autoRefresh ? 'animate-pulse' : ''}`} />
            {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </button>
          
          <button
            onClick={fetchPredictions}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 inline mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Satellite Selector */}
      <div className="flex gap-3">
        <button
          onClick={() => setSatellite('MEO')}
          className={`flex-1 p-4 rounded-xl border-2 transition-all ${
            satellite === 'MEO'
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-200 bg-white hover:border-gray-300'
          }`}
        >
          <Satellite className="w-6 h-6 mb-2 text-blue-600" />
          <div className="font-semibold">MEO Satellite</div>
          <div className="text-sm text-gray-600">Medium Earth Orbit</div>
        </button>
        
        <button
          onClick={() => setSatellite('GEO')}
          className={`flex-1 p-4 rounded-xl border-2 transition-all ${
            satellite === 'GEO'
              ? 'border-purple-500 bg-purple-50'
              : 'border-gray-200 bg-white hover:border-gray-300'
          }`}
        >
          <Satellite className="w-6 h-6 mb-2 text-purple-600" />
          <div className="font-semibold">GEO Satellite</div>
          <div className="text-sm text-gray-600">Geostationary Orbit</div>
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
          <div>
            <div className="font-semibold text-red-900">Connection Error</div>
            <div className="text-sm text-red-700">{error}</div>
            <div className="text-xs text-red-600 mt-1">
              Make sure the backend API is running at http://localhost:8000
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && predictions.length === 0 && (
        <div className="bg-white rounded-xl p-12 text-center">
          <RefreshCw className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <div className="text-gray-600">Loading predictions...</div>
        </div>
      )}

      {/* Current Prediction Card */}
      {currentPrediction && !loading && (
        <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-sm opacity-90">Next Prediction (15 min)</div>
              <div className="text-2xl font-bold">{currentPrediction.horizon_label}</div>
            </div>
            <TrendingUp className="w-8 h-8 opacity-80" />
          </div>
          
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-white/10 rounded-lg p-3 backdrop-blur">
              <div className="text-xs opacity-80">X Error</div>
              <div className="text-xl font-bold">{currentPrediction.x_error_pred.toFixed(2)}m</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3 backdrop-blur">
              <div className="text-xs opacity-80">Y Error</div>
              <div className="text-xl font-bold">{currentPrediction.y_error_pred.toFixed(2)}m</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3 backdrop-blur">
              <div className="text-xs opacity-80">Z Error</div>
              <div className="text-xl font-bold">{currentPrediction.z_error_pred.toFixed(2)}m</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3 backdrop-blur">
              <div className="text-xs opacity-80">Clock Error</div>
              <div className="text-xl font-bold">{currentPrediction.satclockerror_pred.toFixed(2)}ns</div>
            </div>
          </div>
          
          <div className="mt-4 text-xs opacity-75">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      )}

      {/* Predictions Chart */}
      {chartData.length > 0 && (
        <div className="bg-white rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Error Forecast by Horizon</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="horizon" />
              <YAxis label={{ value: 'Error (m)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="x_error" stroke="#3b82f6" name="X Error" strokeWidth={2} />
              <Line type="monotone" dataKey="y_error" stroke="#8b5cf6" name="Y Error" strokeWidth={2} />
              <Line type="monotone" dataKey="z_error" stroke="#ec4899" name="Z Error" strokeWidth={2} />
              <Line type="monotone" dataKey="total_error" stroke="#f59e0b" name="Total Error" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Predictions Table */}
      {predictions.length > 0 && (
        <div className="bg-white rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Horizon</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">X Error (m)</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Y Error (m)</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Z Error (m)</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total (m)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {predictions.map((pred, idx) => {
                  const totalError = Math.sqrt(
                    pred.x_error_pred ** 2 + 
                    pred.y_error_pred ** 2 + 
                    pred.z_error_pred ** 2
                  );
                  
                  return (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="font-medium text-gray-900">{pred.horizon_label}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {new Date(pred.timestamp_predicted).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                        {pred.x_error_pred.toFixed(3)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                        {pred.y_error_pred.toFixed(3)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                        {pred.z_error_pred.toFixed(3)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right">
                        <span className={`font-semibold ${
                          totalError < 2 ? 'text-green-600' :
                          totalError < 5 ? 'text-yellow-600' :
                          'text-red-600'
                        }`}>
                          {totalError.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
