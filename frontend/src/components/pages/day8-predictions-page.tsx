import { TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const predictionData = Array.from({ length: 8 }, (_, i) => ({
  day: `Day ${i + 1}`,
  xError: 3 + Math.random() * 2 + i * 0.3,
  yError: 2.5 + Math.random() * 2 + i * 0.25,
  zError: 4 + Math.random() * 2 + i * 0.35,
  clockError: 0.3 + Math.random() * 0.2 + i * 0.05,
}));

const satellitePredictions = Array.from({ length: 12 }, (_, i) => ({
  satellite: `SAT-${String(i + 1).padStart(3, '0')}`,
  type: i % 3 === 0 ? 'GEO' : 'MEO',
  day8X: (3 + Math.random() * 3).toFixed(2),
  day8Y: (2.5 + Math.random() * 2.5).toFixed(2),
  day8Z: (4 + Math.random() * 3).toFixed(2),
  day8Clock: (0.5 + Math.random() * 0.3).toFixed(3),
  confidence: (85 + Math.random() * 10).toFixed(1),
}));

export function Day8PredictionsPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-100 rounded-xl">
            <TrendingUp className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-gray-800">8-Day Error Predictions</h2>
            <p className="text-sm text-gray-600">Forecasted satellite errors for the next 8 days</p>
          </div>
        </div>
      </div>

      {/* Trend Chart */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
        <h3 className="mb-4 text-gray-800">Error Trend Forecast</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={predictionData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis dataKey="day" stroke="#6B7280" />
            <YAxis stroke="#6B7280" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #E5E7EB',
                borderRadius: '12px',
              }}
            />
            <Line type="monotone" dataKey="xError" stroke="#3B82F6" strokeWidth={2} name="X Error" />
            <Line type="monotone" dataKey="yError" stroke="#8B5CF6" strokeWidth={2} name="Y Error" />
            <Line type="monotone" dataKey="zError" stroke="#10B981" strokeWidth={2} name="Z Error" />
            <Line
              type="monotone"
              dataKey="clockError"
              stroke="#F59E0B"
              strokeWidth={2}
              name="Clock Error"
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex gap-6 justify-center mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-blue-500"></div>
            <span className="text-gray-600">X Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-purple-500"></div>
            <span className="text-gray-600">Y Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-green-500"></div>
            <span className="text-gray-600">Z Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-orange-500"></div>
            <span className="text-gray-600">Clock Error</span>
          </div>
        </div>
      </div>

      {/* Predictions Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-gray-800">Day-8 Predictions by Satellite</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Satellite</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Type</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">X Error (m)</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Y Error (m)</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Z Error (m)</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Clock (ns)</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {satellitePredictions.map((row) => (
                <tr key={row.satellite} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm text-gray-800">{row.satellite}</td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-2 py-1 rounded-full text-xs ${
                        row.type === 'GEO'
                          ? 'bg-purple-100 text-purple-700'
                          : 'bg-blue-100 text-blue-700'
                      }`}
                    >
                      {row.type}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">{row.day8X}</td>
                  <td className="px-6 py-4 text-sm text-gray-600">{row.day8Y}</td>
                  <td className="px-6 py-4 text-sm text-gray-600">{row.day8Z}</td>
                  <td className="px-6 py-4 text-sm text-gray-600">{row.day8Clock}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500"
                          style={{ width: `${row.confidence}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-600">{row.confidence}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
