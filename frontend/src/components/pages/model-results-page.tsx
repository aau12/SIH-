import { Target, TrendingUp, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const performanceData = [
  { metric: 'RMSE', value: 2.34 },
  { metric: 'MAE', value: 1.87 },
  { metric: 'R²', value: 0.943 },
  { metric: 'MAPE', value: 3.2 },
];

const confusionMatrix = [
  [850, 12, 8],
  [15, 920, 10],
  [5, 8, 892],
];

export function ModelResultsPage() {
  return (
    <div className="space-y-6">
      {/* Performance Cards */}
      <div className="grid grid-cols-4 gap-6">
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-blue-100 rounded-xl">
              <Target className="w-5 h-5 text-blue-600" />
            </div>
            <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded-full">
              Excellent
            </span>
          </div>
          <p className="text-3xl text-gray-800 mb-1">2.34m</p>
          <p className="text-sm text-gray-600">RMSE</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-purple-100 rounded-xl">
              <TrendingUp className="w-5 h-5 text-purple-600" />
            </div>
            <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded-full">Good</span>
          </div>
          <p className="text-3xl text-gray-800 mb-1">1.87m</p>
          <p className="text-sm text-gray-600">MAE</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-green-100 rounded-xl">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded-full">
              High
            </span>
          </div>
          <p className="text-3xl text-gray-800 mb-1">94.3%</p>
          <p className="text-sm text-gray-600">R² Score</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-orange-100 rounded-xl">
              <Target className="w-5 h-5 text-orange-600" />
            </div>
            <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
              Low Error
            </span>
          </div>
          <p className="text-3xl text-gray-800 mb-1">3.2%</p>
          <p className="text-sm text-gray-600">MAPE</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Performance Metrics */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <h3 className="mb-4 text-gray-800">Performance Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis dataKey="metric" stroke="#6B7280" />
              <YAxis stroke="#6B7280" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #E5E7EB',
                  borderRadius: '12px',
                }}
              />
              <Bar dataKey="value" fill="#3B82F6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Confusion Matrix */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <h3 className="mb-4 text-gray-800">Confusion Matrix</h3>
          <div className="grid grid-cols-3 gap-2">
            {confusionMatrix.flat().map((value, index) => (
              <div
                key={index}
                className={`aspect-square flex items-center justify-center rounded-xl ${
                  index % 4 === 0
                    ? 'bg-green-500 text-white'
                    : value > 10
                    ? 'bg-orange-100 text-orange-700'
                    : 'bg-gray-100 text-gray-700'
                }`}
              >
                <span className="text-xl">{value}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 flex justify-center gap-6 text-xs text-gray-600">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span>Correct</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-orange-100 rounded"></div>
              <span>Misclassified</span>
            </div>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl p-6 text-white shadow-lg">
        <h3 className="mb-2">Model Training Complete</h3>
        <p className="text-green-100 text-sm mb-4">
          Model trained successfully with high accuracy and low error metrics
        </p>
        <div className="grid grid-cols-4 gap-6">
          <div>
            <p className="text-green-100 text-sm mb-1">Training Time</p>
            <p className="text-xl">2.5 hrs</p>
          </div>
          <div>
            <p className="text-green-100 text-sm mb-1">Epochs</p>
            <p className="text-xl">100</p>
          </div>
          <div>
            <p className="text-green-100 text-sm mb-1">Best Epoch</p>
            <p className="text-xl">87</p>
          </div>
          <div>
            <p className="text-green-100 text-sm mb-1">Validation Acc</p>
            <p className="text-xl">93.8%</p>
          </div>
        </div>
      </div>
    </div>
  );
}
