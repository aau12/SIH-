import { BarChart3 } from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const residualVsFitted = Array.from({ length: 100 }, () => ({
  fitted: Math.random() * 10 + 2,
  residual: (Math.random() - 0.5) * 4,
}));

const errorDistribution = Array.from({ length: 20 }, (_, i) => ({
  bin: i - 10,
  count: Math.floor(Math.random() * 50) + 10,
}));

export function ResidualAnalysisPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-purple-100 rounded-xl">
            <BarChart3 className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h2 className="text-gray-800">Residual Analysis</h2>
            <p className="text-sm text-gray-600">
              Analyze model prediction errors and residual patterns
            </p>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Residual vs Fitted */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <h3 className="mb-4 text-gray-800">Residual vs Fitted Values</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis
                dataKey="fitted"
                name="Fitted"
                stroke="#6B7280"
                label={{ value: 'Fitted Values', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                dataKey="residual"
                name="Residual"
                stroke="#6B7280"
                label={{ value: 'Residuals', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #E5E7EB',
                  borderRadius: '12px',
                }}
              />
              <Scatter data={residualVsFitted} fill="#3B82F6" />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Random scatter indicates good model fit
          </p>
        </div>

        {/* Error Distribution */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <h3 className="mb-4 text-gray-800">Error Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={errorDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis
                dataKey="bin"
                stroke="#6B7280"
                label={{ value: 'Error Bins', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                stroke="#6B7280"
                label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #E5E7EB',
                  borderRadius: '12px',
                }}
              />
              <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                {errorDistribution.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={Math.abs(entry.bin) < 2 ? '#10B981' : '#3B82F6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Normal distribution centered at zero
          </p>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-4 gap-6">
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 mb-2">Mean Residual</p>
          <p className="text-2xl text-gray-800">0.02m</p>
          <p className="text-xs text-green-600 mt-1">Nearly zero ✓</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 mb-2">Std Deviation</p>
          <p className="text-2xl text-gray-800">1.34m</p>
          <p className="text-xs text-blue-600 mt-1">Low variance ✓</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 mb-2">Skewness</p>
          <p className="text-2xl text-gray-800">-0.08</p>
          <p className="text-xs text-green-600 mt-1">Symmetric ✓</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 mb-2">Kurtosis</p>
          <p className="text-2xl text-gray-800">2.95</p>
          <p className="text-xs text-green-600 mt-1">Normal ✓</p>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl p-6 text-white shadow-lg">
        <h3 className="mb-2">Residual Analysis Summary</h3>
        <p className="text-purple-100 text-sm mb-4">
          Model residuals show excellent characteristics with no apparent patterns or biases
        </p>
        <div className="grid grid-cols-3 gap-6">
          <div>
            <p className="text-purple-100 text-sm mb-1">Normality Test</p>
            <p className="text-xl">Passed</p>
          </div>
          <div>
            <p className="text-purple-100 text-sm mb-1">Homoscedasticity</p>
            <p className="text-xl">Confirmed</p>
          </div>
          <div>
            <p className="text-purple-100 text-sm mb-1">Independence</p>
            <p className="text-xl">Verified</p>
          </div>
        </div>
      </div>
    </div>
  );
}
