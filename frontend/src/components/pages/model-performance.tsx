import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { TrendingUp, Target, Activity } from 'lucide-react';
import { StatCard } from '../stat-card';

const predictedVsActual = Array.from({ length: 30 }, (_, i) => ({
  index: i,
  predicted: Math.random() * 10 + 2,
  actual: Math.random() * 10 + 2,
}));

const residuals = Array.from({ length: 50 }, () => ({
  value: (Math.random() - 0.5) * 6,
}));

const featureImportance = [
  { feature: 'x_error_lag_1', importance: 0.23 },
  { feature: 'clock_lag_2', importance: 0.18 },
  { feature: 'rolling_mean_12', importance: 0.15 },
  { feature: 'z_error_lag_4', importance: 0.12 },
  { feature: 'y_error_lag_1', importance: 0.11 },
  { feature: 'diff1_x', importance: 0.09 },
  { feature: 'hour_sin', importance: 0.06 },
  { feature: 'rolling_std_6', importance: 0.04 },
  { feature: 'diff2_z', importance: 0.02 },
];

export function ModelPerformance() {
  return (
    <div className="p-6">
      {/* Stats Row */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard title="RMSE" value="2.34m" icon={Target} badge={{ text: 'Excellent', color: 'green' }} />
        <StatCard title="MAE" value="1.87m" icon={Activity} badge={{ text: 'Good', color: 'blue' }} />
        <StatCard title="R² Score" value="0.943" icon={TrendingUp} badge={{ text: 'High Accuracy', color: 'green' }} />
        <StatCard title="Shapiro-Wilk p" value="0.082" badge={{ text: 'Normal', color: 'blue' }} />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Predicted vs Actual */}
        <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6">
          <h3 className="mb-4 text-[#00E5FF]">Predicted vs Actual</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
              <XAxis
                dataKey="actual"
                name="Actual"
                stroke="#B0B6C2"
                style={{ fontSize: '10px' }}
                label={{ value: 'Actual Error (m)', position: 'insideBottom', offset: -5, fill: '#B0B6C2', fontSize: 12 }}
              />
              <YAxis
                dataKey="predicted"
                name="Predicted"
                stroke="#B0B6C2"
                style={{ fontSize: '10px' }}
                label={{ value: 'Predicted Error (m)', angle: -90, position: 'insideLeft', fill: '#B0B6C2', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#11141A',
                  border: '1px solid #00E5FF',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                cursor={{ strokeDasharray: '3 3' }}
              />
              <Scatter data={predictedVsActual} fill="#00E5FF" />
              <Line
                data={[
                  { actual: 0, predicted: 0 },
                  { actual: 15, predicted: 15 },
                ]}
                dataKey="predicted"
                stroke="#FF5252"
                strokeWidth={2}
                dot={false}
                strokeDasharray="5 5"
              />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-[#B0B6C2] mt-2 text-center">
            Dashed line represents perfect prediction
          </p>
        </div>

        {/* Residual Distribution */}
        <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6">
          <h3 className="mb-4 text-[#00E5FF]">Residual Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={residuals.map((r, i) => ({ ...r, index: i }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
              <XAxis
                dataKey="index"
                stroke="#B0B6C2"
                style={{ fontSize: '10px' }}
                label={{ value: 'Sample Index', position: 'insideBottom', offset: -5, fill: '#B0B6C2', fontSize: 12 }}
              />
              <YAxis
                stroke="#B0B6C2"
                style={{ fontSize: '10px' }}
                label={{ value: 'Residual (m)', angle: -90, position: 'insideLeft', fill: '#B0B6C2', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#11141A',
                  border: '1px solid #00E5FF',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {residuals.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#00E5FF' : '#FF5252'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-[#B0B6C2] mt-2 text-center">
            Residuals should be randomly distributed around zero
          </p>
        </div>

        {/* Feature Importance */}
        <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6 col-span-2">
          <h3 className="mb-4 text-[#00E5FF]">Top Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
              <XAxis
                type="number"
                stroke="#B0B6C2"
                style={{ fontSize: '10px' }}
                label={{ value: 'Importance Score', position: 'insideBottom', offset: -5, fill: '#B0B6C2', fontSize: 12 }}
              />
              <YAxis
                dataKey="feature"
                type="category"
                stroke="#B0B6C2"
                style={{ fontSize: '10px' }}
                width={120}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#11141A',
                  border: '1px solid #00E5FF',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {featureImportance.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={`rgba(0, 229, 255, ${0.3 + (entry.importance / 0.25) * 0.7})`}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Shapiro-Wilk Panel */}
        <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6">
          <h3 className="mb-4 text-[#00E5FF]">Normality Test (Shapiro-Wilk)</h3>
          <div className="space-y-4">
            <div className="bg-[#0E0E12] rounded-lg p-4">
              <div className="flex justify-between mb-2">
                <span className="text-sm text-[#B0B6C2]">Test Statistic</span>
                <span className="text-white">0.987</span>
              </div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-[#B0B6C2]">p-value</span>
                <span className="text-white">0.082</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-[#B0B6C2]">Significance Level</span>
                <span className="text-white">α = 0.05</span>
              </div>
            </div>

            <div className="bg-[#00C853]/10 border border-[#00C853]/30 rounded-lg p-4">
              <p className="text-[#00C853] mb-2">✓ Residuals are normally distributed</p>
              <p className="text-xs text-[#B0B6C2]">
                p-value (0.082) {'>'} α (0.05), so we fail to reject the null hypothesis of normality.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-[#B0B6C2]">Model Confidence</span>
                <span className="text-[#00E5FF]">94.3%</span>
              </div>
              <div className="h-2 bg-[#0E0E12] rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#00E5FF] to-[#42A5F5]"
                  style={{ width: '94.3%' }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Error Metrics Over Time */}
        <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6">
          <h3 className="mb-4 text-[#00E5FF]">Performance Metrics History</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart
              data={Array.from({ length: 12 }, (_, i) => ({
                month: `M${i + 1}`,
                rmse: 2.5 - Math.random() * 0.3,
                mae: 2.0 - Math.random() * 0.3,
              }))}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
              <XAxis dataKey="month" stroke="#B0B6C2" style={{ fontSize: '10px' }} />
              <YAxis stroke="#B0B6C2" style={{ fontSize: '10px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#11141A',
                  border: '1px solid #00E5FF',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Line type="monotone" dataKey="rmse" stroke="#00E5FF" strokeWidth={2} name="RMSE" />
              <Line type="monotone" dataKey="mae" stroke="#FF5252" strokeWidth={2} name="MAE" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-3 text-xs justify-center">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-[#00E5FF]"></div>
              <span className="text-[#B0B6C2]">RMSE</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-[#FF5252]"></div>
              <span className="text-[#B0B6C2]">MAE</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
