import { Satellite, MapPin, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const performanceData = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i}:00`,
  xError: 3 + Math.random() * 4,
  yError: 2.5 + Math.random() * 3,
  zError: 4 + Math.random() * 4.5,
}));

const meoSatellites = [
  {
    name: 'MEO-SAT-01',
    orbitHeight: '20,200 km',
    coverage: 'Global',
    status: 'Active',
    lastUpdate: '30 sec ago',
  },
  {
    name: 'MEO-SAT-02',
    orbitHeight: '20,200 km',
    coverage: 'Global',
    status: 'Active',
    lastUpdate: '45 sec ago',
  },
  {
    name: 'MEO-SAT-03',
    orbitHeight: '20,200 km',
    coverage: 'Global',
    status: 'Active',
    lastUpdate: '1 min ago',
  },
  {
    name: 'MEO-SAT-04',
    orbitHeight: '20,200 km',
    coverage: 'Global',
    status: 'Active',
    lastUpdate: '1 min ago',
  },
  {
    name: 'MEO-SAT-05',
    orbitHeight: '20,200 km',
    coverage: 'Global',
    status: 'Active',
    lastUpdate: '2 min ago',
  },
  {
    name: 'MEO-SAT-06',
    orbitHeight: '20,200 km',
    coverage: 'Global',
    status: 'Warning',
    lastUpdate: '5 min ago',
  },
];

export function MeoSatellitePage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl p-6 text-white shadow-lg">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-4 bg-white/20 backdrop-blur-sm rounded-xl">
            <Satellite className="w-8 h-8" />
          </div>
          <div>
            <h2>MEO Satellites</h2>
            <p className="text-blue-100 text-sm">Medium Earth Orbit - 20,200 km altitude</p>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-6">
          <div>
            <p className="text-blue-100 text-sm mb-1">Total</p>
            <p className="text-2xl">16</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Active</p>
            <p className="text-2xl">15</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Avg Error</p>
            <p className="text-2xl">4.8m</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Health</p>
            <p className="text-2xl">94%</p>
          </div>
        </div>
      </div>

      {/* Satellite Cards */}
      <div className="grid grid-cols-3 gap-6">
        {meoSatellites.map((sat) => (
          <div key={sat.name} className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-blue-100 rounded-xl">
                  <Satellite className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="text-gray-800 text-sm">{sat.name}</h3>
                  <p className="text-xs text-gray-500">{sat.lastUpdate}</p>
                </div>
              </div>
              <span
                className={`px-2 py-1 rounded-full text-xs ${
                  sat.status === 'Active'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-yellow-100 text-yellow-700'
                }`}
              >
                {sat.status}
              </span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <MapPin className="w-3 h-3 text-gray-400" />
                <span className="text-gray-600">Orbit:</span>
                <span className="text-gray-800">{sat.orbitHeight}</span>
              </div>
              <div className="flex items-center gap-2">
                <Activity className="w-3 h-3 text-gray-400" />
                <span className="text-gray-600">Coverage:</span>
                <span className="text-gray-800">{sat.coverage}</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-xs text-gray-500 mb-1">X</p>
                  <p className="text-xs text-gray-800">{(3 + Math.random() * 3).toFixed(2)}m</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-1">Y</p>
                  <p className="text-xs text-gray-800">{(2.5 + Math.random() * 2.5).toFixed(2)}m</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-1">Z</p>
                  <p className="text-xs text-gray-800">{(4 + Math.random() * 3).toFixed(2)}m</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Performance Chart */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
        <h3 className="mb-4 text-gray-800">24-Hour Error History</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis dataKey="hour" stroke="#6B7280" />
            <YAxis stroke="#6B7280" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #E5E7EB',
                borderRadius: '12px',
              }}
            />
            <Line type="monotone" dataKey="xError" stroke="#3B82F6" strokeWidth={2} />
            <Line type="monotone" dataKey="yError" stroke="#06B6D4" strokeWidth={2} />
            <Line type="monotone" dataKey="zError" stroke="#10B981" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Parameters Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-gray-800">Satellite Parameters</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Parameter</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Value</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Unit</th>
                <th className="px-6 py-3 text-left text-xs text-gray-500 uppercase">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              <tr className="hover:bg-gray-50">
                <td className="px-6 py-4 text-sm text-gray-800">Orbital Period</td>
                <td className="px-6 py-4 text-sm text-gray-600">12.03</td>
                <td className="px-6 py-4 text-sm text-gray-600">hours</td>
                <td className="px-6 py-4">
                  <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">
                    Nominal
                  </span>
                </td>
              </tr>
              <tr className="hover:bg-gray-50">
                <td className="px-6 py-4 text-sm text-gray-800">Inclination</td>
                <td className="px-6 py-4 text-sm text-gray-600">55.0</td>
                <td className="px-6 py-4 text-sm text-gray-600">degrees</td>
                <td className="px-6 py-4">
                  <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">
                    Nominal
                  </span>
                </td>
              </tr>
              <tr className="hover:bg-gray-50">
                <td className="px-6 py-4 text-sm text-gray-800">Eccentricity</td>
                <td className="px-6 py-4 text-sm text-gray-600">0.0002</td>
                <td className="px-6 py-4 text-sm text-gray-600">-</td>
                <td className="px-6 py-4">
                  <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">
                    Nominal
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
