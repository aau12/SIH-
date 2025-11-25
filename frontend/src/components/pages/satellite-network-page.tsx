import { MapPin, Activity, Radio, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const geoPerformanceData = Array.from({ length: 12 }, (_, i) => ({
  time: `${i * 2}h`,
  x: 2.1 + Math.random() * 0.5,
  y: 1.8 + Math.random() * 0.4,
  z: 2.5 + Math.random() * 0.6,
}));

const meoPerformanceData = Array.from({ length: 12 }, (_, i) => ({
  time: `${i * 2}h`,
  x: 3.2 + Math.random() * 0.8,
  y: 2.9 + Math.random() * 0.7,
  z: 3.8 + Math.random() * 0.9,
}));

export function SatelliteNetworkPage() {
  return (
    <div className="space-y-8">
      {/* Page Header */}
      <motion.div
        className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl p-8 text-white shadow-xl shadow-blue-500/20"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="mb-2">Satellite Network</h2>
        <p className="text-blue-100">
          Monitor and analyze performance of GEO and MEO satellites
        </p>
        <div className="grid grid-cols-3 gap-8 mt-8">
          <div>
            <p className="text-blue-100 text-sm mb-1">Total Active</p>
            <p className="text-3xl">2</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Network Health</p>
            <p className="text-3xl">100%</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Last Update</p>
            <p className="text-3xl">Live</p>
          </div>
        </div>
      </motion.div>

      {/* GEO Satellite Card */}
      <motion.div
        className="bg-white rounded-3xl shadow-lg border border-gray-200 overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {/* Card Header */}
        <div className="bg-gradient-to-r from-purple-500 to-indigo-600 p-8 text-white">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className="p-4 bg-white/20 backdrop-blur-sm rounded-2xl">
                <Radio className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-2xl mb-1">GEO Satellite</h3>
                <p className="text-purple-100">Geostationary Orbit</p>
              </div>
            </div>
            <span className="px-4 py-2 bg-green-500 text-white rounded-xl text-sm shadow-lg">
              Active
            </span>
          </div>
        </div>

        {/* Card Content */}
        <div className="p-8">
          <div className="grid grid-cols-2 gap-8 mb-8">
            {/* Left: Specs */}
            <div className="space-y-6">
              <h4 className="text-gray-900 mb-4">Satellite Specifications</h4>

              <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-3 bg-purple-100 rounded-xl">
                    <MapPin className="w-5 h-5 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Orbit Height</p>
                    <p className="text-xl text-gray-900">35,786 km</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 mb-4">
                  <div className="p-3 bg-indigo-100 rounded-xl">
                    <Activity className="w-5 h-5 text-indigo-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Coverage</p>
                    <p className="text-xl text-gray-900">Global</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="p-3 bg-blue-100 rounded-xl">
                    <TrendingUp className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Orbital Period</p>
                    <p className="text-xl text-gray-900">24 hours</p>
                  </div>
                </div>
              </div>

              {/* Position Data */}
              <div className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-2xl p-6 border border-purple-100">
                <h4 className="text-gray-900 mb-4">Current Position (m)</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-white rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-600 mb-1">X-Axis</p>
                    <p className="text-2xl text-gray-900">2.34</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-600 mb-1">Y-Axis</p>
                    <p className="text-2xl text-gray-900">1.92</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-600 mb-1">Z-Axis</p>
                    <p className="text-2xl text-gray-900">2.78</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Performance Chart */}
            <div>
              <h4 className="text-gray-900 mb-4">24-Hour Performance</h4>
              <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={geoPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="time" stroke="#6B7280" fontSize={12} />
                    <YAxis stroke="#6B7280" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid #E5E7EB',
                        borderRadius: '12px',
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="x"
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      dot={false}
                      name="X Error"
                    />
                    <Line
                      type="monotone"
                      dataKey="y"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={false}
                      name="Y Error"
                    />
                    <Line
                      type="monotone"
                      dataKey="z"
                      stroke="#10B981"
                      strokeWidth={2}
                      dot={false}
                      name="Z Error"
                    />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex gap-4 justify-center mt-4 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-purple-500 rounded"></div>
                    <span className="text-gray-600">X Error</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-blue-500 rounded"></div>
                    <span className="text-gray-600">Y Error</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-green-500 rounded"></div>
                    <span className="text-gray-600">Z Error</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* MEO Satellite Card */}
      <motion.div
        className="bg-white rounded-3xl shadow-lg border border-gray-200 overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* Card Header */}
        <div className="bg-gradient-to-r from-blue-500 to-cyan-600 p-8 text-white">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className="p-4 bg-white/20 backdrop-blur-sm rounded-2xl">
                <Radio className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-2xl mb-1">MEO Satellite</h3>
                <p className="text-blue-100">Medium Earth Orbit</p>
              </div>
            </div>
            <span className="px-4 py-2 bg-green-500 text-white rounded-xl text-sm shadow-lg">
              Active
            </span>
          </div>
        </div>

        {/* Card Content */}
        <div className="p-8">
          <div className="grid grid-cols-2 gap-8 mb-8">
            {/* Left: Specs */}
            <div className="space-y-6">
              <h4 className="text-gray-900 mb-4">Satellite Specifications</h4>

              <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-3 bg-blue-100 rounded-xl">
                    <MapPin className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Orbit Height</p>
                    <p className="text-xl text-gray-900">20,200 km</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 mb-4">
                  <div className="p-3 bg-cyan-100 rounded-xl">
                    <Activity className="w-5 h-5 text-cyan-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Coverage</p>
                    <p className="text-xl text-gray-900">Global</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="p-3 bg-indigo-100 rounded-xl">
                    <TrendingUp className="w-5 h-5 text-indigo-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Orbital Period</p>
                    <p className="text-xl text-gray-900">12 hours</p>
                  </div>
                </div>
              </div>

              {/* Position Data */}
              <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border border-blue-100">
                <h4 className="text-gray-900 mb-4">Current Position (m)</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-white rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-600 mb-1">X-Axis</p>
                    <p className="text-2xl text-gray-900">3.56</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-600 mb-1">Y-Axis</p>
                    <p className="text-2xl text-gray-900">3.12</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-600 mb-1">Z-Axis</p>
                    <p className="text-2xl text-gray-900">4.23</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Performance Chart */}
            <div>
              <h4 className="text-gray-900 mb-4">24-Hour Performance</h4>
              <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={meoPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="time" stroke="#6B7280" fontSize={12} />
                    <YAxis stroke="#6B7280" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid #E5E7EB',
                        borderRadius: '12px',
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="x"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={false}
                      name="X Error"
                    />
                    <Line
                      type="monotone"
                      dataKey="y"
                      stroke="#06B6D4"
                      strokeWidth={2}
                      dot={false}
                      name="Y Error"
                    />
                    <Line
                      type="monotone"
                      dataKey="z"
                      stroke="#10B981"
                      strokeWidth={2}
                      dot={false}
                      name="Z Error"
                    />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex gap-4 justify-center mt-4 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-blue-500 rounded"></div>
                    <span className="text-gray-600">X Error</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-cyan-500 rounded"></div>
                    <span className="text-gray-600">Y Error</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-green-500 rounded"></div>
                    <span className="text-gray-600">Z Error</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
