import { Satellite, Activity, CheckCircle, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

type PageType =
  | 'home'
  | 'data-overview'
  | 'preprocessing'
  | 'feature-engineering'
  | 'model-results'
  | 'realtime-predictions'
  | 'day8-predictions'
  | 'residual-analysis'
  | 'satellite-network';

interface HomePageProps {
  onNavigate: (page: PageType) => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <div className="space-y-8">
      {/* Overview Cards */}
      <div className="grid grid-cols-4 gap-6">
        <motion.div
          className="bg-white rounded-3xl p-8 shadow-sm border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-start justify-between mb-6">
            <div className="p-4 bg-blue-50 rounded-2xl">
              <Satellite className="w-7 h-7 text-blue-600" />
            </div>
          </div>
          <p className="text-4xl text-gray-900 mb-2">2</p>
          <p className="text-gray-600">Total Satellites</p>
        </motion.div>

        <motion.div
          className="bg-white rounded-3xl p-8 shadow-sm border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-start justify-between mb-6">
            <div className="p-4 bg-purple-50 rounded-2xl">
              <Activity className="w-7 h-7 text-purple-600" />
            </div>
            <span className="text-xs text-purple-600 bg-purple-50 px-3 py-1 rounded-full">
              GEO
            </span>
          </div>
          <p className="text-4xl text-gray-900 mb-2">1</p>
          <p className="text-gray-600">Geostationary</p>
        </motion.div>

        <motion.div
          className="bg-white rounded-3xl p-8 shadow-sm border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-start justify-between mb-6">
            <div className="p-4 bg-indigo-50 rounded-2xl">
              <Activity className="w-7 h-7 text-indigo-600" />
            </div>
            <span className="text-xs text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
              MEO
            </span>
          </div>
          <p className="text-4xl text-gray-900 mb-2">1</p>
          <p className="text-gray-600">Medium Orbit</p>
        </motion.div>

        <motion.div
          className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-3xl p-8 shadow-lg shadow-green-500/20 text-white"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-start justify-between mb-6">
            <div className="p-4 bg-white/20 backdrop-blur-sm rounded-2xl">
              <CheckCircle className="w-7 h-7 text-white" />
            </div>
          </div>
          <p className="text-4xl mb-2">Active</p>
          <p className="text-green-100">System Status</p>
        </motion.div>
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-3 gap-8">
        {/* Visualization */}
        <motion.div
          className="col-span-2 bg-white rounded-3xl p-8 shadow-sm border border-gray-200"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="mb-6">
            <h3 className="text-gray-900 mb-2">Satellite Network Overview</h3>
            <p className="text-gray-600">
              Real-time positioning of GEO and MEO satellites
            </p>
          </div>

          {/* Earth Visualization */}
          <div className="relative h-[450px] bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl flex items-center justify-center overflow-hidden">
            {/* Orbit rings */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="absolute w-64 h-64 border-2 border-blue-200 rounded-full" />
              <div className="absolute w-80 h-80 border-2 border-purple-200 rounded-full" />
            </div>

            {/* Earth */}
            <motion.div
              className="relative w-40 h-40 rounded-full shadow-2xl"
              animate={{ rotate: 360 }}
              transition={{ duration: 50, repeat: Infinity, ease: 'linear' }}
              style={{
                background:
                  'radial-gradient(circle at 35% 35%, #60A5FA 0%, #3B82F6 30%, #2563EB 60%, #1E40AF 100%)',
                boxShadow:
                  '0 20px 60px rgba(59, 130, 246, 0.3), inset -20px -20px 40px rgba(0, 0, 0, 0.3), inset 10px 10px 30px rgba(255, 255, 255, 0.2)',
              }}
            >
              {/* Continents */}
              <svg className="absolute inset-0 w-full h-full rounded-full" viewBox="0 0 128 128">
                <defs>
                  <radialGradient id="landHome">
                    <stop offset="0%" stopColor="#34D399" />
                    <stop offset="100%" stopColor="#10B981" />
                  </radialGradient>
                </defs>
                <ellipse cx="45" cy="40" rx="20" ry="14" fill="url(#landHome)" opacity="0.5" />
                <ellipse cx="75" cy="55" rx="18" ry="22" fill="url(#landHome)" opacity="0.5" />
                <ellipse cx="35" cy="80" rx="14" ry="18" fill="url(#landHome)" opacity="0.5" />
              </svg>

              {/* Atmosphere */}
              <div
                className="absolute inset-0 rounded-full"
                style={{
                  background:
                    'radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.5) 0%, transparent 70%)',
                }}
              />

              <div
                className="absolute -inset-3 rounded-full"
                style={{
                  background:
                    'radial-gradient(circle, rgba(96, 165, 250, 0.4) 0%, transparent 70%)',
                  filter: 'blur(10px)',
                }}
              />
            </motion.div>

            {/* Satellites */}
            {[
              { name: 'GEO', angle: 0, radius: 150, color: '#8B5CF6' },
              { name: 'MEO', angle: 180, radius: 120, color: '#3B82F6' },
            ].map((sat) => {
              const x = Math.cos((sat.angle * Math.PI) / 180) * sat.radius;
              const y = Math.sin((sat.angle * Math.PI) / 180) * sat.radius;

              return (
                <motion.div
                  key={sat.name}
                  className="absolute cursor-pointer"
                  style={{
                    left: '50%',
                    top: '50%',
                    marginLeft: `${x}px`,
                    marginTop: `${y}px`,
                  }}
                  whileHover={{ scale: 1.3 }}
                  onClick={() => onNavigate('satellite-network')}
                >
                  <div
                    className="w-4 h-4 rounded-full shadow-lg"
                    style={{
                      backgroundColor: sat.color,
                      boxShadow: `0 0 20px ${sat.color}`,
                    }}
                  />
                  <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap bg-white px-3 py-1 rounded-lg shadow-lg text-xs text-gray-700 border border-gray-200">
                    {sat.name}
                  </div>
                </motion.div>
              );
            })}

            {/* Legend */}
            <div className="absolute bottom-6 right-6 bg-white/90 backdrop-blur-sm rounded-2xl p-4 shadow-lg border border-gray-200">
              <p className="text-xs text-gray-500 mb-3">Click satellites to view details</p>
              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <span className="text-gray-700">GEO Satellite</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700">MEO Satellite</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Side Stats */}
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
        >
          {/* System Health */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-gray-200">
            <h3 className="text-gray-900 mb-6">System Health</h3>
            <div className="space-y-5">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600">Network Status</span>
                  <span className="text-green-600">100%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full w-full bg-gradient-to-r from-green-400 to-green-600"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600">Data Quality</span>
                  <span className="text-blue-600">98%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full w-[98%] bg-gradient-to-r from-blue-400 to-blue-600"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600">Model Accuracy</span>
                  <span className="text-purple-600">94%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full w-[94%] bg-gradient-to-r from-purple-400 to-purple-600"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl p-6 text-white shadow-lg shadow-blue-500/20">
            <h3 className="mb-3">Quick Access</h3>
            <p className="text-blue-100 text-sm mb-6">
              View detailed satellite information
            </p>
            <button
              onClick={() => onNavigate('satellite-network')}
              className="w-full bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-2xl py-3 text-sm transition-colors"
            >
              View Satellite Network
            </button>
          </div>

          {/* Recent Activity */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-gray-200">
            <h3 className="text-gray-900 mb-6">Recent Activity</h3>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                <div>
                  <p className="text-sm text-gray-800">Data updated successfully</p>
                  <p className="text-xs text-gray-500">2 minutes ago</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                <div>
                  <p className="text-sm text-gray-800">Model training completed</p>
                  <p className="text-xs text-gray-500">1 hour ago</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                <div>
                  <p className="text-sm text-gray-800">Predictions generated</p>
                  <p className="text-xs text-gray-500">3 hours ago</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
