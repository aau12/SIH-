import { motion } from 'framer-motion';
import { X, Clock, Satellite } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface SatellitePopupProps {
  satelliteId: string;
  onClose: () => void;
}

const satelliteData: Record<string, any> = {
  'sat-1': { name: 'NAVSTAR-72', type: 'MEO', lastUpdate: '2 min ago' },
  'sat-2': { name: 'GALILEO-12', type: 'MEO', lastUpdate: '1 min ago' },
  'sat-3': { name: 'BEIDOU-G2', type: 'GEO', lastUpdate: '3 min ago' },
  'sat-4': { name: 'GLONASS-K1', type: 'MEO', lastUpdate: '2 min ago' },
  'sat-5': { name: 'NAVSTAR-45', type: 'MEO', lastUpdate: '4 min ago' },
  'sat-6': { name: 'GALILEO-24', type: 'MEO', lastUpdate: '1 min ago' },
  'sat-7': { name: 'BEIDOU-I3', type: 'IGSO', lastUpdate: '5 min ago' },
  'sat-8': { name: 'GLONASS-M7', type: 'MEO', lastUpdate: '2 min ago' },
  'sat-9': { name: 'NAVSTAR-68', type: 'MEO', lastUpdate: '3 min ago' },
  'sat-10': { name: 'BEIDOU-M4', type: 'MEO', lastUpdate: '1 min ago' },
  'sat-11': { name: 'GALILEO-18', type: 'MEO', lastUpdate: '2 min ago' },
  'sat-12': { name: 'BEIDOU-G3', type: 'GEO', lastUpdate: '4 min ago' },
};

const generate7DayData = () => {
  return Array.from({ length: 7 }, (_, i) => ({
    day: `Day ${i + 1}`,
    error: Math.random() * 10 + 2,
  }));
};

export function SatellitePopup({ satelliteId, onClose }: SatellitePopupProps) {
  const sat = satelliteData[satelliteId];
  const chartData = generate7DayData();

  if (!sat) return null;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm"></div>

      {/* Popup Card */}
      <motion.div
        className="relative bg-[#11141A] border border-[#00E5FF]/30 rounded-2xl p-6 max-w-md w-full shadow-2xl shadow-[#00E5FF]/10"
        initial={{ scale: 0.8, y: 50 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.8, y: 50 }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 hover:bg-[#1A1D26] rounded-lg transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#00E5FF] to-[#42A5F5] flex items-center justify-center">
              <Satellite className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-[#00E5FF]">{sat.name}</h3>
              <p className="text-xs text-[#B0B6C2]">Satellite ID: {satelliteId}</p>
            </div>
          </div>

          <div className="flex gap-2">
            <span
              className={`px-3 py-1 rounded-full text-xs ${
                sat.type === 'GEO'
                  ? 'bg-[#FF5252]/20 text-[#FF5252]'
                  : sat.type === 'IGSO'
                  ? 'bg-[#FFD700]/20 text-[#FFD700]'
                  : 'bg-[#00E5FF]/20 text-[#00E5FF]'
              }`}
            >
              {sat.type}
            </span>
            <span className="px-3 py-1 rounded-full text-xs bg-[#00C853]/20 text-[#00C853] flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {sat.lastUpdate}
            </span>
          </div>
        </div>

        {/* 7-Day Error Chart */}
        <div className="mb-4">
          <h4 className="mb-3 text-sm text-[#B0B6C2]">7-Day Error Trend</h4>
          <div className="bg-[#0E0E12] rounded-lg p-4">
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
                <XAxis
                  dataKey="day"
                  stroke="#B0B6C2"
                  style={{ fontSize: '10px' }}
                />
                <YAxis stroke="#B0B6C2" style={{ fontSize: '10px' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#11141A',
                    border: '1px solid #00E5FF',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="error"
                  stroke="#00E5FF"
                  strokeWidth={2}
                  dot={{ fill: '#00E5FF', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0E0E12] rounded-lg p-3">
            <p className="text-xs text-[#B0B6C2] mb-1">Avg Error</p>
            <p className="text-[#00E5FF]">5.2m</p>
          </div>
          <div className="bg-[#0E0E12] rounded-lg p-3">
            <p className="text-xs text-[#B0B6C2] mb-1">Max Error</p>
            <p className="text-[#FF5252]">11.8m</p>
          </div>
          <div className="bg-[#0E0E12] rounded-lg p-3">
            <p className="text-xs text-[#B0B6C2] mb-1">Clock Error</p>
            <p className="text-[#00E5FF]">0.23ns</p>
          </div>
          <div className="bg-[#0E0E12] rounded-lg p-3">
            <p className="text-xs text-[#B0B6C2] mb-1">Status</p>
            <p className="text-[#00C853]">Nominal</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
