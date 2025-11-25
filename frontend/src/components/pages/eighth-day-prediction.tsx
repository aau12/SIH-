import { motion } from 'framer-motion';
import { EarthSatelliteViz } from '../earth-satellite-viz';
import { PredictionCard } from '../prediction-card';
import { TrendingUp, Zap } from 'lucide-react';

interface EighthDayPredictionProps {
  selectedSatellite: string | null;
  onSelectSatellite: (id: string | null) => void;
}

const predictionHorizons = [
  { horizon: '0.15h (9 min)', predictions: { clock: 0.12, x: 2.3, y: 1.8, z: 3.1 } },
  { horizon: '0.30h (18 min)', predictions: { clock: 0.24, x: 3.1, y: 2.4, z: 4.2 } },
  { horizon: '1h', predictions: { clock: 0.45, x: 4.8, y: 3.9, z: 5.7 } },
  { horizon: '2h', predictions: { clock: 0.78, x: 6.2, y: 5.1, z: 7.3 } },
  { horizon: '3h', predictions: { clock: 1.12, x: 7.9, y: 6.8, z: 8.9 } },
  { horizon: '6h', predictions: { clock: 1.89, x: 9.4, y: 8.2, z: 10.5 } },
  { horizon: '12h', predictions: { clock: 2.56, x: 11.2, y: 9.8, z: 12.1 } },
  { horizon: '24h (1 day)', predictions: { clock: 3.42, x: 13.8, y: 11.5, z: 14.7 } },
];

export function EighthDayPrediction({ selectedSatellite, onSelectSatellite }: EighthDayPredictionProps) {
  return (
    <div className="p-6">
      <div className="grid grid-cols-12 gap-6">
        {/* Left Side - 3D Earth with Future Orbit */}
        <div className="col-span-8">
          <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-xl text-[#00E5FF] mb-1">8th Day Prediction Visualization</h2>
                <p className="text-sm text-[#B0B6C2]">
                  Predicted orbital drift and error accumulation
                </p>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-[#00E5FF]/10 rounded-lg">
                <Zap className="w-4 h-4 text-[#00E5FF]" />
                <span className="text-sm text-[#00E5FF]">Real-time Prediction</span>
              </div>
            </div>

            {/* 3D Earth Visualization */}
            <div className="h-[500px] relative overflow-hidden rounded-lg bg-[#0E0E12]">
              <EarthSatelliteViz
                selectedSatellite={selectedSatellite}
                onSelectSatellite={onSelectSatellite}
              />

              {/* Future Orbit Path Overlay */}
              {selectedSatellite && (
                <motion.div
                  className="absolute inset-0 pointer-events-none"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <svg className="w-full h-full" viewBox="0 0 600 600">
                    <defs>
                      <linearGradient id="drift-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" style={{ stopColor: '#00E5FF', stopOpacity: 0 }} />
                        <stop offset="50%" style={{ stopColor: '#00E5FF', stopOpacity: 0.5 }} />
                        <stop offset="100%" style={{ stopColor: '#FF5252', stopOpacity: 0.8 }} />
                      </linearGradient>
                    </defs>
                    <motion.path
                      d="M 300 150 Q 400 200, 450 300 T 450 450"
                      stroke="url(#drift-gradient)"
                      strokeWidth="3"
                      fill="none"
                      strokeDasharray="10 5"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 2, ease: 'easeInOut' }}
                    />
                    <motion.circle
                      cx="450"
                      cy="450"
                      r="8"
                      fill="#FF5252"
                      initial={{ scale: 0 }}
                      animate={{ scale: [0, 1.5, 1] }}
                      transition={{ delay: 2, duration: 0.5 }}
                    >
                      <animate
                        attributeName="opacity"
                        values="1;0.3;1"
                        dur="2s"
                        repeatCount="indefinite"
                      />
                    </motion.circle>
                  </svg>
                </motion.div>
              )}
            </div>
          </div>

          {/* Prediction Accuracy Indicators */}
          <div className="grid grid-cols-3 gap-4">
            <motion.div
              className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-[#00C853]" />
                <span className="text-xs text-[#B0B6C2]">Short-term (0-3h)</span>
              </div>
              <p className="text-xl text-[#00C853] mb-1">96.2%</p>
              <p className="text-xs text-[#B0B6C2]">Accuracy</p>
            </motion.div>

            <motion.div
              className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-[#FFD700]" />
                <span className="text-xs text-[#B0B6C2]">Medium-term (3-12h)</span>
              </div>
              <p className="text-xl text-[#FFD700] mb-1">91.8%</p>
              <p className="text-xs text-[#B0B6C2]">Accuracy</p>
            </motion.div>

            <motion.div
              className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-[#FF5252]" />
                <span className="text-xs text-[#B0B6C2]">Long-term (12-24h)</span>
              </div>
              <p className="text-xl text-[#FF5252] mb-1">84.5%</p>
              <p className="text-xs text-[#B0B6C2]">Accuracy</p>
            </motion.div>
          </div>
        </div>

        {/* Right Side - Prediction Cards */}
        <div className="col-span-4">
          <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4 h-full flex flex-col">
            <div className="mb-4">
              <h3 className="text-[#00E5FF] mb-1">Time Horizon Predictions</h3>
              <p className="text-xs text-[#B0B6C2]">
                Error predictions at different future time points
              </p>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
              {predictionHorizons.map((item, index) => (
                <PredictionCard
                  key={item.horizon}
                  horizon={item.horizon}
                  predictions={item.predictions}
                  delay={index * 0.05}
                />
              ))}
            </div>

            {/* Legend */}
            <div className="mt-4 pt-4 border-t border-[#1A1D26]">
              <p className="text-xs text-[#B0B6C2] mb-2">Severity Levels:</p>
              <div className="space-y-1 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#00C853]"></div>
                  <span className="text-[#B0B6C2]">Low (≤ 5)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#FFD700]"></div>
                  <span className="text-[#B0B6C2]">Medium (5-8)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#FF5252]"></div>
                  <span className="text-[#B0B6C2]">High ({'>'}8)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Info Banner */}
      <motion.div
        className="mt-6 bg-gradient-to-r from-[#00E5FF]/10 to-[#E040FB]/10 border border-[#00E5FF]/30 rounded-xl p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-[#00E5FF] mb-2">About 8th Day Predictions</h3>
            <p className="text-sm text-[#B0B6C2] max-w-2xl">
              These predictions use all 134 ML features including lag features, rolling windows, and trend
              analysis to forecast satellite position and clock errors up to 24 hours in advance. The model
              continuously learns from real-time data to improve accuracy.
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-[#B0B6C2] mb-1">Model Version</p>
            <p className="text-[#00E5FF]">v3.2.1</p>
            <p className="text-xs text-[#B0B6C2] mt-2">Last Updated</p>
            <p className="text-white">2 hours ago</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
