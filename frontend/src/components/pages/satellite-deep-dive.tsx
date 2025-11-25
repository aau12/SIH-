import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { EarthSatelliteViz } from '../earth-satellite-viz';
import { Clock, TrendingUp, TrendingDown } from 'lucide-react';

interface SatelliteDeepDiveProps {
  selectedSatellite: string | null;
  onSelectSatellite: (id: string | null) => void;
}

const satelliteInfo: Record<string, any> = {
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

const generateErrorHistory = () => {
  return Array.from({ length: 24 }, (_, i) => ({
    time: `${i}:00`,
    x: Math.random() * 8 + 2,
    y: Math.random() * 8 + 2,
    z: Math.random() * 8 + 2,
    clock: Math.random() * 2 + 0.5,
  }));
};

const generateRollingWindow = () => {
  return Array.from({ length: 20 }, (_, i) => ({
    time: i,
    mean: Math.random() * 6 + 3,
    std: Math.random() * 2 + 0.5,
    min: Math.random() * 3,
    max: Math.random() * 4 + 8,
  }));
};

export function SatelliteDeepDive({ selectedSatellite, onSelectSatellite }: SatelliteDeepDiveProps) {
  const errorHistory = generateErrorHistory();
  const rollingWindow = generateRollingWindow();
  const sat = selectedSatellite ? satelliteInfo[selectedSatellite] : null;

  return (
    <div className="p-6">
      {!selectedSatellite ? (
        <div className="flex items-center justify-center h-[600px] bg-[#11141A] border border-[#1A1D26] rounded-xl">
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-[#00E5FF] to-[#42A5F5] rounded-full flex items-center justify-center mx-auto mb-4 opacity-50">
              <Clock className="w-8 h-8" />
            </div>
            <p className="text-xl text-[#B0B6C2] mb-2">No Satellite Selected</p>
            <p className="text-sm text-[#B0B6C2]/60">
              Please select a satellite from the Global Overview page
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-12 gap-6">
          {/* Left Side - Satellite Info */}
          <div className="col-span-3 space-y-4">
            <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
              <h3 className="mb-4 text-[#00E5FF]">Satellite Info</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-[#B0B6C2] mb-1">Name</p>
                  <p className="text-white">{sat?.name}</p>
                </div>
                <div>
                  <p className="text-xs text-[#B0B6C2] mb-1">Orbit Type</p>
                  <span
                    className={`px-3 py-1 rounded-full text-xs ${
                      sat?.type === 'GEO'
                        ? 'bg-[#FF5252]/20 text-[#FF5252]'
                        : sat?.type === 'IGSO'
                        ? 'bg-[#FFD700]/20 text-[#FFD700]'
                        : 'bg-[#00E5FF]/20 text-[#00E5FF]'
                    }`}
                  >
                    {sat?.type}
                  </span>
                </div>
                <div>
                  <p className="text-xs text-[#B0B6C2] mb-1">Last Update</p>
                  <p className="text-white">{sat?.lastUpdate}</p>
                </div>
              </div>
            </div>

            <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
              <h3 className="mb-4 text-sm text-[#B0B6C2]">Current Errors</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-xs text-[#B0B6C2]">X Error:</span>
                  <span className="text-[#00E5FF]">4.2m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs text-[#B0B6C2]">Y Error:</span>
                  <span className="text-[#00E5FF]">3.8m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs text-[#B0B6C2]">Z Error:</span>
                  <span className="text-[#00E5FF]">5.1m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs text-[#B0B6C2]">Clock Error:</span>
                  <span className="text-[#00E5FF]">0.42ns</span>
                </div>
              </div>
            </div>

            <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
              <h3 className="mb-4 text-sm text-[#B0B6C2]">Trend Analysis</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#B0B6C2]">Velocity (diff1)</span>
                  <div className="flex items-center gap-1 text-[#00C853]">
                    <TrendingUp className="w-3 h-3" />
                    <span className="text-xs">+0.12</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#B0B6C2]">Acceleration (diff2)</span>
                  <div className="flex items-center gap-1 text-[#FF5252]">
                    <TrendingDown className="w-3 h-3" />
                    <span className="text-xs">-0.05</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Middle - 3D Earth Zoomed */}
          <div className="col-span-5">
            <div className="bg-[#11141A] border border-[#00E5FF]/30 rounded-xl p-6 h-[400px] mb-6 relative overflow-hidden">
              <div className="absolute inset-0">
                <EarthSatelliteViz
                  selectedSatellite={selectedSatellite}
                  onSelectSatellite={onSelectSatellite}
                  showSatelliteDetails={false}
                />
              </div>
            </div>

            {/* Error History Graph */}
            <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
              <h3 className="mb-4 text-sm text-[#B0B6C2]">24-Hour Error History</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={errorHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
                  <XAxis dataKey="time" stroke="#B0B6C2" style={{ fontSize: '10px' }} />
                  <YAxis stroke="#B0B6C2" style={{ fontSize: '10px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#11141A',
                      border: '1px solid #00E5FF',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Line type="monotone" dataKey="x" stroke="#00E5FF" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="y" stroke="#42A5F5" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="z" stroke="#FF5252" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="clock" stroke="#FFD700" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-3 text-xs justify-center">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-[#00E5FF]"></div>
                  <span className="text-[#B0B6C2]">X Error</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-[#42A5F5]"></div>
                  <span className="text-[#B0B6C2]">Y Error</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-[#FF5252]"></div>
                  <span className="text-[#B0B6C2]">Z Error</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-[#FFD700]"></div>
                  <span className="text-[#B0B6C2]">Clock</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side - Rolling Window & Trend */}
          <div className="col-span-4 space-y-4">
            <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
              <h3 className="mb-4 text-sm text-[#B0B6C2]">Rolling Window Comparison</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={rollingWindow}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1A1D26" />
                  <XAxis dataKey="time" stroke="#B0B6C2" style={{ fontSize: '10px' }} />
                  <YAxis stroke="#B0B6C2" style={{ fontSize: '10px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#11141A',
                      border: '1px solid #00E5FF',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="mean"
                    stroke="#00E5FF"
                    fill="#00E5FF"
                    fillOpacity={0.2}
                  />
                  <Area
                    type="monotone"
                    dataKey="max"
                    stroke="#FF5252"
                    fill="#FF5252"
                    fillOpacity={0.1}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
              <h3 className="mb-4 text-sm text-[#B0B6C2]">Rolling Window Stats</h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-[#0E0E12] rounded-lg p-3">
                  <p className="text-xs text-[#B0B6C2] mb-1">Mean (3h)</p>
                  <p className="text-[#00E5FF]">4.8m</p>
                </div>
                <div className="bg-[#0E0E12] rounded-lg p-3">
                  <p className="text-xs text-[#B0B6C2] mb-1">Std (3h)</p>
                  <p className="text-[#00E5FF]">1.2m</p>
                </div>
                <div className="bg-[#0E0E12] rounded-lg p-3">
                  <p className="text-xs text-[#B0B6C2] mb-1">Min (12h)</p>
                  <p className="text-[#00C853]">1.8m</p>
                </div>
                <div className="bg-[#0E0E12] rounded-lg p-3">
                  <p className="text-xs text-[#B0B6C2] mb-1">Max (12h)</p>
                  <p className="text-[#FF5252]">9.4m</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
