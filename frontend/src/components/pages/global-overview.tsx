import { Activity, Satellite, Clock, AlertTriangle } from 'lucide-react';
import { StatCard } from '../stat-card';
import { PredictionCard } from '../prediction-card';
import { EarthSatelliteViz } from '../earth-satellite-viz';
import { SatellitePopup } from '../satellite-popup';

interface GlobalOverviewProps {
  selectedSatellite: string | null;
  onSelectSatellite: (id: string | null) => void;
}

const predictionHorizons = [
  { horizon: '0.15h', predictions: { clock: 0.12, x: 2.3, y: 1.8, z: 3.1 } },
  { horizon: '0.30h', predictions: { clock: 0.24, x: 3.1, y: 2.4, z: 4.2 } },
  { horizon: '1h', predictions: { clock: 0.45, x: 4.8, y: 3.9, z: 5.7 } },
  { horizon: '2h', predictions: { clock: 0.78, x: 6.2, y: 5.1, z: 7.3 } },
  { horizon: '3h', predictions: { clock: 1.12, x: 7.9, y: 6.8, z: 8.9 } },
  { horizon: '6h', predictions: { clock: 1.89, x: 9.4, y: 8.2, z: 10.5 } },
  { horizon: '12h', predictions: { clock: 2.56, x: 11.2, y: 9.8, z: 12.1 } },
  { horizon: '24h', predictions: { clock: 3.42, x: 13.8, y: 11.5, z: 14.7 } },
];

export function GlobalOverview({ selectedSatellite, onSelectSatellite }: GlobalOverviewProps) {
  return (
    <div className="p-6">
      <div className="grid grid-cols-12 gap-6">
        {/* Left Column - Stats */}
        <div className="col-span-3 space-y-4">
          <StatCard
            title="Model Status"
            value="Online"
            icon={Activity}
            badge={{ text: 'Predicting', color: 'green' }}
            delay={0}
          />
          <StatCard
            title="Satellites with High Error"
            value="3"
            icon={AlertTriangle}
            badge={{ text: 'Action Required', color: 'red' }}
            delay={0.1}
          />
          <StatCard
            title="Last Data Update"
            value="2 min ago"
            icon={Clock}
            badge={{ text: 'Real-time', color: 'blue' }}
            delay={0.2}
          />
          <StatCard
            title="Active Satellites"
            value="12"
            icon={Satellite}
            badge={{ text: 'All Systems', color: 'green' }}
            delay={0.3}
          />

          {/* Quick Stats */}
          <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4">
            <h3 className="mb-4 text-sm text-[#B0B6C2]">System Health</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1 text-xs">
                  <span className="text-[#B0B6C2]">MEO Satellites</span>
                  <span className="text-[#00E5FF]">8/8</span>
                </div>
                <div className="h-2 bg-[#0E0E12] rounded-full overflow-hidden">
                  <div className="h-full w-full bg-gradient-to-r from-[#00E5FF] to-[#42A5F5]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1 text-xs">
                  <span className="text-[#B0B6C2]">GEO Satellites</span>
                  <span className="text-[#00E5FF]">2/2</span>
                </div>
                <div className="h-2 bg-[#0E0E12] rounded-full overflow-hidden">
                  <div className="h-full w-full bg-gradient-to-r from-[#FF5252] to-[#FF1744]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1 text-xs">
                  <span className="text-[#B0B6C2]">IGSO Satellites</span>
                  <span className="text-[#00E5FF]">2/2</span>
                </div>
                <div className="h-2 bg-[#0E0E12] rounded-full overflow-hidden">
                  <div className="h-full w-full bg-gradient-to-r from-[#FFD700] to-[#FFA000]"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Center Column - 3D Earth */}
        <div className="col-span-6">
          <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-6 h-[600px] relative overflow-hidden">
            <div className="absolute inset-0">
              <EarthSatelliteViz
                selectedSatellite={selectedSatellite}
                onSelectSatellite={onSelectSatellite}
              />
            </div>
          </div>

          {selectedSatellite && (
            <div className="mt-4 bg-[#11141A] border border-[#00E5FF]/30 rounded-xl p-4">
              <p className="text-sm text-[#B0B6C2] mb-2">Selected Satellite</p>
              <p className="text-[#00E5FF]">
                {selectedSatellite === 'sat-1' && 'NAVSTAR-72'}
                {selectedSatellite === 'sat-2' && 'GALILEO-12'}
                {selectedSatellite === 'sat-3' && 'BEIDOU-G2'}
                {selectedSatellite === 'sat-4' && 'GLONASS-K1'}
                {selectedSatellite === 'sat-5' && 'NAVSTAR-45'}
                {selectedSatellite === 'sat-6' && 'GALILEO-24'}
                {selectedSatellite === 'sat-7' && 'BEIDOU-I3'}
                {selectedSatellite === 'sat-8' && 'GLONASS-M7'}
                {selectedSatellite === 'sat-9' && 'NAVSTAR-68'}
                {selectedSatellite === 'sat-10' && 'BEIDOU-M4'}
                {selectedSatellite === 'sat-11' && 'GALILEO-18'}
                {selectedSatellite === 'sat-12' && 'BEIDOU-G3'}
              </p>
              <p className="text-xs text-[#B0B6C2] mt-1">
                Click outside or select another satellite to deselect
              </p>
            </div>
          )}
        </div>

        {/* Right Column - 8th Day Predictions */}
        <div className="col-span-3">
          <div className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4 h-[600px] flex flex-col">
            <h3 className="mb-4 text-[#00E5FF]">8th Day Error Prediction</h3>
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
          </div>
        </div>
      </div>

      {/* Satellite Popup */}
      {selectedSatellite && (
        <SatellitePopup satelliteId={selectedSatellite} onClose={() => onSelectSatellite(null)} />
      )}
    </div>
  );
}
