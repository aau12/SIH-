import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface InteractivePlanetProps {
  onNavigate: (page: string) => void;
}

interface Satellite {
  id: string;
  name: string;
  type: 'GEO' | 'MEO';
  angle: number;
  radius: number;
  orbitSpeed: number;
}

const satellites: Satellite[] = [
  { id: 'sat-1', name: 'GEO-SAT-01', type: 'GEO', angle: 0, radius: 180, orbitSpeed: 0.15 },
  { id: 'sat-2', name: 'GEO-SAT-02', type: 'GEO', angle: 90, radius: 180, orbitSpeed: 0.15 },
  { id: 'sat-3', name: 'GEO-SAT-03', type: 'GEO', angle: 180, radius: 180, orbitSpeed: 0.15 },
  { id: 'sat-4', name: 'GEO-SAT-04', type: 'GEO', angle: 270, radius: 180, orbitSpeed: 0.15 },
  { id: 'sat-5', name: 'MEO-SAT-01', type: 'MEO', angle: 30, radius: 130, orbitSpeed: 0.4 },
  { id: 'sat-6', name: 'MEO-SAT-02', type: 'MEO', angle: 90, radius: 130, orbitSpeed: 0.4 },
  { id: 'sat-7', name: 'MEO-SAT-03', type: 'MEO', angle: 150, radius: 130, orbitSpeed: 0.4 },
  { id: 'sat-8', name: 'MEO-SAT-04', type: 'MEO', angle: 210, radius: 130, orbitSpeed: 0.4 },
  { id: 'sat-9', name: 'MEO-SAT-05', type: 'MEO', angle: 270, radius: 130, orbitSpeed: 0.4 },
  { id: 'sat-10', name: 'MEO-SAT-06', type: 'MEO', angle: 330, radius: 130, orbitSpeed: 0.4 },
];

export function InteractivePlanet({ onNavigate }: InteractivePlanetProps) {
  const [satellitePositions, setSatellitePositions] = useState(satellites);
  const [hoveredSat, setHoveredSat] = useState<string | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setSatellitePositions((prev) =>
        prev.map((sat) => ({
          ...sat,
          angle: (sat.angle + sat.orbitSpeed) % 360,
        }))
      );
    }, 50);

    return () => clearInterval(interval);
  }, []);

  const handleSatelliteClick = (type: 'GEO' | 'MEO') => {
    if (type === 'GEO') {
      onNavigate('geo-satellite');
    } else {
      onNavigate('meo-satellite');
    }
  };

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl">
      {/* Orbit Rings */}
      <div className="absolute">
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-blue-200"
          style={{ width: '260px', height: '260px' }}
        />
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-purple-200"
          style={{ width: '360px', height: '360px' }}
        />
      </div>

      {/* Earth */}
      <motion.div
        className="relative w-32 h-32 rounded-full shadow-2xl cursor-pointer"
        animate={{ rotate: 360 }}
        transition={{ duration: 60, repeat: Infinity, ease: 'linear' }}
        style={{
          background:
            'radial-gradient(circle at 30% 30%, #4A90E2 0%, #2563EB 30%, #1E40AF 60%, #1E3A8A 100%)',
          boxShadow:
            '0 10px 40px rgba(37, 99, 235, 0.3), inset -15px -15px 30px rgba(0, 0, 0, 0.3), inset 5px 5px 20px rgba(255, 255, 255, 0.2)',
        }}
      >
        {/* Continents */}
        <svg className="absolute inset-0 w-full h-full rounded-full" viewBox="0 0 128 128">
          <defs>
            <radialGradient id="landGradient">
              <stop offset="0%" stopColor="#34D399" />
              <stop offset="100%" stopColor="#10B981" />
            </radialGradient>
          </defs>
          <ellipse cx="45" cy="40" rx="18" ry="12" fill="url(#landGradient)" opacity="0.6" />
          <ellipse cx="70" cy="50" rx="15" ry="20" fill="url(#landGradient)" opacity="0.6" />
          <ellipse cx="35" cy="75" rx="12" ry="15" fill="url(#landGradient)" opacity="0.6" />
        </svg>

        {/* Atmosphere */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: 'radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.4) 0%, transparent 70%)',
          }}
        />

        {/* Outer glow */}
        <div
          className="absolute -inset-2 rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(96, 165, 250, 0.4) 0%, transparent 70%)',
            filter: 'blur(8px)',
          }}
        />
      </motion.div>

      {/* Satellites */}
      {satellitePositions.map((sat) => {
        const x = Math.cos((sat.angle * Math.PI) / 180) * sat.radius;
        const y = Math.sin((sat.angle * Math.PI) / 180) * sat.radius;

        return (
          <motion.div
            key={sat.id}
            className="absolute cursor-pointer"
            style={{
              left: '50%',
              top: '50%',
              marginLeft: `${x}px`,
              marginTop: `${y}px`,
            }}
            onMouseEnter={() => setHoveredSat(sat.id)}
            onMouseLeave={() => setHoveredSat(null)}
            onClick={() => handleSatelliteClick(sat.type)}
            whileHover={{ scale: 1.3 }}
          >
            <div
              className={`w-3 h-3 rounded-full ${
                sat.type === 'GEO' ? 'bg-purple-500' : 'bg-blue-500'
              } shadow-lg`}
              style={{
                boxShadow: `0 0 20px ${sat.type === 'GEO' ? '#A855F7' : '#3B82F6'}`,
              }}
            />

            {/* Hover tooltip */}
            {hoveredSat === sat.id && (
              <motion.div
                className="absolute left-1/2 -translate-x-1/2 -top-10 whitespace-nowrap bg-white px-3 py-2 rounded-lg shadow-lg border border-gray-200 text-xs"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <p className="text-gray-800">{sat.name}</p>
                <p className="text-gray-500">{sat.type}</p>
              </motion.div>
            )}
          </motion.div>
        );
      })}

      {/* Legend */}
      <div className="absolute bottom-6 right-6 bg-white/90 backdrop-blur-sm rounded-xl p-4 shadow-lg border border-gray-200">
        <p className="text-xs text-gray-500 mb-2">Click satellites to view details</p>
        <div className="flex gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span className="text-gray-600">GEO</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-gray-600">MEO</span>
          </div>
        </div>
      </div>
    </div>
  );
}
