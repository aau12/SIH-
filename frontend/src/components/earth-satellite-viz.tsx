import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface Satellite {
  id: string;
  name: string;
  type: 'GEO' | 'MEO' | 'IGSO';
  angle: number;
  radius: number;
  orbitSpeed: number;
}

interface EarthSatelliteVizProps {
  selectedSatellite: string | null;
  onSelectSatellite: (id: string | null) => void;
  showSatelliteDetails?: boolean;
}

const satellites: Satellite[] = [
  { id: 'sat-1', name: 'NAVSTAR-72', type: 'MEO', angle: 0, radius: 180, orbitSpeed: 0.3 },
  { id: 'sat-2', name: 'GALILEO-12', type: 'MEO', angle: 45, radius: 180, orbitSpeed: 0.28 },
  { id: 'sat-3', name: 'BEIDOU-G2', type: 'GEO', angle: 90, radius: 220, orbitSpeed: 0.15 },
  { id: 'sat-4', name: 'GLONASS-K1', type: 'MEO', angle: 135, radius: 180, orbitSpeed: 0.32 },
  { id: 'sat-5', name: 'NAVSTAR-45', type: 'MEO', angle: 180, radius: 190, orbitSpeed: 0.29 },
  { id: 'sat-6', name: 'GALILEO-24', type: 'MEO', angle: 225, radius: 185, orbitSpeed: 0.31 },
  { id: 'sat-7', name: 'BEIDOU-I3', type: 'IGSO', angle: 270, radius: 205, orbitSpeed: 0.2 },
  { id: 'sat-8', name: 'GLONASS-M7', type: 'MEO', angle: 315, radius: 180, orbitSpeed: 0.3 },
  { id: 'sat-9', name: 'NAVSTAR-68', type: 'MEO', angle: 30, radius: 175, orbitSpeed: 0.33 },
  { id: 'sat-10', name: 'BEIDOU-M4', type: 'MEO', angle: 60, radius: 188, orbitSpeed: 0.28 },
  { id: 'sat-11', name: 'GALILEO-18', type: 'MEO', angle: 120, radius: 182, orbitSpeed: 0.31 },
  { id: 'sat-12', name: 'BEIDOU-G3', type: 'GEO', angle: 200, radius: 220, orbitSpeed: 0.15 },
];

export function EarthSatelliteViz({
  selectedSatellite,
  onSelectSatellite,
  showSatelliteDetails = true,
}: EarthSatelliteVizProps) {
  const [satellitePositions, setSatellitePositions] = useState(satellites);
  const [stars, setStars] = useState<{ x: number; y: number; size: number; opacity: number }[]>([]);

  useEffect(() => {
    // Generate random stars
    const newStars = Array.from({ length: 100 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 2 + 0.5,
      opacity: Math.random() * 0.5 + 0.3,
    }));
    setStars(newStars);
  }, []);

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

  const handleSatelliteClick = (satId: string) => {
    if (selectedSatellite === satId) {
      onSelectSatellite(null);
    } else {
      onSelectSatellite(satId);
    }
  };

  const handleBackgroundClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onSelectSatellite(null);
    }
  };

  return (
    <div
      className="relative w-full h-full flex items-center justify-center"
      onClick={handleBackgroundClick}
    >
      {/* Stars Background */}
      <div className="absolute inset-0">
        {stars.map((star, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-white"
            style={{
              left: `${star.x}%`,
              top: `${star.y}%`,
              width: `${star.size}px`,
              height: `${star.size}px`,
              opacity: star.opacity,
            }}
          />
        ))}
      </div>

      {/* Orbit Rings */}
      <div className="absolute">
        {[160, 180, 200, 220].map((size) => (
          <div
            key={size}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#00E5FF]/10"
            style={{
              width: `${size * 2}px`,
              height: `${size * 2}px`,
            }}
          />
        ))}
      </div>

      {/* Earth */}
      <motion.div
        className="relative w-32 h-32 rounded-full shadow-2xl"
        animate={{ rotate: 360 }}
        transition={{ duration: 60, repeat: Infinity, ease: 'linear' }}
        style={{
          background: 'radial-gradient(circle at 30% 30%, #4A90E2 0%, #1E5A9E 30%, #0D3A6E 60%, #041F3D 100%)',
          boxShadow: '0 0 60px rgba(0, 229, 255, 0.3), inset -20px -20px 40px rgba(0, 0, 0, 0.5), inset 10px 10px 30px rgba(255, 255, 255, 0.1)',
        }}
      >
        {/* Continents/Land masses */}
        <svg className="absolute inset-0 w-full h-full rounded-full" viewBox="0 0 128 128">
          <defs>
            <radialGradient id="landGradient">
              <stop offset="0%" stopColor="#2D5A3D" />
              <stop offset="100%" stopColor="#1A3A2A" />
            </radialGradient>
          </defs>
          {/* Simplified continent shapes */}
          <ellipse cx="45" cy="40" rx="18" ry="12" fill="url(#landGradient)" opacity="0.7" />
          <ellipse cx="70" cy="50" rx="15" ry="20" fill="url(#landGradient)" opacity="0.7" />
          <ellipse cx="35" cy="75" rx="12" ry="15" fill="url(#landGradient)" opacity="0.7" />
          <path d="M 80 30 Q 90 35, 95 45 L 88 50 Q 82 42, 78 38 Z" fill="url(#landGradient)" opacity="0.7" />
          <circle cx="55" cy="95" r="8" fill="url(#landGradient)" opacity="0.7" />
        </svg>

        {/* Cloud layer */}
        <motion.div
          className="absolute inset-0 rounded-full opacity-20"
          animate={{ rotate: -360 }}
          transition={{ duration: 80, repeat: Infinity, ease: 'linear' }}
        >
          <svg className="w-full h-full" viewBox="0 0 128 128">
            <ellipse cx="30" cy="35" rx="15" ry="8" fill="white" opacity="0.3" />
            <ellipse cx="70" cy="25" rx="12" ry="6" fill="white" opacity="0.25" />
            <ellipse cx="50" cy="70" rx="18" ry="9" fill="white" opacity="0.3" />
            <ellipse cx="85" cy="80" rx="10" ry="5" fill="white" opacity="0.25" />
          </svg>
        </motion.div>

        {/* Atmosphere glow */}
        <div className="absolute inset-0 rounded-full" style={{
          background: 'radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.4) 0%, transparent 70%)',
        }}></div>
        
        {/* Outer atmosphere */}
        <div className="absolute -inset-2 rounded-full" style={{
          background: 'radial-gradient(circle, rgba(100, 180, 255, 0.3) 0%, transparent 70%)',
          filter: 'blur(8px)',
        }}></div>

        {/* Terminator line (day/night) */}
        <div className="absolute inset-0 rounded-full" style={{
          background: 'linear-gradient(135deg, transparent 40%, rgba(0, 0, 0, 0.4) 60%, rgba(0, 0, 0, 0.7) 100%)',
        }}></div>

        {/* Specular highlight */}
        <div className="absolute top-0 left-0 right-0 h-1/2 rounded-t-full" style={{
          background: 'radial-gradient(ellipse at 35% 25%, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0.2) 30%, transparent 60%)',
        }}></div>

        {/* Inner shadow for depth */}
        <div className="absolute inset-0 rounded-full shadow-inner" style={{
          boxShadow: 'inset -15px -15px 30px rgba(0, 0, 0, 0.6), inset 5px 5px 20px rgba(255, 255, 255, 0.1)',
        }}></div>
      </motion.div>

      {/* Satellites */}
      {satellitePositions.map((sat) => {
        const isSelected = selectedSatellite === sat.id;
        const isOtherSelected = selectedSatellite && selectedSatellite !== sat.id;
        const shouldShow = !selectedSatellite || isSelected;

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
            initial={{ opacity: 1, scale: 1 }}
            animate={{
              opacity: shouldShow ? 1 : 0,
              scale: isSelected ? 1.5 : 1,
              pointerEvents: shouldShow ? 'auto' : 'none',
            }}
            transition={{ duration: 0.3 }}
            onClick={(e) => {
              e.stopPropagation();
              handleSatelliteClick(sat.id);
            }}
          >
            {/* Glow Trail */}
            {isSelected && (
              <motion.div
                className="absolute w-20 h-20 -left-10 -top-10 rounded-full bg-[#00E5FF]/20 blur-xl"
                animate={{ scale: [1, 1.5, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}

            {/* Satellite */}
            <div className="relative group">
              <motion.div
                className={`w-3 h-3 rounded-sm bg-gradient-to-br ${
                  sat.type === 'GEO'
                    ? 'from-[#FF5252] to-[#FF1744]'
                    : sat.type === 'IGSO'
                    ? 'from-[#FFD700] to-[#FFA000]'
                    : 'from-[#00E5FF] to-[#42A5F5]'
                } shadow-lg`}
                whileHover={{ scale: 1.3 }}
                style={{
                  boxShadow: `0 0 20px ${
                    sat.type === 'GEO' ? '#FF5252' : sat.type === 'IGSO' ? '#FFD700' : '#00E5FF'
                  }`,
                }}
              />

              {/* Satellite Name on Hover */}
              {showSatelliteDetails && !selectedSatellite && (
                <div className="absolute left-1/2 -translate-x-1/2 -top-8 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap bg-[#11141A] px-2 py-1 rounded border border-[#00E5FF]/30 text-xs pointer-events-none">
                  {sat.name}
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
