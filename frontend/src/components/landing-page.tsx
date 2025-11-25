import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

interface LandingPageProps {
  onGetStarted: () => void;
}

export function LandingPage({ onGetStarted }: LandingPageProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center p-8 overflow-hidden relative">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-300/30 rounded-full"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
            animate={{
              y: [null, Math.random() * window.innerHeight],
              opacity: [0.3, 0.8, 0.3],
            }}
            transition={{
              duration: Math.random() * 10 + 5,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        ))}
      </div>

      <div className="relative z-10 max-w-6xl mx-auto grid md:grid-cols-2 gap-12 items-center">
        {/* Left Content */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.div
            className="inline-block px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm mb-6"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            Advanced Monitoring System
          </motion.div>

          <h1 className="mb-6 text-gray-900 leading-tight">
            Satellite Monitoring System
          </h1>

          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Real-time tracking and analysis for GEO and MEO satellites. 
            Monitor orbital parameters, predict errors, and ensure optimal performance with AI-powered insights.
          </p>

          <motion.button
            onClick={onGetStarted}
            className="group inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl shadow-xl hover:shadow-2xl transition-all"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.98 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <span className="text-lg">Get Started</span>
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </motion.button>

          <div className="flex gap-8 mt-12 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span>Real-time Data</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-indigo-500 rounded-full"></div>
              <span>AI Predictions</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
              <span>2 Satellites</span>
            </div>
          </div>
        </motion.div>

        {/* Right - 3D Planet */}
        <motion.div
          className="relative h-[500px] flex items-center justify-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.3 }}
        >
          {/* Orbit rings */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              className="absolute w-64 h-64 border-2 border-blue-200 rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
            />
            <motion.div
              className="absolute w-80 h-80 border-2 border-indigo-200 rounded-full"
              animate={{ rotate: -360 }}
              transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
            />
          </div>

          {/* Earth */}
          <motion.div
            className="relative w-48 h-48 rounded-full shadow-2xl"
            animate={{ rotate: 360 }}
            transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
            style={{
              background:
                'radial-gradient(circle at 35% 35%, #60A5FA 0%, #3B82F6 30%, #2563EB 60%, #1E40AF 100%)',
              boxShadow:
                '0 20px 60px rgba(59, 130, 246, 0.4), inset -20px -20px 40px rgba(0, 0, 0, 0.3), inset 10px 10px 30px rgba(255, 255, 255, 0.2)',
            }}
          >
            {/* Continents */}
            <svg className="absolute inset-0 w-full h-full rounded-full" viewBox="0 0 128 128">
              <defs>
                <radialGradient id="land">
                  <stop offset="0%" stopColor="#34D399" />
                  <stop offset="100%" stopColor="#10B981" />
                </radialGradient>
              </defs>
              <ellipse cx="45" cy="40" rx="20" ry="14" fill="url(#land)" opacity="0.5" />
              <ellipse cx="75" cy="55" rx="18" ry="22" fill="url(#land)" opacity="0.5" />
              <ellipse cx="35" cy="80" rx="14" ry="18" fill="url(#land)" opacity="0.5" />
              <circle cx="90" cy="35" r="10" fill="url(#land)" opacity="0.5" />
            </svg>

            {/* Atmosphere glow */}
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background:
                  'radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.5) 0%, transparent 70%)',
              }}
            />

            {/* Outer atmosphere */}
            <div
              className="absolute -inset-3 rounded-full"
              style={{
                background: 'radial-gradient(circle, rgba(96, 165, 250, 0.4) 0%, transparent 70%)',
                filter: 'blur(12px)',
              }}
            />
          </motion.div>

          {/* Satellites */}
          {[
            { angle: 45, radius: 140, color: '#8B5CF6', label: 'GEO' },
            { angle: 225, radius: 170, color: '#3B82F6', label: 'MEO' },
          ].map((sat, i) => {
            const x = Math.cos((sat.angle * Math.PI) / 180) * sat.radius;
            const y = Math.sin((sat.angle * Math.PI) / 180) * sat.radius;

            return (
              <motion.div
                key={i}
                className="absolute"
                style={{
                  left: '50%',
                  top: '50%',
                  marginLeft: `${x}px`,
                  marginTop: `${y}px`,
                }}
                animate={{
                  marginLeft: [
                    `${x}px`,
                    `${Math.cos(((sat.angle + 180) * Math.PI) / 180) * sat.radius}px`,
                    `${x}px`,
                  ],
                  marginTop: [
                    `${y}px`,
                    `${Math.sin(((sat.angle + 180) * Math.PI) / 180) * sat.radius}px`,
                    `${y}px`,
                  ],
                }}
                transition={{ duration: 15 + i * 5, repeat: Infinity, ease: 'linear' }}
              >
                <div
                  className="w-3 h-3 rounded-full shadow-lg"
                  style={{
                    backgroundColor: sat.color,
                    boxShadow: `0 0 20px ${sat.color}`,
                  }}
                />
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </div>
  );
}
