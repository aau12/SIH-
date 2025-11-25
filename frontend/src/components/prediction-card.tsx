import { motion } from 'framer-motion';

interface PredictionCardProps {
  horizon: string;
  predictions: {
    clock: number;
    x: number;
    y: number;
    z: number;
  };
  delay?: number;
}

export function PredictionCard({ horizon, predictions, delay = 0 }: PredictionCardProps) {
  const getSeverityColor = (value: number) => {
    if (value > 8) return 'text-[#FF5252]';
    if (value > 5) return 'text-[#FFD700]';
    return 'text-[#00C853]';
  };

  return (
    <motion.div
      className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4 hover:border-[#00E5FF]/30 transition-all min-w-[200px]"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay }}
      whileHover={{ y: -2, boxShadow: '0 10px 30px rgba(0, 229, 255, 0.1)' }}
    >
      <div className="mb-3">
        <span className="text-[#00E5FF]">{horizon}</span>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#B0B6C2]">Clock:</span>
          <span className={`${getSeverityColor(predictions.clock)}`}>
            {predictions.clock.toFixed(2)}ns
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#B0B6C2]">X:</span>
          <span className={`${getSeverityColor(predictions.x)}`}>
            {predictions.x.toFixed(2)}m
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#B0B6C2]">Y:</span>
          <span className={`${getSeverityColor(predictions.y)}`}>
            {predictions.y.toFixed(2)}m
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#B0B6C2]">Z:</span>
          <span className={`${getSeverityColor(predictions.z)}`}>
            {predictions.z.toFixed(2)}m
          </span>
        </div>
      </div>
    </motion.div>
  );
}
