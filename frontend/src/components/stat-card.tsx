import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  badge?: {
    text: string;
    color: 'green' | 'red' | 'blue' | 'yellow';
  };
  delay?: number;
}

export function StatCard({ title, value, icon: Icon, trend, badge, delay = 0 }: StatCardProps) {
  const badgeColors = {
    green: 'bg-[#00C853]/20 text-[#00C853]',
    red: 'bg-[#FF5252]/20 text-[#FF5252]',
    blue: 'bg-[#00E5FF]/20 text-[#00E5FF]',
    yellow: 'bg-[#FFD700]/20 text-[#FFD700]',
  };

  return (
    <motion.div
      className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4 hover:border-[#00E5FF]/30 transition-all group"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      whileHover={{ y: -2, boxShadow: '0 10px 30px rgba(0, 229, 255, 0.1)' }}
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-sm text-[#B0B6C2]">{title}</p>
        {Icon && (
          <div className="p-2 rounded-lg bg-[#00E5FF]/10 group-hover:bg-[#00E5FF]/20 transition-colors">
            <Icon className="w-4 h-4 text-[#00E5FF]" />
          </div>
        )}
      </div>

      <div className="flex items-end justify-between">
        <div>
          <p className="text-white mb-1">{value}</p>
          {badge && (
            <span className={`px-2 py-0.5 rounded-full text-xs ${badgeColors[badge.color]}`}>
              {badge.text}
            </span>
          )}
        </div>
        {trend && (
          <div
            className={`text-xs ${
              trend === 'up'
                ? 'text-[#00C853]'
                : trend === 'down'
                ? 'text-[#FF5252]'
                : 'text-[#B0B6C2]'
            }`}
          >
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '−'}
          </div>
        )}
      </div>
    </motion.div>
  );
}
