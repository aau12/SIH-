import { useState } from 'react';
import { ChevronDown, ChevronUp, Layers, TrendingUp, Clock, Target, BarChart3, Wind } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface FeatureCategory {
  id: string;
  title: string;
  icon: any;
  count: number;
  color: string;
  features: { name: string; value: number; unit?: string }[];
}

const featureCategories: FeatureCategory[] = [
  {
    id: 'base',
    title: 'Base Error Columns',
    icon: Layers,
    count: 4,
    color: '#00E5FF',
    features: [
      { name: 'x_error', value: 4.23, unit: 'm' },
      { name: 'y_error', value: 3.87, unit: 'm' },
      { name: 'z_error', value: 5.12, unit: 'm' },
      { name: 'satclockerror', value: 0.42, unit: 'ns' },
    ],
  },
  {
    id: 'lag',
    title: 'Lag Features',
    icon: Wind,
    count: 32,
    color: '#42A5F5',
    features: [
      { name: 'x_error_lag_1', value: 4.18 },
      { name: 'x_error_lag_2', value: 4.32 },
      { name: 'x_error_lag_4', value: 4.45 },
      { name: 'x_error_lag_8', value: 4.67 },
      { name: 'x_error_lag_12', value: 4.82 },
      { name: 'x_error_lag_16', value: 4.91 },
      { name: 'x_error_lag_24', value: 5.02 },
      { name: 'x_error_lag_48', value: 5.34 },
      { name: 'y_error_lag_1', value: 3.82 },
      { name: 'y_error_lag_2', value: 3.91 },
      { name: 'y_error_lag_4', value: 4.05 },
      { name: 'y_error_lag_8', value: 4.18 },
      { name: 'y_error_lag_12', value: 4.32 },
      { name: 'y_error_lag_16', value: 4.43 },
      { name: 'y_error_lag_24', value: 4.58 },
      { name: 'y_error_lag_48', value: 4.89 },
      { name: 'z_error_lag_1', value: 5.07 },
      { name: 'z_error_lag_2', value: 5.18 },
      { name: 'z_error_lag_4', value: 5.34 },
      { name: 'z_error_lag_8', value: 5.51 },
      { name: 'z_error_lag_12', value: 5.68 },
      { name: 'z_error_lag_16', value: 5.82 },
      { name: 'z_error_lag_24', value: 6.01 },
      { name: 'z_error_lag_48', value: 6.45 },
      { name: 'satclockerror_lag_1', value: 0.41 },
      { name: 'satclockerror_lag_2', value: 0.43 },
      { name: 'satclockerror_lag_4', value: 0.45 },
      { name: 'satclockerror_lag_8', value: 0.48 },
      { name: 'satclockerror_lag_12', value: 0.51 },
      { name: 'satclockerror_lag_16', value: 0.53 },
      { name: 'satclockerror_lag_24', value: 0.57 },
      { name: 'satclockerror_lag_48', value: 0.64 },
    ],
  },
  {
    id: 'rolling',
    title: 'Rolling Window Features',
    icon: BarChart3,
    count: 48,
    color: '#FF5252',
    features: [
      { name: 'x_error_rolling_mean_3', value: 4.21 },
      { name: 'x_error_rolling_std_3', value: 0.87 },
      { name: 'x_error_rolling_min_3', value: 2.34 },
      { name: 'x_error_rolling_max_3', value: 6.78 },
      { name: 'x_error_rolling_mean_6', value: 4.35 },
      { name: 'x_error_rolling_std_6', value: 1.12 },
      { name: 'x_error_rolling_min_6', value: 1.89 },
      { name: 'x_error_rolling_max_6', value: 7.23 },
      { name: 'x_error_rolling_mean_12', value: 4.52 },
      { name: 'x_error_rolling_std_12', value: 1.45 },
      { name: 'x_error_rolling_min_12', value: 1.45 },
      { name: 'x_error_rolling_max_12', value: 8.12 },
      { name: 'y_error_rolling_mean_3', value: 3.84 },
      { name: 'y_error_rolling_std_3', value: 0.76 },
      { name: 'y_error_rolling_min_3', value: 2.12 },
      { name: 'y_error_rolling_max_3', value: 6.23 },
      { name: 'y_error_rolling_mean_6', value: 3.98 },
      { name: 'y_error_rolling_std_6', value: 0.98 },
      { name: 'y_error_rolling_min_6', value: 1.67 },
      { name: 'y_error_rolling_max_6', value: 6.89 },
      { name: 'y_error_rolling_mean_12', value: 4.15 },
      { name: 'y_error_rolling_std_12', value: 1.23 },
      { name: 'y_error_rolling_min_12', value: 1.23 },
      { name: 'y_error_rolling_max_12', value: 7.56 },
      { name: 'z_error_rolling_mean_3', value: 5.09 },
      { name: 'z_error_rolling_std_3', value: 1.02 },
      { name: 'z_error_rolling_min_3', value: 2.67 },
      { name: 'z_error_rolling_max_3', value: 7.89 },
      { name: 'z_error_rolling_mean_6', value: 5.28 },
      { name: 'z_error_rolling_std_6', value: 1.34 },
      { name: 'z_error_rolling_min_6', value: 2.23 },
      { name: 'z_error_rolling_max_6', value: 8.45 },
      { name: 'z_error_rolling_mean_12', value: 5.51 },
      { name: 'z_error_rolling_std_12', value: 1.67 },
      { name: 'z_error_rolling_min_12', value: 1.89 },
      { name: 'z_error_rolling_max_12', value: 9.23 },
      { name: 'satclockerror_rolling_mean_3', value: 0.41 },
      { name: 'satclockerror_rolling_std_3', value: 0.08 },
      { name: 'satclockerror_rolling_min_3', value: 0.21 },
      { name: 'satclockerror_rolling_max_3', value: 0.67 },
      { name: 'satclockerror_rolling_mean_6', value: 0.43 },
      { name: 'satclockerror_rolling_std_6', value: 0.11 },
      { name: 'satclockerror_rolling_min_6', value: 0.18 },
      { name: 'satclockerror_rolling_max_6', value: 0.72 },
      { name: 'satclockerror_rolling_mean_12', value: 0.46 },
      { name: 'satclockerror_rolling_std_12', value: 0.14 },
      { name: 'satclockerror_rolling_min_12', value: 0.15 },
      { name: 'satclockerror_rolling_max_12', value: 0.81 },
    ],
  },
  {
    id: 'trend',
    title: 'Trend Features (Velocity & Acceleration)',
    icon: TrendingUp,
    count: 8,
    color: '#FFD700',
    features: [
      { name: 'x_error_diff1', value: 0.12, unit: 'm/h' },
      { name: 'x_error_diff2', value: -0.03, unit: 'm/h²' },
      { name: 'y_error_diff1', value: 0.09, unit: 'm/h' },
      { name: 'y_error_diff2', value: -0.02, unit: 'm/h²' },
      { name: 'z_error_diff1', value: 0.15, unit: 'm/h' },
      { name: 'z_error_diff2', value: -0.05, unit: 'm/h²' },
      { name: 'satclockerror_diff1', value: 0.02, unit: 'ns/h' },
      { name: 'satclockerror_diff2', value: -0.01, unit: 'ns/h²' },
    ],
  },
  {
    id: 'time',
    title: 'Time Features',
    icon: Clock,
    count: 5,
    color: '#00C853',
    features: [
      { name: 'hour', value: 14 },
      { name: 'hour_sin', value: 0.78 },
      { name: 'hour_cos', value: -0.62 },
      { name: 'day_of_week', value: 3 },
      { name: 'day_index', value: 245 },
    ],
  },
  {
    id: 'target',
    title: 'Target Features (Future Horizons)',
    icon: Target,
    count: 36,
    color: '#E040FB',
    features: [
      { name: 'x_error_t+1', value: 4.28 },
      { name: 'x_error_t+2', value: 4.35 },
      { name: 'x_error_t+3', value: 4.42 },
      { name: 'x_error_t+4', value: 4.51 },
      { name: 'x_error_t+8', value: 4.78 },
      { name: 'x_error_t+12', value: 5.02 },
      { name: 'x_error_t+24', value: 5.68 },
      { name: 'x_error_t+48', value: 6.89 },
      { name: 'x_error_t+96', value: 8.45 },
      { name: 'y_error_t+1', value: 3.92 },
      { name: 'y_error_t+2', value: 3.98 },
      { name: 'y_error_t+3', value: 4.05 },
      { name: 'y_error_t+4', value: 4.13 },
      { name: 'y_error_t+8', value: 4.38 },
      { name: 'y_error_t+12', value: 4.61 },
      { name: 'y_error_t+24', value: 5.23 },
      { name: 'y_error_t+48', value: 6.34 },
      { name: 'y_error_t+96', value: 7.89 },
      { name: 'z_error_t+1', value: 5.18 },
      { name: 'z_error_t+2', value: 5.26 },
      { name: 'z_error_t+3', value: 5.35 },
      { name: 'z_error_t+4', value: 5.45 },
      { name: 'z_error_t+8', value: 5.78 },
      { name: 'z_error_t+12', value: 6.08 },
      { name: 'z_error_t+24', value: 6.89 },
      { name: 'z_error_t+48', value: 8.23 },
      { name: 'z_error_t+96', value: 10.12 },
      { name: 'satclockerror_t+1', value: 0.43 },
      { name: 'satclockerror_t+2', value: 0.44 },
      { name: 'satclockerror_t+3', value: 0.46 },
      { name: 'satclockerror_t+4', value: 0.47 },
      { name: 'satclockerror_t+8', value: 0.52 },
      { name: 'satclockerror_t+12', value: 0.57 },
      { name: 'satclockerror_t+24', value: 0.68 },
      { name: 'satclockerror_t+48', value: 0.84 },
      { name: 'satclockerror_t+96', value: 1.12 },
    ],
  },
];

export function FeatureInsights() {
  const [expandedCategories, setExpandedCategories] = useState<string[]>(['base']);

  const toggleCategory = (id: string) => {
    setExpandedCategories((prev) =>
      prev.includes(id) ? prev.filter((cat) => cat !== id) : [...prev, id]
    );
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl text-[#00E5FF] mb-2">Feature Insights</h2>
        <p className="text-[#B0B6C2]">Complete overview of all 134 machine learning features</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-6 gap-4 mb-6">
        {featureCategories.map((cat, index) => (
          <motion.div
            key={cat.id}
            className="bg-[#11141A] border border-[#1A1D26] rounded-xl p-4 hover:border-[#00E5FF]/30 transition-all"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center mb-3"
              style={{ backgroundColor: `${cat.color}20` }}
            >
              <cat.icon className="w-5 h-5" style={{ color: cat.color }} />
            </div>
            <p className="text-xs text-[#B0B6C2] mb-1">{cat.title}</p>
            <p className="text-xl" style={{ color: cat.color }}>
              {cat.count}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Feature Categories */}
      <div className="space-y-4">
        {featureCategories.map((category) => {
          const isExpanded = expandedCategories.includes(category.id);
          const Icon = category.icon;

          return (
            <motion.div
              key={category.id}
              className="bg-[#11141A] border border-[#1A1D26] rounded-xl overflow-hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(category.id)}
                className="w-full p-6 flex items-center justify-between hover:bg-[#1A1D26] transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div
                    className="w-12 h-12 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${category.color}20` }}
                  >
                    <Icon className="w-6 h-6" style={{ color: category.color }} />
                  </div>
                  <div className="text-left">
                    <h3 className="mb-1" style={{ color: category.color }}>
                      {category.title}
                    </h3>
                    <p className="text-sm text-[#B0B6C2]">{category.count} features</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <span
                    className="px-3 py-1 rounded-full text-sm"
                    style={{
                      backgroundColor: `${category.color}20`,
                      color: category.color,
                    }}
                  >
                    {category.count}
                  </span>
                  {isExpanded ? (
                    <ChevronUp className="w-5 h-5 text-[#B0B6C2]" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-[#B0B6C2]" />
                  )}
                </div>
              </button>

              {/* Category Features */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="px-6 pb-6 border-t border-[#1A1D26]">
                      <div className="grid grid-cols-4 gap-3 mt-4">
                        {category.features.map((feature, index) => (
                          <motion.div
                            key={feature.name}
                            className="bg-[#0E0E12] rounded-lg p-3 hover:bg-[#1A1D26] transition-colors"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.02 }}
                          >
                            <p className="text-xs text-[#B0B6C2] mb-2 font-mono">{feature.name}</p>
                            <p className="text-white">
                              {feature.value.toFixed(2)}
                              {feature.unit && <span className="text-xs text-[#B0B6C2] ml-1">{feature.unit}</span>}
                            </p>
                            {/* Mini visualization bar */}
                            <div className="mt-2 h-1 bg-[#1A1D26] rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${Math.min((Math.abs(feature.value) / 10) * 100, 100)}%`,
                                  backgroundColor: category.color,
                                  opacity: 0.6,
                                }}
                              ></div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Total Features Summary */}
      <div className="mt-6 bg-gradient-to-r from-[#00E5FF]/10 to-[#42A5F5]/10 border border-[#00E5FF]/30 rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl text-[#00E5FF] mb-1">Total Features</h3>
            <p className="text-sm text-[#B0B6C2]">Complete feature set for satellite error prediction</p>
          </div>
          <div className="text-right">
            <p className="text-4xl text-[#00E5FF]">134</p>
            <p className="text-xs text-[#B0B6C2] mt-1">ML Features</p>
          </div>
        </div>
      </div>
    </div>
  );
}
