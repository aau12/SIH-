import { Layers, Plus, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';

export function FeatureEngineeringPage() {
  const featureCategories = [
    {
      title: 'Lag Features',
      count: 32,
      description: 'Historical time-shifted values',
      color: 'blue',
      examples: ['x_error_lag_1', 'y_error_lag_2', 'z_error_lag_4'],
    },
    {
      title: 'Rolling Window',
      count: 48,
      description: 'Statistical aggregations over time',
      color: 'purple',
      examples: ['x_error_rolling_mean_3', 'y_error_rolling_std_6', 'z_error_rolling_max_12'],
    },
    {
      title: 'Trend Features',
      count: 8,
      description: 'Velocity and acceleration metrics',
      color: 'green',
      examples: ['x_error_diff1', 'y_error_diff2', 'z_error_diff1'],
    },
    {
      title: 'Time Features',
      count: 5,
      description: 'Temporal characteristics',
      color: 'orange',
      examples: ['hour', 'hour_sin', 'day_of_week'],
    },
  ];

  const correlationData = [
    { feature: 'x_error_lag_1', correlation: 0.92 },
    { feature: 'clock_lag_2', correlation: 0.87 },
    { feature: 'rolling_mean_12', correlation: 0.84 },
    { feature: 'z_error_lag_4', correlation: 0.79 },
    { feature: 'y_error_lag_1', correlation: 0.76 },
  ];

  const getColorClasses = (color: string) => {
    const colors = {
      blue: { bg: 'bg-blue-100', text: 'text-blue-600' },
      purple: { bg: 'bg-purple-100', text: 'text-purple-600' },
      green: { bg: 'bg-green-100', text: 'text-green-600' },
      orange: { bg: 'bg-orange-100', text: 'text-orange-600' },
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
        <h2 className="mb-2 text-gray-800">Feature Engineering</h2>
        <p className="text-gray-600">Created features for enhanced model performance</p>
      </div>

      {/* Feature Categories */}
      <div className="grid grid-cols-2 gap-6">
        {featureCategories.map((category, index) => {
          const colors = getColorClasses(category.color);

          return (
            <motion.div
              key={category.title}
              className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-start gap-4 mb-4">
                <div className={`p-3 ${colors.bg} rounded-xl`}>
                  <Layers className={`w-6 h-6 ${colors.text}`} />
                </div>
                <div className="flex-1">
                  <h3 className="text-gray-800 mb-1">{category.title}</h3>
                  <p className="text-sm text-gray-600">{category.description}</p>
                </div>
                <div className={`px-3 py-1 ${colors.bg} ${colors.text} rounded-full text-sm`}>
                  {category.count}
                </div>
              </div>

              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-500 mb-2">Example Features:</p>
                <div className="space-y-1">
                  {category.examples.map((example) => (
                    <div
                      key={example}
                      className="text-sm text-gray-700 font-mono bg-white px-3 py-2 rounded-lg"
                    >
                      {example}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Feature Selection & Correlation */}
      <div className="grid grid-cols-2 gap-6">
        {/* Feature Selection */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-blue-100 rounded-xl">
              <Plus className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-gray-800">New Features Created</h3>
              <p className="text-sm text-gray-600">Total engineered features</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 rounded-xl p-4">
              <p className="text-2xl text-gray-800 mb-1">93</p>
              <p className="text-sm text-gray-600">New Features</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4">
              <p className="text-2xl text-gray-800 mb-1">134</p>
              <p className="text-sm text-gray-600">Total Features</p>
            </div>
          </div>
        </div>

        {/* Top Correlations */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-green-100 rounded-xl">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h3 className="text-gray-800">Top Correlations</h3>
              <p className="text-sm text-gray-600">Highly correlated features</p>
            </div>
          </div>

          <div className="space-y-3">
            {correlationData.map((item) => (
              <div key={item.feature} className="bg-gray-50 rounded-xl p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-700 font-mono">{item.feature}</span>
                  <span className="text-sm text-green-600">{item.correlation}</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-400 to-green-600"
                    style={{ width: `${item.correlation * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl p-6 text-white shadow-lg">
        <h3 className="mb-2">Feature Engineering Summary</h3>
        <p className="text-indigo-100 text-sm mb-4">
          Successfully created 93 new features across 4 categories to enhance model predictive power
        </p>
        <div className="grid grid-cols-4 gap-6">
          <div>
            <p className="text-indigo-100 text-sm mb-1">Lag Features</p>
            <p className="text-2xl">32</p>
          </div>
          <div>
            <p className="text-indigo-100 text-sm mb-1">Rolling Window</p>
            <p className="text-2xl">48</p>
          </div>
          <div>
            <p className="text-indigo-100 text-sm mb-1">Trend Features</p>
            <p className="text-2xl">8</p>
          </div>
          <div>
            <p className="text-indigo-100 text-sm mb-1">Time Features</p>
            <p className="text-2xl">5</p>
          </div>
        </div>
      </div>
    </div>
  );
}
