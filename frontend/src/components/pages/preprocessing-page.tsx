import { AlertTriangle, TrendingUp, Filter, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export function PreprocessingPage() {
  const steps = [
    {
      icon: AlertTriangle,
      title: 'Missing Values',
      description: 'Handle missing data points',
      status: 'completed',
      stats: { total: 1200, missing: 45, handled: 45 },
      color: 'orange',
    },
    {
      icon: TrendingUp,
      title: 'Normalization',
      description: 'Scale features to standard range',
      status: 'completed',
      stats: { method: 'Min-Max Scaling', range: '0-1' },
      color: 'blue',
    },
    {
      icon: Filter,
      title: 'Outlier Removal',
      description: 'Remove anomalous data points',
      status: 'completed',
      stats: { total: 1200, outliers: 23, removed: 23 },
      color: 'purple',
    },
    {
      icon: CheckCircle,
      title: 'Data Validation',
      description: 'Verify data integrity',
      status: 'completed',
      stats: { passed: 1132, failed: 0 },
      color: 'green',
    },
  ];

  const getColorClasses = (color: string) => {
    const colors = {
      orange: {
        bg: 'bg-orange-100',
        text: 'text-orange-600',
        border: 'border-orange-200',
      },
      blue: {
        bg: 'bg-blue-100',
        text: 'text-blue-600',
        border: 'border-blue-200',
      },
      purple: {
        bg: 'bg-purple-100',
        text: 'text-purple-600',
        border: 'border-purple-200',
      },
      green: {
        bg: 'bg-green-100',
        text: 'text-green-600',
        border: 'border-green-200',
      },
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
        <h2 className="mb-2 text-gray-800">Data Preprocessing Pipeline</h2>
        <p className="text-gray-600">Clean and prepare data for model training</p>
      </div>

      {/* Processing Steps */}
      <div className="grid grid-cols-2 gap-6">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const colors = getColorClasses(step.color);

          return (
            <motion.div
              key={step.title}
              className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-start gap-4">
                <div className={`p-3 ${colors.bg} rounded-xl`}>
                  <Icon className={`w-6 h-6 ${colors.text}`} />
                </div>
                <div className="flex-1">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-gray-800">{step.title}</h3>
                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs">
                      Completed
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">{step.description}</p>

                  {/* Stats */}
                  <div className={`bg-gray-50 rounded-xl p-4 border ${colors.border}`}>
                    <div className="space-y-2 text-sm">
                      {Object.entries(step.stats).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-gray-600 capitalize">{key.replace('_', ' ')}:</span>
                          <span className="text-gray-800">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl p-6 text-white shadow-lg">
        <h3 className="mb-4">Preprocessing Summary</h3>
        <div className="grid grid-cols-4 gap-6">
          <div>
            <p className="text-blue-100 text-sm mb-1">Total Records</p>
            <p className="text-2xl">1,200</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Clean Records</p>
            <p className="text-2xl">1,132</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Removed</p>
            <p className="text-2xl">68</p>
          </div>
          <div>
            <p className="text-blue-100 text-sm mb-1">Success Rate</p>
            <p className="text-2xl">94.3%</p>
          </div>
        </div>
      </div>
    </div>
  );
}
