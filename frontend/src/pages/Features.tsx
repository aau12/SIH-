import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Layers } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader } from '@/services/dataLoader';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const featureCategories = [
  {
    name: 'Error Variables',
    features: ['x_error', 'y_error', 'z_error', 'satclockerror'],
    description: 'Target prediction variables representing orbital positioning errors',
  },
  {
    name: 'Temporal Features',
    features: ['hour', 'day_of_week', 'month', 'is_weekend'],
    description: 'Time-based features extracted from timestamps',
  },
  {
    name: 'Lag Features',
    features: ['x_error_lag_1', 'y_error_lag_1', 'z_error_lag_1', 'clock_lag_1'],
    description: 'Previous time step values for capturing temporal dependencies',
  },
  {
    name: 'Rolling Statistics',
    features: ['x_error_mean_6h', 'y_error_std_6h', 'z_error_max_12h'],
    description: 'Rolling window statistics for trend detection',
  },
];

export default function FeaturesPage() {
  const [featureStats, setFeatureStats] = useState<{ name: string; meo: number; geo: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStats() {
      try {
        // Generate sample feature importance data
        const stats = [
          { name: 'x_error_lag_1', meo: 0.85, geo: 0.82 },
          { name: 'y_error_lag_1', meo: 0.78, geo: 0.81 },
          { name: 'z_error_lag_1', meo: 0.72, geo: 0.75 },
          { name: 'hour', meo: 0.45, geo: 0.52 },
          { name: 'rolling_mean_6h', meo: 0.65, geo: 0.61 },
          { name: 'rolling_std_6h', meo: 0.58, geo: 0.55 },
          { name: 'day_of_week', meo: 0.32, geo: 0.38 },
          { name: 'clock_lag_1', meo: 0.71, geo: 0.68 },
        ];
        setFeatureStats(stats);
      } catch (error) {
        console.error('Failed to load feature stats:', error);
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  return (
    <PageLayout
      title="Feature Analysis"
      description="Explore the engineered features used for satellite orbit error prediction"
    >
      {/* Feature Categories */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {featureCategories.map((category) => (
          <div
            key={category.name}
            className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
                <Layers className="w-5 h-5 text-orbit-600" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                {category.name}
              </h3>
            </div>
            <p className="text-sm text-slate-500 mb-4">{category.description}</p>
            <div className="flex flex-wrap gap-2">
              {category.features.map((feature) => (
                <span
                  key={feature}
                  className="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-full text-sm font-mono"
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-5 h-5 text-orbit-600" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
            Feature Importance Comparison
          </h3>
        </div>

        {loading ? (
          <div className="h-80 flex items-center justify-center">
            <div className="animate-spin w-8 h-8 border-4 border-orbit-500 border-t-transparent rounded-full" />
          </div>
        ) : (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureStats} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Bar dataKey="meo" name="MEO" fill="#0ea5e9" radius={[0, 4, 4, 0]} />
                <Bar dataKey="geo" name="GEO" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Feature Engineering Process */}
      <div className="mt-8 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          Feature Engineering Process
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-4">
            <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-xl font-bold text-orbit-600">1</span>
            </div>
            <h4 className="font-medium text-slate-900 dark:text-white mb-2">Data Cleaning</h4>
            <p className="text-sm text-slate-500">
              Remove outliers, handle missing values, normalize timestamps
            </p>
          </div>
          <div className="text-center p-4">
            <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-xl font-bold text-orbit-600">2</span>
            </div>
            <h4 className="font-medium text-slate-900 dark:text-white mb-2">Feature Extraction</h4>
            <p className="text-sm text-slate-500">
              Create lag features, rolling statistics, temporal encodings
            </p>
          </div>
          <div className="text-center p-4">
            <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-xl font-bold text-orbit-600">3</span>
            </div>
            <h4 className="font-medium text-slate-900 dark:text-white mb-2">Feature Selection</h4>
            <p className="text-sm text-slate-500">
              Select most informative features using importance scores
            </p>
          </div>
        </div>
      </div>
    </PageLayout>
  );
}
