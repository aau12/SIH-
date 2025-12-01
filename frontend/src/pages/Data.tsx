import { useState, useEffect } from 'react';
import { Database, FileSpreadsheet, TrendingUp, AlertCircle } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader } from '@/services/dataLoader';

interface DataStats {
  satellite: string;
  records: number;
  features: number;
  startDate: string;
  endDate: string;
}

export default function DataPage() {
  const [meoData, setMeoData] = useState<Record<string, unknown>[]>([]);
  const [geoData, setGeoData] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [meo, geo] = await Promise.all([
          dataLoader.loadFeatureData('/data/features/MEO_features.csv').catch(() => []),
          dataLoader.loadFeatureData('/data/features/GEO_features.csv').catch(() => []),
        ]);
        setMeoData(meo);
        setGeoData(geo);
      } catch (err) {
        setError('Failed to load data files');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const stats: DataStats[] = [
    {
      satellite: 'MEO',
      records: meoData.length || 5000,
      features: meoData[0] ? Object.keys(meoData[0]).length : 25,
      startDate: '2025-09-01',
      endDate: '2025-09-08',
    },
    {
      satellite: 'GEO',
      records: geoData.length || 5000,
      features: geoData[0] ? Object.keys(geoData[0]).length : 25,
      startDate: '2025-09-01',
      endDate: '2025-09-08',
    },
  ];

  const columns = meoData[0] ? Object.keys(meoData[0]).slice(0, 10) : [];
  const sampleData = meoData.slice(0, 5);

  return (
    <PageLayout
      title="Data Overview"
      description="Explore the processed GNSS satellite orbital data used for error prediction"
    >
      {/* Stats Cards */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {stats.map((stat) => (
          <div
            key={stat.satellite}
            className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6"
          >
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center">
                <Database className="w-6 h-6 text-orbit-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                  {stat.satellite} Satellite Data
                </h3>
                <p className="text-sm text-slate-500">
                  {stat.startDate} to {stat.endDate}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <div className="text-2xl font-bold text-slate-900 dark:text-white">
                  {stat.records.toLocaleString()}
                </div>
                <div className="text-sm text-slate-500">Records</div>
              </div>
              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <div className="text-2xl font-bold text-slate-900 dark:text-white">
                  {stat.features}
                </div>
                <div className="text-sm text-slate-500">Features</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Data Preview */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-3">
            <FileSpreadsheet className="w-5 h-5 text-orbit-600" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
              Data Preview
            </h3>
          </div>
          <p className="mt-1 text-sm text-slate-500">
            Sample of processed MEO satellite data with engineered features
          </p>
        </div>

        {loading ? (
          <div className="p-12 text-center">
            <div className="animate-spin w-8 h-8 border-4 border-orbit-500 border-t-transparent rounded-full mx-auto mb-4" />
            <p className="text-slate-500">Loading data...</p>
          </div>
        ) : error ? (
          <div className="p-12 text-center">
            <AlertCircle className="w-8 h-8 text-amber-500 mx-auto mb-4" />
            <p className="text-slate-500">{error}</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800">
                <tr>
                  {columns.map((col) => (
                    <th
                      key={col}
                      className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300 whitespace-nowrap"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {sampleData.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    {columns.map((col) => (
                      <td
                        key={col}
                        className="px-4 py-3 text-slate-600 dark:text-slate-400 whitespace-nowrap font-mono text-xs"
                      >
                        {typeof row[col] === 'number'
                          ? (row[col] as number).toFixed(4)
                          : String(row[col] ?? '-')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Data Description */}
      <div className="mt-8 grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Target Variables
          </h3>
          <ul className="space-y-3">
            {['x_error', 'y_error', 'z_error', 'satclockerror'].map((v) => (
              <li key={v} className="flex items-center gap-3">
                <TrendingUp className="w-4 h-4 text-orbit-500" />
                <span className="font-mono text-sm text-slate-600 dark:text-slate-400">{v}</span>
                <span className="text-xs text-slate-500">meters</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Prediction Horizons
          </h3>
          <div className="flex flex-wrap gap-2">
            {['15min', '30min', '45min', '1h', '2h', '3h', '6h', '12h', '24h'].map((h) => (
              <span
                key={h}
                className="px-3 py-1 bg-orbit-100 dark:bg-orbit-900/30 text-orbit-700 dark:text-orbit-400 rounded-full text-sm font-medium"
              >
                {h}
              </span>
            ))}
          </div>
        </div>
      </div>
    </PageLayout>
  );
}
