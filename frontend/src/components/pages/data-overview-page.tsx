import { useState, useEffect } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, AlertCircle } from 'lucide-react';

interface DataRow {
  utc_time: string;
  'x_error (m)': number;
  'y_error (m)': number;
  'z_error (m)': number;
  'satclockerror (m)': number;
}

export function DataOverviewPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'GEO' | 'MEO'>('all');
  const [meoData, setMeoData] = useState<DataRow[]>([]);
  const [geoData, setGeoData] = useState<DataRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [meoResponse, geoResponse] = await Promise.all([
          fetch('http://localhost:8000/data/sample/MEO?limit=50'),
          fetch('http://localhost:8000/data/sample/GEO?limit=50')
        ]);
        
        if (meoResponse.ok && geoResponse.ok) {
          const meoJson = await meoResponse.json();
          const geoJson = await geoResponse.json();
          setMeoData(meoJson);
          setGeoData(geoJson);
        } else {
          setError('Failed to fetch data');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  const combinedData = [
    ...meoData.slice(0, 10).map((row, idx) => ({
      id: `meo-${idx}`,
      satellite: `MEO Satellite`,
      type: 'MEO' as const,
      xError: row['x_error (m)'].toFixed(3),
      yError: row['y_error (m)'].toFixed(3),
      zError: row['z_error (m)'].toFixed(3),
      clockError: row['satclockerror (m)'].toFixed(3),
      status: 'Active',
      lastUpdate: row.utc_time,
    })),
    ...geoData.slice(0, 10).map((row, idx) => ({
      id: `geo-${idx}`,
      satellite: `GEO Satellite`,
      type: 'GEO' as const,
      xError: row['x_error (m)'].toFixed(3),
      yError: row['y_error (m)'].toFixed(3),
      zError: row['z_error (m)'].toFixed(3),
      clockError: row['satclockerror (m)'].toFixed(3),
      status: 'Active',
      lastUpdate: row.utc_time,
    }))
  ];

  const filteredData = combinedData.filter((item) => {
    const matchesSearch = item.satellite.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || item.type === filterType;
    return matchesSearch && matchesFilter;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
          <div>
            <h3 className="font-semibold text-red-900">Error Loading Data</h3>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-3xl p-8 shadow-sm border border-gray-200">
        <h2 className="mb-6 text-gray-900">Satellite Data Overview</h2>
        <div className="flex gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search satellites..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-12 pr-4 py-3 border border-gray-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Filters */}
          <div className="flex gap-3">
            <button
              onClick={() => setFilterType('all')}
              className={`px-6 py-3 rounded-2xl transition-all ${
                filterType === 'all'
                  ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilterType('GEO')}
              className={`px-6 py-3 rounded-2xl transition-all ${
                filterType === 'GEO'
                  ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/30'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              GEO
            </button>
            <button
              onClick={() => setFilterType('MEO')}
              className={`px-6 py-3 rounded-2xl transition-all ${
                filterType === 'MEO'
                  ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              MEO
            </button>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-3xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Satellite
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  X Error (m)
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Y Error (m)
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Z Error (m)
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Clock Error (ns)
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-8 py-4 text-left text-xs text-gray-500 uppercase tracking-wider">
                  Last Update
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredData.map((row) => (
                <tr key={row.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-8 py-5 text-gray-800">{row.satellite}</td>
                  <td className="px-8 py-5">
                    <span
                      className={`px-3 py-1 rounded-full text-xs ${
                        row.type === 'GEO'
                          ? 'bg-purple-100 text-purple-700'
                          : 'bg-blue-100 text-blue-700'
                      }`}
                    >
                      {row.type}
                    </span>
                  </td>
                  <td className="px-8 py-5 text-gray-600">{row.xError}</td>
                  <td className="px-8 py-5 text-gray-600">{row.yError}</td>
                  <td className="px-8 py-5 text-gray-600">{row.zError}</td>
                  <td className="px-8 py-5 text-gray-600">{row.clockError}</td>
                  <td className="px-8 py-5">
                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs">
                      {row.status}
                    </span>
                  </td>
                  <td className="px-8 py-5 text-gray-500">{row.lastUpdate}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div className="px-8 py-5 border-t border-gray-200 flex items-center justify-between bg-gray-50">
          <p className="text-sm text-gray-600">
            Showing {filteredData.length} of 2 total satellites
          </p>
          <div className="text-xs text-gray-500">
            Last updated: Just now
          </div>
        </div>
      </div>
    </div>
  );
}
