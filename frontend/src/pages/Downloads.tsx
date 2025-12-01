import { Download, FileJson, FileSpreadsheet, Image, FileText } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';
import { dataLoader } from '@/services/dataLoader';

interface DownloadItem {
  name: string;
  description: string;
  path: string;
  type: 'json' | 'csv' | 'image' | 'text';
  category: string;
}

const downloadItems: DownloadItem[] = [
  // Predictions
  {
    name: 'MEO Predictions',
    description: 'Day 8 predictions for MEO satellite',
    path: '/data/predictions/MEO_Day8_Predictions.json',
    type: 'json',
    category: 'Predictions',
  },
  {
    name: 'GEO Predictions',
    description: 'Day 8 predictions for GEO satellite',
    path: '/data/predictions/GEO_Day8_Predictions.json',
    type: 'json',
    category: 'Predictions',
  },
  // Models
  {
    name: 'LightGBM MEO Metrics',
    description: 'Model performance metrics for MEO',
    path: '/data/models/metrics/lightgbm_meo_metrics.json',
    type: 'json',
    category: 'Model Metrics',
  },
  {
    name: 'LightGBM GEO Metrics',
    description: 'Model performance metrics for GEO',
    path: '/data/models/metrics/lightgbm_geo_metrics.json',
    type: 'json',
    category: 'Model Metrics',
  },
  // Residuals
  {
    name: 'Residual Summary',
    description: 'Complete residual analysis summary',
    path: '/data/residuals/residual_summary.csv',
    type: 'csv',
    category: 'Residuals',
  },
  // Evaluation
  {
    name: 'MEO Evaluation Metrics',
    description: 'Evaluation metrics for MEO models',
    path: '/data/evaluation/MEO_metrics.csv',
    type: 'csv',
    category: 'Evaluation',
  },
  {
    name: 'GEO Evaluation Metrics',
    description: 'Evaluation metrics for GEO models',
    path: '/data/evaluation/GEO_metrics.csv',
    type: 'csv',
    category: 'Evaluation',
  },
  // Features
  {
    name: 'MEO Features Dataset',
    description: 'Engineered features for MEO satellite',
    path: '/data/features/MEO_features.csv',
    type: 'csv',
    category: 'Features',
  },
  {
    name: 'GEO Features Dataset',
    description: 'Engineered features for GEO satellite',
    path: '/data/features/GEO_features.csv',
    type: 'csv',
    category: 'Features',
  },
];

const getIcon = (type: string) => {
  switch (type) {
    case 'json':
      return <FileJson className="w-5 h-5" />;
    case 'csv':
      return <FileSpreadsheet className="w-5 h-5" />;
    case 'image':
      return <Image className="w-5 h-5" />;
    default:
      return <FileText className="w-5 h-5" />;
  }
};

const categories = [...new Set(downloadItems.map((item) => item.category))];

export default function DownloadsPage() {
  const handleDownload = async (item: DownloadItem) => {
    try {
      const url = dataLoader.getImageUrl(item.path);
      const response = await fetch(url);
      const blob = await response.blob();
      
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = item.path.split('/').pop() || 'download';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download file. Please try again.');
    }
  };

  return (
    <PageLayout
      title="Downloads"
      description="Download datasets, model outputs, and analysis results"
    >
      {categories.map((category) => (
        <div key={category} className="mb-8">
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
            {category}
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {downloadItems
              .filter((item) => item.category === category)
              .map((item) => (
                <div
                  key={item.path}
                  className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 hover:border-orbit-300 dark:hover:border-orbit-700 transition-colors"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center text-orbit-600">
                      {getIcon(item.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-slate-900 dark:text-white truncate">
                        {item.name}
                      </h3>
                      <p className="text-sm text-slate-500 mt-1">{item.description}</p>
                      <div className="flex items-center gap-2 mt-3">
                        <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 rounded text-xs uppercase">
                          {item.type}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDownload(item)}
                    className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2 bg-orbit-500 hover:bg-orbit-600 text-white rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                </div>
              ))}
          </div>
        </div>
      ))}

      {/* Usage Note */}
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-6 mt-8">
        <h3 className="font-semibold text-slate-900 dark:text-white mb-2">Usage Notes</h3>
        <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
          <li>• All datasets are in standard formats (CSV, JSON) for easy integration</li>
          <li>• Prediction files contain multi-horizon forecasts for both MEO and GEO satellites</li>
          <li>• Model metrics include training and validation performance for all horizons</li>
          <li>• Feature datasets contain preprocessed and engineered features ready for modeling</li>
        </ul>
      </div>
    </PageLayout>
  );
}
