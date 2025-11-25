import { useState } from 'react';
import {
  Home,
  FileText,
  Filter,
  Settings,
  Brain,
  TrendingUp,
  BarChart3,
  Satellite,
  Menu,
  X,
  Activity,
} from 'lucide-react';
import { HomePage } from './pages/home-page';
import { DataOverviewPage } from './pages/data-overview-page';
import { PreprocessingPage } from './pages/preprocessing-page';
import { FeatureEngineeringPage } from './pages/feature-engineering-page';
import { ModelResultsPage } from './pages/model-results-page';
import { Day8PredictionsPage } from './pages/day8-predictions-page';
import { ResidualAnalysisPage } from './pages/residual-analysis-page';
import { SatelliteNetworkPage } from './pages/satellite-network-page';
import { RealtimePredictionsPage } from './pages/realtime-predictions-page';

type PageType =
  | 'home'
  | 'data-overview'
  | 'preprocessing'
  | 'feature-engineering'
  | 'model-results'
  | 'realtime-predictions'
  | 'day8-predictions'
  | 'residual-analysis'
  | 'satellite-network';

const menuItems = [
  { id: 'home' as PageType, label: 'Home', icon: Home },
  { id: 'realtime-predictions' as PageType, label: 'Real-time Predictions', icon: Activity },
  { id: 'data-overview' as PageType, label: 'Data Overview', icon: FileText },
  { id: 'preprocessing' as PageType, label: 'Preprocessing', icon: Filter },
  { id: 'feature-engineering' as PageType, label: 'Feature Engineering', icon: Settings },
  { id: 'model-results' as PageType, label: 'Model Results', icon: Brain },
  { id: 'day8-predictions' as PageType, label: 'Day-8 Predictions', icon: TrendingUp },
  { id: 'residual-analysis' as PageType, label: 'Residual Analysis', icon: BarChart3 },
  { id: 'satellite-network' as PageType, label: 'Satellite Network', icon: Satellite },
];

export function DashboardLayout() {
  const [activePage, setActivePage] = useState<PageType>('home');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const renderPage = () => {
    switch (activePage) {
      case 'home':
        return <HomePage onNavigate={setActivePage} />;
      case 'realtime-predictions':
        return <RealtimePredictionsPage />;
      case 'data-overview':
        return <DataOverviewPage />;
      case 'preprocessing':
        return <PreprocessingPage />;
      case 'feature-engineering':
        return <FeatureEngineeringPage />;
      case 'model-results':
        return <ModelResultsPage />;
      case 'day8-predictions':
        return <Day8PredictionsPage />;
      case 'residual-analysis':
        return <ResidualAnalysisPage />;
      case 'satellite-network':
        return <SatelliteNetworkPage />;
      default:
        return <HomePage onNavigate={setActivePage} />;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'w-72' : 'w-0'
        } bg-white border-r border-gray-200 transition-all duration-300 overflow-hidden`}
      >
        <div className="p-6">
          {/* Logo */}
          <div className="flex items-center gap-3 mb-10">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
              <Satellite className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-gray-900">Satellite Monitor</h2>
              <p className="text-xs text-gray-500">GEO & MEO Tracking</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  className={`w-full flex items-center gap-4 px-4 py-3 rounded-xl transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/30'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-sm">{item.label}</span>
                  {isActive && (
                    <div className="ml-auto w-2 h-2 bg-white rounded-full shadow-sm"></div>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-8 py-5 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-xl transition-colors"
            >
              {sidebarOpen ? (
                <X className="w-5 h-5 text-gray-600" />
              ) : (
                <Menu className="w-5 h-5 text-gray-600" />
              )}
            </button>
            <div>
              <h1 className="text-gray-900">
                {menuItems.find((item) => item.id === activePage)?.label}
              </h1>
              <p className="text-sm text-gray-500">Monitor and analyze satellite performance</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-4 py-2 bg-green-50 text-green-700 rounded-xl text-sm border border-green-200">
              <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              Active
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-8">{renderPage()}</main>
      </div>
    </div>
  );
}
