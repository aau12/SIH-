import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Database,
  BarChart3,
  Brain,
  TrendingUp,
  Activity,
  Satellite,
  Globe,
  Zap,
  Shield,
} from 'lucide-react';

const features = [
  {
    icon: <Database className="w-6 h-6" />,
    title: 'Data Processing',
    description: 'Advanced preprocessing of GNSS orbital data with feature engineering',
    link: '/data',
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: 'Feature Analysis',
    description: 'Comprehensive statistical analysis and visualization of extracted features',
    link: '/features',
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: 'ML Models',
    description: 'State-of-the-art LightGBM and LSTM models for error prediction',
    link: '/models',
  },
  {
    icon: <TrendingUp className="w-6 h-6" />,
    title: 'Predictions',
    description: 'Multi-horizon predictions from 15 minutes to 24 hours ahead',
    link: '/predictions',
  },
  {
    icon: <Activity className="w-6 h-6" />,
    title: 'Residual Analysis',
    description: 'In-depth analysis of model residuals with box plot visualizations',
    link: '/residuals',
  },
  {
    icon: <Satellite className="w-6 h-6" />,
    title: 'Satellite Support',
    description: 'Support for MEO and GEO satellite orbit error prediction',
    link: '/data',
  },
];

const stats = [
  { label: 'Prediction Horizons', value: '9', icon: <Zap className="w-5 h-5" /> },
  { label: 'Error Variables', value: '4', icon: <Globe className="w-5 h-5" /> },
  { label: 'Satellite Types', value: '2', icon: <Satellite className="w-5 h-5" /> },
  { label: 'Model Accuracy', value: '>95%', icon: <Shield className="w-5 h-5" /> },
];

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-orbit-950 to-slate-900 text-white">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-orbit-500/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-orbit-400/10 rounded-full blur-3xl animate-pulse delay-1000" />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-orbit-300 via-orbit-400 to-orbit-500 bg-clip-text text-transparent">
                ORBITHQ
              </span>
            </h1>
            <p className="mt-6 text-xl sm:text-2xl text-slate-300 max-w-3xl mx-auto">
              Advanced GNSS Satellite Orbit Error Prediction System
            </p>
            <p className="mt-4 text-lg text-slate-400 max-w-2xl mx-auto">
              Leveraging machine learning to predict satellite positioning errors across multiple time horizons for enhanced navigation accuracy
            </p>
            
            <div className="mt-10 flex flex-wrap justify-center gap-4">
              <Link
                to="/predictions"
                className="px-8 py-3 bg-orbit-500 hover:bg-orbit-600 text-white font-semibold rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-orbit-500/25"
              >
                View Predictions
              </Link>
              <Link
                to="/models"
                className="px-8 py-3 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-lg backdrop-blur transition-all duration-200"
              >
                Explore Models
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="inline-flex items-center justify-center w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 text-orbit-600 rounded-lg mb-4">
                  {stat.icon}
                </div>
                <div className="text-3xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
                <div className="text-sm text-slate-600 dark:text-slate-400">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-slate-50 dark:bg-slate-800/50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white">
              Platform Capabilities
            </h2>
            <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
              Comprehensive tools for satellite orbit error analysis and prediction
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <Link
                  to={feature.link}
                  className="block h-full p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-orbit-300 dark:hover:border-orbit-700 hover:shadow-lg transition-all duration-200 group"
                >
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 text-orbit-600 rounded-lg mb-4 group-hover:bg-orbit-500 group-hover:text-white transition-colors">
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-slate-600 dark:text-slate-400">{feature.description}</p>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-orbit-600 to-orbit-500 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Explore the Data?
          </h2>
          <p className="text-orbit-100 mb-8 max-w-2xl mx-auto">
            Dive into our comprehensive analysis of satellite orbit errors and discover insights from our machine learning models.
          </p>
          <Link
            to="/data"
            className="inline-block px-8 py-3 bg-white text-orbit-600 font-semibold rounded-lg hover:bg-orbit-50 transition-colors"
          >
            Get Started
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-400 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <Satellite className="w-6 h-6 text-orbit-500" />
              <span className="text-xl font-bold text-white">ORBITHQ</span>
            </div>
            <p className="text-sm">
              GNSS Satellite Orbit Error Prediction System
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
