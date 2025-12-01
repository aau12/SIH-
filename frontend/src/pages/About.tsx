import { Users, Satellite, Brain, Target, Github, Mail, Globe } from 'lucide-react';
import PageLayout from '@/components/layout/PageLayout';

const teamMembers = [
  {
    name: 'Research Team',
    role: 'GNSS Analysis & ML Development',
    description: 'Developed the core prediction algorithms and model architectures',
  },
  {
    name: 'Data Science Team',
    role: 'Feature Engineering & Validation',
    description: 'Created advanced features and validated model performance',
  },
  {
    name: 'Engineering Team',
    role: 'System Architecture & Deployment',
    description: 'Built the platform infrastructure and visualization tools',
  },
];

const projectHighlights = [
  {
    icon: <Satellite className="w-6 h-6" />,
    title: 'Multi-Satellite Support',
    description: 'Predictions for both MEO and GEO satellite constellations',
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: 'Advanced ML Models',
    description: 'LightGBM and LSTM architectures optimized for time-series prediction',
  },
  {
    icon: <Target className="w-6 h-6" />,
    title: 'Multi-Horizon Forecasting',
    description: 'Predictions from 15 minutes to 24 hours ahead',
  },
];

export default function AboutPage() {
  return (
    <PageLayout
      title="About ORBITHQ"
      description="Learn about our satellite orbit error prediction platform"
    >
      {/* Project Overview */}
      <div className="bg-gradient-to-br from-orbit-600 to-orbit-500 rounded-2xl p-8 mb-8 text-white">
        <h2 className="text-2xl font-bold mb-4">Project Overview</h2>
        <p className="text-orbit-100 leading-relaxed">
          ORBITHQ is an advanced satellite orbit error prediction system designed to improve GNSS
          navigation accuracy. Using state-of-the-art machine learning models, we predict orbital
          positioning errors across multiple time horizons, enabling better satellite-based
          navigation and positioning services.
        </p>
      </div>

      {/* Project Highlights */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        {projectHighlights.map((highlight) => (
          <div
            key={highlight.title}
            className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6"
          >
            <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-lg flex items-center justify-center text-orbit-600 mb-4">
              {highlight.icon}
            </div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
              {highlight.title}
            </h3>
            <p className="text-slate-600 dark:text-slate-400">{highlight.description}</p>
          </div>
        ))}
      </div>

      {/* Technical Details */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
          Technical Approach
        </h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="font-medium text-slate-900 dark:text-white mb-3">Data Processing</h3>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
              <li>• High-frequency GNSS orbital data collection</li>
              <li>• Outlier detection and data cleaning</li>
              <li>• Feature engineering with temporal patterns</li>
              <li>• Rolling statistics and lag features</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium text-slate-900 dark:text-white mb-3">Model Architecture</h3>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
              <li>• LightGBM for gradient boosting predictions</li>
              <li>• LSTM networks for sequence modeling</li>
              <li>• Multi-output prediction for all error variables</li>
              <li>• Horizon-specific model optimization</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Team Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
          Our Team
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {teamMembers.map((member) => (
            <div
              key={member.name}
              className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6"
            >
              <div className="w-12 h-12 bg-orbit-100 dark:bg-orbit-900/30 rounded-full flex items-center justify-center text-orbit-600 mb-4">
                <Users className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white">{member.name}</h3>
              <p className="text-sm text-orbit-600 mb-2">{member.role}</p>
              <p className="text-sm text-slate-600 dark:text-slate-400">{member.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
          Technology Stack
        </h2>
        <div className="flex flex-wrap gap-3">
          {[
            'Python',
            'LightGBM',
            'TensorFlow/Keras',
            'Pandas',
            'NumPy',
            'React',
            'TypeScript',
            'Tailwind CSS',
            'Recharts',
            'Vite',
          ].map((tech) => (
            <span
              key={tech}
              className="px-4 py-2 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg text-sm font-medium"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>

      {/* Contact */}
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-6">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
          Get in Touch
        </h2>
        <div className="flex flex-wrap gap-4">
          <a
            href="#"
            className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-600 dark:text-slate-300 hover:border-orbit-300 transition-colors"
          >
            <Github className="w-4 h-4" />
            GitHub Repository
          </a>
          <a
            href="#"
            className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-600 dark:text-slate-300 hover:border-orbit-300 transition-colors"
          >
            <Mail className="w-4 h-4" />
            Contact Us
          </a>
          <a
            href="#"
            className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-600 dark:text-slate-300 hover:border-orbit-300 transition-colors"
          >
            <Globe className="w-4 h-4" />
            Documentation
          </a>
        </div>
      </div>
    </PageLayout>
  );
}
