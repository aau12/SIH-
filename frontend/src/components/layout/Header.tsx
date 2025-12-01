import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Home,
  Database,
  BarChart3,
  Brain,
  TrendingUp,
  Activity,
  Download,
  Users,
  ChevronDown,
  Orbit,
  Menu,
  X,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  label: string;
  path?: string;
  icon: React.ReactNode;
  children?: { label: string; path: string }[];
}

const navItems: NavItem[] = [
  { label: 'Home', path: '/', icon: <Home className="w-4 h-4" /> },
  { label: 'Data', path: '/data', icon: <Database className="w-4 h-4" /> },
  { label: 'Features', path: '/features', icon: <BarChart3 className="w-4 h-4" /> },
  {
    label: 'Models',
    icon: <Brain className="w-4 h-4" />,
    children: [
      { label: 'Overview', path: '/models' },
      { label: 'LightGBM', path: '/models/lightgbm' },
      { label: 'LSTM', path: '/models/lstm' },
    ],
  },
  { label: 'Predictions', path: '/predictions', icon: <TrendingUp className="w-4 h-4" /> },
  {
    label: 'Residuals',
    icon: <Activity className="w-4 h-4" />,
    children: [
      { label: 'Analysis', path: '/residuals' },
      { label: 'Box Plots', path: '/residuals/boxplots' },
    ],
  },
  { label: 'Downloads', path: '/downloads', icon: <Download className="w-4 h-4" /> },
  { label: 'About', path: '/about', icon: <Users className="w-4 h-4" /> },
];

export default function Header() {
  const location = useLocation();
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path?: string, children?: { path: string }[]) => {
    if (path) return location.pathname === path;
    if (children) return children.some((child) => location.pathname === child.path);
    return false;
  };

  return (
    <header className="sticky top-0 z-50 w-full bg-white/95 dark:bg-slate-900/95 backdrop-blur supports-[backdrop-filter]:bg-white/80 border-b border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="relative">
              <Orbit className="w-8 h-8 text-orbit-500 group-hover:text-orbit-600 transition-colors" />
              <div className="absolute inset-0 bg-orbit-500/20 rounded-full blur-lg opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-orbit-600 to-orbit-400 bg-clip-text text-transparent">
              ORBITHQ
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => (
              <div
                key={item.label}
                className="relative"
                onMouseEnter={() => item.children && setOpenDropdown(item.label)}
                onMouseLeave={() => setOpenDropdown(null)}
              >
                {item.path ? (
                  <Link
                    to={item.path}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200',
                      isActive(item.path)
                        ? 'text-orbit-600 bg-orbit-50 dark:bg-orbit-900/30'
                        : 'text-slate-600 dark:text-slate-300 hover:text-orbit-600 hover:bg-slate-100 dark:hover:bg-slate-800'
                    )}
                  >
                    {item.icon}
                    {item.label}
                  </Link>
                ) : (
                  <button
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200',
                      isActive(undefined, item.children)
                        ? 'text-orbit-600 bg-orbit-50 dark:bg-orbit-900/30'
                        : 'text-slate-600 dark:text-slate-300 hover:text-orbit-600 hover:bg-slate-100 dark:hover:bg-slate-800'
                    )}
                  >
                    {item.icon}
                    {item.label}
                    <ChevronDown
                      className={cn(
                        'w-4 h-4 transition-transform duration-200',
                        openDropdown === item.label && 'rotate-180'
                      )}
                    />
                  </button>
                )}

                {/* Dropdown Menu */}
                <AnimatePresence>
                  {item.children && openDropdown === item.label && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.15 }}
                      className="absolute top-full left-0 mt-1 w-48 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 py-1 overflow-hidden"
                    >
                      {item.children.map((child) => (
                        <Link
                          key={child.path}
                          to={child.path}
                          className={cn(
                            'block px-4 py-2 text-sm transition-colors',
                            location.pathname === child.path
                              ? 'text-orbit-600 bg-orbit-50 dark:bg-orbit-900/30'
                              : 'text-slate-600 dark:text-slate-300 hover:text-orbit-600 hover:bg-slate-50 dark:hover:bg-slate-700'
                          )}
                        >
                          {child.label}
                        </Link>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 text-slate-600 hover:text-orbit-600 transition-colors"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900"
          >
            <nav className="px-4 py-4 space-y-1">
              {navItems.map((item) => (
                <div key={item.label}>
                  {item.path ? (
                    <Link
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={cn(
                        'flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-lg transition-colors',
                        isActive(item.path)
                          ? 'text-orbit-600 bg-orbit-50 dark:bg-orbit-900/30'
                          : 'text-slate-600 dark:text-slate-300'
                      )}
                    >
                      {item.icon}
                      {item.label}
                    </Link>
                  ) : (
                    <div className="space-y-1">
                      <div className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-slate-500 dark:text-slate-400">
                        {item.icon}
                        {item.label}
                      </div>
                      <div className="pl-8 space-y-1">
                        {item.children?.map((child) => (
                          <Link
                            key={child.path}
                            to={child.path}
                            onClick={() => setMobileMenuOpen(false)}
                            className={cn(
                              'block px-4 py-2 text-sm rounded-lg transition-colors',
                              location.pathname === child.path
                                ? 'text-orbit-600 bg-orbit-50 dark:bg-orbit-900/30'
                                : 'text-slate-600 dark:text-slate-300'
                            )}
                          >
                            {child.label}
                          </Link>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
