import { Link, useLocation } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BreadcrumbsProps {
  className?: string;
}

const routeLabels: Record<string, string> = {
  '': 'Home',
  'data': 'Data',
  'features': 'Features',
  'models': 'Models',
  'lightgbm': 'LightGBM',
  'lstm': 'LSTM',
  'predictions': 'Predictions',
  'residuals': 'Residuals',
  'boxplots': 'Box Plots',
  'downloads': 'Downloads',
  'about': 'About',
};

export default function Breadcrumbs({ className }: BreadcrumbsProps) {
  const location = useLocation();
  const pathnames = location.pathname.split('/').filter((x) => x);

  if (pathnames.length === 0) return null;

  return (
    <nav className={cn('flex items-center gap-2 text-sm', className)}>
      <Link
        to="/"
        className="flex items-center gap-1 text-slate-500 hover:text-orbit-600 transition-colors"
      >
        <Home className="w-4 h-4" />
        <span className="sr-only">Home</span>
      </Link>

      {pathnames.map((segment, index) => {
        const path = `/${pathnames.slice(0, index + 1).join('/')}`;
        const isLast = index === pathnames.length - 1;
        const label = routeLabels[segment] || segment;

        return (
          <div key={path} className="flex items-center gap-2">
            <ChevronRight className="w-4 h-4 text-slate-400" />
            {isLast ? (
              <span className="font-medium text-slate-900 dark:text-slate-100">{label}</span>
            ) : (
              <Link
                to={path}
                className="text-slate-500 hover:text-orbit-600 transition-colors"
              >
                {label}
              </Link>
            )}
          </div>
        );
      })}
    </nav>
  );
}
