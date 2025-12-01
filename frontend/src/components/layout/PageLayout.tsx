import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import Breadcrumbs from './Breadcrumbs';

interface PageLayoutProps {
  title: string;
  description?: string;
  children: ReactNode;
  showBreadcrumbs?: boolean;
}

export default function PageLayout({
  title,
  description,
  children,
  showBreadcrumbs = true,
}: PageLayoutProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="min-h-screen"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {showBreadcrumbs && <Breadcrumbs className="mb-6" />}
        
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">{title}</h1>
          {description && (
            <p className="mt-2 text-lg text-slate-600 dark:text-slate-400">{description}</p>
          )}
        </header>

        <main>{children}</main>
      </div>
    </motion.div>
  );
}
