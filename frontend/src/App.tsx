import { useState } from 'react';
import { LandingPage } from './components/landing-page';
import { DashboardLayout } from './components/dashboard-layout';

export default function App() {
  const [showDashboard, setShowDashboard] = useState(false);

  if (!showDashboard) {
    return <LandingPage onGetStarted={() => setShowDashboard(true)} />;
  }

  return <DashboardLayout />;
}
