import { Routes, Route } from 'react-router-dom';
import Header from '@/components/layout/Header';
import Home from '@/pages/Home';
import Data from '@/pages/Data';
import Features from '@/pages/Features';
import Models from '@/pages/Models';
import ModelLightGBM from '@/pages/ModelLightGBM';
import ModelLSTM from '@/pages/ModelLSTM';
import Predictions from '@/pages/Predictions';
import Residuals from '@/pages/Residuals';
import BoxPlots from '@/pages/BoxPlots';
import Downloads from '@/pages/Downloads';
import About from '@/pages/About';

export default function App() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/data" element={<Data />} />
        <Route path="/features" element={<Features />} />
        <Route path="/models" element={<Models />} />
        <Route path="/models/lightgbm" element={<ModelLightGBM />} />
        <Route path="/models/lstm" element={<ModelLSTM />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/residuals" element={<Residuals />} />
        <Route path="/residuals/boxplots" element={<BoxPlots />} />
        <Route path="/downloads" element={<Downloads />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </div>
  );
}
