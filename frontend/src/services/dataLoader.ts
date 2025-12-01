import Papa from 'papaparse';

export interface Prediction {
  horizon_label: string;
  horizon_minutes: number;
  timestamp_predicted: string;
  x_error_pred: number;
  y_error_pred: number;
  z_error_pred: number;
  satclockerror_pred: number;
}

export interface PredictionData {
  satellite_type: string;
  prediction_count: number;
  horizons: string[];
  predictions: Prediction[];
}

export interface ResidualSummary {
  satellite: string;
  variable: string;
  horizon_minutes: number;
  rmse: number;
  mae: number;
  bias: number;
  std: number;
  W_shapiro?: number;
  p_shapiro?: number;
  normality_flag: string;
}

export interface ModelMetrics {
  [variable: string]: {
    [horizon: string]: {
      train_rmse: number;
      val_rmse: number;
      train_mae: number;
      val_mae: number;
      training_time: number;
      n_estimators: number;
    };
  };
}

export interface EvaluationMetrics {
  satellite: string;
  variable: string;
  horizon: string;
  rmse: number;
  mae: number;
  r2: number;
}

export interface BoxPlotData {
  variable: string;
  horizon: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
}

class DataLoaderService {
  private baseUrl: string;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }

  private async fetchFile(path: string): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load file: ${path} (${response.status})`);
    }
    return response;
  }

  async loadJSON<T = unknown>(path: string): Promise<T> {
    const response = await this.fetchFile(path);
    return response.json();
  }

  async loadCSV<T = Record<string, unknown>>(path: string): Promise<T[]> {
    const response = await this.fetchFile(path);
    const text = await response.text();
    
    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          resolve(results.data as T[]);
        },
        error: (error: Error) => {
          reject(new Error(`CSV parsing error: ${error.message}`));
        },
      });
    });
  }

  async loadImage(path: string): Promise<string> {
    const response = await this.fetchFile(path);
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async loadText(path: string): Promise<string> {
    const response = await this.fetchFile(path);
    return response.text();
  }

  // Domain-specific loaders
  async loadPredictions(path: string): Promise<PredictionData> {
    return this.loadJSON<PredictionData>(path);
  }

  async loadModelMetrics(path: string): Promise<ModelMetrics> {
    return this.loadJSON<ModelMetrics>(path);
  }

  async loadResidualSummary(path: string): Promise<ResidualSummary[]> {
    return this.loadCSV<ResidualSummary>(path);
  }

  async loadEvaluationMetrics(path: string): Promise<EvaluationMetrics[]> {
    return this.loadCSV<EvaluationMetrics>(path);
  }

  async loadFeatureData(path: string): Promise<Record<string, unknown>[]> {
    return this.loadCSV(path);
  }

  // Utility methods
  async loadBatch<T = unknown>(paths: string[]): Promise<T[]> {
    return Promise.all(paths.map((path) => this.loadJSON<T>(path)));
  }

  async fileExists(path: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}${path}`, { method: 'HEAD' });
      return response.ok;
    } catch {
      return false;
    }
  }

  getImageUrl(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  // Generate box plot data from residual summary
  generateBoxPlotData(residuals: ResidualSummary[]): BoxPlotData[] {
    const boxPlotData: BoxPlotData[] = [];
    
    // Group by variable and horizon
    const grouped = residuals.reduce((acc, r) => {
      const key = `${r.variable}-${r.horizon_minutes}`;
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(r);
      return acc;
    }, {} as Record<string, ResidualSummary[]>);

    for (const [key, items] of Object.entries(grouped)) {
      const [variable, horizonStr] = key.split('-');
      const values = items.map(i => i.rmse);
      
      const sorted = [...values].sort((a, b) => a - b);
      const q1Index = Math.floor(sorted.length * 0.25);
      const medianIndex = Math.floor(sorted.length * 0.5);
      const q3Index = Math.floor(sorted.length * 0.75);

      boxPlotData.push({
        variable,
        horizon: `${horizonStr}min`,
        min: sorted[0] || 0,
        q1: sorted[q1Index] || 0,
        median: sorted[medianIndex] || 0,
        q3: sorted[q3Index] || 0,
        max: sorted[sorted.length - 1] || 0,
        outliers: [],
      });
    }

    return boxPlotData;
  }
}

export const dataLoader = new DataLoaderService();
export default DataLoaderService;
