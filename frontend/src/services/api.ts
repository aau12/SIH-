/**
 * Data Loader Service
 * Loads static files (CSV, JSON, images) from the local filesystem
 * No backend API - all data comes from pre-generated static files
 */

import Papa from 'papaparse';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface Prediction {
  satellite: string;
  timestamp_current: string;
  timestamp_predicted: string;
  horizon_label: string;
  horizon_minutes: number;
  x_error_pred: number;
  y_error_pred: number;
  z_error_pred: number;
  satclockerror_pred: number;
}

export interface ModelMetrics {
  model_name: string;
  satellite?: string;
  rmse: number;
  mae: number;
  r2: number;
  mape?: number;
}

export interface DataStats {
  total_rows: number;
  time_range: {
    start: string;
    end: string;
  };
  missing_values: {
    [key: string]: number;
  };
  sampling_interval?: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ResidualStats {
  mean: number;
  std: number;
  skewness: number;
  kurtosis: number;
  normality_test?: {
    statistic: number;
    p_value: number;
  };
}

// ============================================================================
// DATA LOADER SERVICE
// ============================================================================

class DataLoaderService {
  /**
   * Load and parse a CSV file from the given path
   * @param path - Absolute or relative path to the CSV file
   * @returns Promise resolving to array of parsed objects
   */
  async loadCSV(path: string): Promise<any[]> {
    try {
      const response = await fetch(path);
      
      if (!response.ok) {
        throw new Error(`Failed to load CSV from ${path}: ${response.statusText}`);
      }
      
      const csvText = await response.text();
      
      return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results: Papa.ParseResult<any>) => {
            if (results.errors.length > 0) {
              console.warn(`CSV parsing warnings for ${path}:`, results.errors);
            }
            resolve(results.data);
          },
          error: (error: Error) => {
            reject(new Error(`CSV parsing error for ${path}: ${error.message}`));
          }
        });
      });
    } catch (error) {
      console.error(`Error loading CSV from ${path}:`, error);
      throw error;
    }
  }

  /**
   * Load and parse a JSON file from the given path
   * @param path - Absolute or relative path to the JSON file
   * @returns Promise resolving to parsed JSON data
   */
  async loadJSON(path: string): Promise<any> {
    try {
      const response = await fetch(path);
      
      if (!response.ok) {
        throw new Error(`Failed to load JSON from ${path}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error loading JSON from ${path}:`, error);
      throw error;
    }
  }

  /**
   * Load an image and convert it to base64 string
   * @param path - Absolute or relative path to the image file
   * @returns Promise resolving to base64 encoded image string
   */
  async loadImage(path: string): Promise<string> {
    try {
      const response = await fetch(path);
      
      if (!response.ok) {
        throw new Error(`Failed to load image from ${path}: ${response.statusText}`);
      }
      
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          if (typeof reader.result === 'string') {
            resolve(reader.result);
          } else {
            reject(new Error(`Failed to convert image to base64: ${path}`));
          }
        };
        reader.onerror = () => reject(new Error(`FileReader error for ${path}`));
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error(`Error loading image from ${path}:`, error);
      throw error;
    }
  }

  /**
   * Load a text file from the given path
   * @param path - Absolute or relative path to the text file
   * @returns Promise resolving to file content as string
   */
  async loadText(path: string): Promise<string> {
    try {
      const response = await fetch(path);
      
      if (!response.ok) {
        throw new Error(`Failed to load text from ${path}: ${response.statusText}`);
      }
      
      return await response.text();
    } catch (error) {
      console.error(`Error loading text from ${path}:`, error);
      throw error;
    }
  }

  /**
   * Check if a file exists and is accessible
   * @param path - Path to check
   * @returns Promise resolving to true if file exists
   */
  async fileExists(path: string): Promise<boolean> {
    try {
      const response = await fetch(path, { method: 'HEAD' });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Load multiple files in parallel
   * @param paths - Array of file paths to load
   * @returns Promise resolving to array of loaded data (order preserved)
   */
  async loadBatch(paths: string[]): Promise<any[]> {
    try {
      const promises = paths.map(path => {
        if (path.endsWith('.json')) {
          return this.loadJSON(path);
        } else if (path.endsWith('.csv')) {
          return this.loadCSV(path);
        } else if (path.match(/\.(png|jpg|jpeg|svg|gif|webp)$/i)) {
          return this.loadImage(path);
        } else {
          return this.loadText(path);
        }
      });
      
      return await Promise.all(promises);
    } catch (error) {
      console.error('Error in batch loading:', error);
      throw error;
    }
  }

  /**
   * Get file metadata without loading full content
   * @param path - Path to file
   * @returns Promise resolving to metadata object
   */
  async getFileMetadata(path: string): Promise<{
    size: number;
    type: string;
    lastModified?: string;
  }> {
    try {
      const response = await fetch(path, { method: 'HEAD' });
      
      if (!response.ok) {
        throw new Error(`Failed to get metadata for ${path}: ${response.statusText}`);
      }
      
      return {
        size: parseInt(response.headers.get('content-length') || '0'),
        type: response.headers.get('content-type') || 'unknown',
        lastModified: response.headers.get('last-modified') || undefined,
      };
    } catch (error) {
      console.error(`Error getting metadata for ${path}:`, error);
      throw error;
    }
  }

  // ============================================================================
  // CONVENIENCE METHODS (Optional - use generic loaders above if preferred)
  // ============================================================================

  /**
   * Load predictions from a file (CSV or JSON)
   * @param path - Path to predictions file
   * @returns Promise resolving to array of predictions
   */
  async loadPredictions(path: string): Promise<Prediction[]> {
    return path.endsWith('.json') ? this.loadJSON(path) : this.loadCSV(path);
  }

  /**
   * Load model metrics from a file
   * @param path - Path to metrics file (JSON or CSV)
   * @returns Promise resolving to array of model metrics
   */
  async loadModelMetrics(path: string): Promise<ModelMetrics[]> {
    return path.endsWith('.json') ? this.loadJSON(path) : this.loadCSV(path);
  }

  /**
   * Load data statistics from a file
   * @param path - Path to stats file (JSON)
   * @returns Promise resolving to data statistics
   */
  async loadDataStats(path: string): Promise<DataStats> {
    return this.loadJSON(path);
  }

  /**
   * Load feature importance from a file
   * @param path - Path to feature importance file (JSON or CSV)
   * @returns Promise resolving to array of feature importance scores
   */
  async loadFeatureImportance(path: string): Promise<FeatureImportance[]> {
    return path.endsWith('.json') ? this.loadJSON(path) : this.loadCSV(path);
  }

  /**
   * Load residual statistics from a file
   * @param path - Path to residuals file (JSON)
   * @returns Promise resolving to residual statistics
   */
  async loadResiduals(path: string): Promise<ResidualStats> {
    return this.loadJSON(path);
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

// Export singleton instance
export const dataLoader = new DataLoaderService();

// Export default
export default dataLoader;
