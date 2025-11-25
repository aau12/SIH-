# Migration Guide: API Service → Data Loader Service

## Overview

The frontend has been refactored to load **static files** instead of making HTTP API calls. This change reflects the architecture where the backend generates output files (CSV, JSON, images) rather than exposing a live API.

## What Changed

### Before (API-based)
```typescript
import { api } from './services/api';

// Made HTTP requests to backend
const predictions = await api.getPredictions('MEO');
const metrics = await api.getModelMetrics('GEO');
```

### After (File-based)
```typescript
import { dataLoader } from './services/dataLoader';

// Loads static files from filesystem
const predictions = await dataLoader.loadPredictions('/data/predictions/MEO_latest.json');
const metrics = await dataLoader.loadModelMetrics('/data/metrics/GEO_metrics.json');
```

## Key Differences

| Aspect | Old (API Service) | New (Data Loader) |
|--------|------------------|-------------------|
| **Data Source** | HTTP API endpoints | Static files (CSV/JSON/images) |
| **Base URL** | `http://localhost:8000` | File paths (relative or absolute) |
| **Method Names** | `getPredictions(satellite)` | `loadPredictions(filePath)` |
| **Parameters** | Satellite type, filters | File path |
| **Dependencies** | `fetch` API | `fetch` + `papaparse` |
| **Error Handling** | HTTP status codes | File not found, parse errors |

## New Service: DataLoaderService

### Location
```
frontend/src/services/dataLoader.ts
```

### Core Methods

#### 1. **Low-Level Loaders**
```typescript
// Load and parse CSV
await dataLoader.loadCSV('/path/to/file.csv');

// Load and parse JSON
await dataLoader.loadJSON('/path/to/file.json');

// Load image as base64
await dataLoader.loadImage('/path/to/plot.png');

// Load text file
await dataLoader.loadText('/path/to/file.txt');
```

#### 2. **Domain-Specific Loaders**
```typescript
// Predictions
await dataLoader.loadPredictions('/data/predictions/MEO_latest.json');

// Model metrics
await dataLoader.loadModelMetrics('/data/metrics/MEO_metrics.json');

// Data statistics
await dataLoader.loadDataStats('/data/stats/MEO_stats.json');

// Sample data
await dataLoader.loadDataSample('/data/samples/MEO_sample.csv');

// Feature importance
await dataLoader.loadFeatureImportance('/data/features/MEO_importance.json');

// Feature statistics
await dataLoader.loadFeatureStats('/data/features/MEO_stats.json');

// Residual analysis
await dataLoader.loadResiduals('/data/residuals/MEO_residuals.json');

// Historical predictions
await dataLoader.loadHistoricalPredictions('/data/history/MEO_history.csv');

// Model comparison
await dataLoader.loadModelComparison('/data/comparison/models.json');

// Plots/charts
await dataLoader.loadPlot('/plots/MEO_forecast.png');

// Cleaned datasets
await dataLoader.loadCleanedData('/data/processed/MEO_clean_15min.csv');

// Feature-engineered data
await dataLoader.loadFeatureData('/data/features/MEO_features.csv');

// Evaluation results
await dataLoader.loadEvaluationResults('/data/evaluation/MEO_eval.json');
```

#### 3. **Utility Methods**
```typescript
// Batch load multiple files
const [predictions, metrics, stats] = await dataLoader.loadBatch([
  '/data/predictions.json',
  '/data/metrics.json',
  '/data/stats.json'
]);

// Check if file exists
const exists = await dataLoader.fileExists('/data/predictions.json');

// Get file metadata
const metadata = await dataLoader.getFileMetadata('/data/large_file.csv');
// Returns: { size: 1024000, type: 'text/csv', lastModified: '...' }
```

## Migration Steps for Components

### Step 1: Update Imports
```typescript
// OLD
import { api } from '../services/api';

// NEW
import { dataLoader } from '../services/dataLoader';
```

### Step 2: Update Method Calls

#### Example: Real-time Predictions Page

**Before:**
```typescript
const fetchPredictions = async () => {
  try {
    const data = await api.getPredictions(satellite);
    setPredictions(data);
  } catch (error) {
    setError(error.message);
  }
};
```

**After:**
```typescript
const fetchPredictions = async () => {
  try {
    // Path should be determined by your file structure
    const filePath = `/data/predictions/${satellite}_latest.json`;
    const data = await dataLoader.loadPredictions(filePath);
    setPredictions(data);
  } catch (error) {
    setError(error.message);
  }
};
```

#### Example: Model Results Page

**Before:**
```typescript
const response = await api.getModelMetrics(satellite);
```

**After:**
```typescript
const filePath = `/data/metrics/${satellite}_metrics.json`;
const response = await dataLoader.loadModelMetrics(filePath);
```

#### Example: Data Overview Page

**Before:**
```typescript
const [meoResponse, geoResponse] = await Promise.all([
  api.getDataSample('MEO', 50),
  api.getDataSample('GEO', 50)
]);
```

**After:**
```typescript
const [meoData, geoData] = await dataLoader.loadBatch([
  '/data/samples/MEO_sample.csv',
  '/data/samples/GEO_sample.csv'
]);
```

### Step 3: Update File Paths

You need to determine where your backend outputs files. Common patterns:

```typescript
// Option 1: Public folder (recommended for Vite)
const filePath = `/data/predictions/MEO_latest.json`;
// Resolves to: frontend/public/data/predictions/MEO_latest.json

// Option 2: Absolute URL (if files are hosted elsewhere)
const filePath = `https://cdn.example.com/data/predictions.json`;

// Option 3: Dynamic paths based on state
const filePath = `/data/${satellite.toLowerCase()}/predictions_${horizon}.json`;
```

## File Organization

### Recommended Structure
```
frontend/
├── public/
│   └── data/
│       ├── predictions/
│       │   ├── MEO_latest.json
│       │   ├── GEO_latest.json
│       │   ├── MEO_15min.json
│       │   └── GEO_1h.json
│       ├── metrics/
│       │   ├── MEO_metrics.json
│       │   └── GEO_metrics.json
│       ├── stats/
│       │   ├── MEO_stats.json
│       │   └── GEO_stats.json
│       ├── samples/
│       │   ├── MEO_sample.csv
│       │   └── GEO_sample.csv
│       ├── features/
│       │   ├── MEO_importance.json
│       │   ├── GEO_importance.json
│       │   ├── MEO_stats.json
│       │   └── GEO_stats.json
│       ├── residuals/
│       │   ├── MEO_residuals.json
│       │   └── GEO_residuals.json
│       ├── history/
│       │   ├── MEO_history.csv
│       │   └── GEO_history.csv
│       ├── processed/
│       │   ├── MEO_clean_15min.csv
│       │   └── GEO_clean_15min.csv
│       └── plots/
│           ├── MEO_forecast.png
│           ├── GEO_forecast.png
│           └── comparison.png
└── src/
    └── services/
        ├── dataLoader.ts  ← NEW
        └── api.ts         ← DEPRECATED
```

## Backend Integration

Your backend scripts should output files to locations accessible by the frontend:

### Option 1: Copy to Frontend Public Folder
```python
# In your backend script
import shutil
from pathlib import Path

output_dir = Path("../frontend/public/data/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# Save predictions
predictions_df.to_json(output_dir / "MEO_latest.json", orient="records")
```

### Option 2: Serve via Static File Server
```bash
# Serve backend output directory
cd backend/output
python -m http.server 8001

# Frontend loads from: http://localhost:8001/predictions.json
```

### Option 3: Build Step Integration
```json
// package.json
{
  "scripts": {
    "prebuild": "node scripts/copy-data-files.js",
    "build": "vite build"
  }
}
```

## Error Handling

The data loader provides consistent error handling:

```typescript
try {
  const data = await dataLoader.loadPredictions(filePath);
} catch (error) {
  if (error.message.includes('Failed to load')) {
    // File not found or network error
    console.error('File not accessible:', filePath);
  } else if (error.message.includes('parsing error')) {
    // CSV/JSON parsing failed
    console.error('Invalid file format');
  }
}
```

## Type Safety

All methods maintain TypeScript type safety:

```typescript
// Typed return values
const predictions: Prediction[] = await dataLoader.loadPredictions(path);
const metrics: ModelMetrics[] = await dataLoader.loadModelMetrics(path);
const stats: DataStats = await dataLoader.loadDataStats(path);
```

## Removed Features

The following features from the old API service are **not available** in the data loader:

- ❌ `trainModel()` - No backend to execute training
- ❌ `getTrainingStatus()` - No job tracking
- ❌ `uploadData()` - No file upload endpoint
- ❌ `healthCheck()` - No API to check

These operations should be performed directly via backend scripts.

## Testing

### Unit Tests
```typescript
import { dataLoader } from './dataLoader';

describe('DataLoaderService', () => {
  it('should load JSON files', async () => {
    const data = await dataLoader.loadJSON('/test/data.json');
    expect(data).toBeDefined();
  });

  it('should load CSV files', async () => {
    const data = await dataLoader.loadCSV('/test/data.csv');
    expect(Array.isArray(data)).toBe(true);
  });

  it('should handle missing files', async () => {
    await expect(
      dataLoader.loadJSON('/nonexistent.json')
    ).rejects.toThrow();
  });
});
```

### Integration Tests
```typescript
it('should load predictions for MEO satellite', async () => {
  const predictions = await dataLoader.loadPredictions(
    '/data/predictions/MEO_latest.json'
  );
  
  expect(predictions).toHaveLength(9); // 9 horizons
  expect(predictions[0]).toHaveProperty('x_error_pred');
});
```

## Performance Considerations

### Caching
Consider implementing a caching layer for frequently accessed files:

```typescript
class CachedDataLoader extends DataLoaderService {
  private cache = new Map<string, any>();

  async loadJSON(path: string): Promise<any> {
    if (this.cache.has(path)) {
      return this.cache.get(path);
    }
    
    const data = await super.loadJSON(path);
    this.cache.set(path, data);
    return data;
  }
}
```

### Lazy Loading
Load data only when needed:

```typescript
const [predictions, setPredictions] = useState<Prediction[]>([]);

useEffect(() => {
  // Only load when component mounts
  dataLoader.loadPredictions(filePath).then(setPredictions);
}, [filePath]);
```

## Checklist

- [ ] Install `papaparse` and `@types/papaparse`
- [ ] Update all component imports from `api` to `dataLoader`
- [ ] Replace API method calls with file loader methods
- [ ] Define file path structure
- [ ] Copy/generate backend output files to frontend
- [ ] Update environment variables (remove `VITE_API_URL`)
- [ ] Test all pages with static files
- [ ] Remove old `api.ts` file (or mark as deprecated)
- [ ] Update documentation

## Support

For questions or issues with the migration, refer to:
- `frontend/src/services/dataLoader.ts` - Implementation
- This guide - Migration instructions
- Component examples in `frontend/src/components/pages/`
