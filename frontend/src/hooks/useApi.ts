/**
 * Custom React hooks for API data fetching
 */

import { useState, useEffect, useCallback } from 'react';
import { api, Prediction, ModelMetrics, DataStats } from '../services/api';

// Generic API hook
export function useApi<T>(
  fetcher: () => Promise<T>,
  dependencies: any[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await fetcher();
      setData(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { data, loading, error, refetch };
}

// Specific hooks for common operations

export function usePredictions(satellite: 'MEO' | 'GEO') {
  return useApi<Prediction[]>(
    () => api.getPredictions(satellite),
    [satellite]
  );
}

export function useDataStats(satellite: 'MEO' | 'GEO') {
  return useApi<DataStats>(
    () => api.getDataStats(satellite),
    [satellite]
  );
}

export function useModelMetrics(satellite: 'MEO' | 'GEO') {
  return useApi<ModelMetrics[]>(
    () => api.getModelMetrics(satellite),
    [satellite]
  );
}

export function useHealthCheck() {
  return useApi(
    () => api.healthCheck(),
    []
  );
}

// Polling hook for real-time updates
export function usePolling<T>(
  fetcher: () => Promise<T>,
  interval: number = 5000,
  enabled: boolean = true
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const fetchData = async () => {
      try {
        const result = await fetcher();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchData();

    // Set up polling
    const intervalId = setInterval(fetchData, interval);

    return () => clearInterval(intervalId);
  }, [fetcher, interval, enabled]);

  return { data, loading, error };
}

// Real-time predictions with polling
export function useRealtimePredictions(
  satellite: 'MEO' | 'GEO',
  interval: number = 10000,
  enabled: boolean = true
) {
  return usePolling<Prediction[]>(
    () => api.getPredictions(satellite),
    interval,
    enabled
  );
}
