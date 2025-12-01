import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number, decimals: number = 4): string {
  return value.toFixed(decimals);
}

export function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleString();
}
