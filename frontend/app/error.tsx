"use client";

import { ErrorState } from "@/components/ui/primitives";

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return <ErrorState message={error.message || "Something went wrong"} onRetry={reset} />;
}