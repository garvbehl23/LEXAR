"use client";

import { useEffect, useRef } from "react";

export function useScrollToBottom<T>(deps: T[]) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}
