import { useEffect, useRef, useCallback } from "react";
import { WS_BASE } from "@/lib/api";

export function useWebSocket(
  path: string | null,
  onMessage: (data: unknown) => void,
  onClose?: () => void
) {
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!path) return;
    const ws = new WebSocket(`${WS_BASE}${path}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch {
        onMessage(event.data);
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      onClose?.();
    };

    ws.onerror = (err) => {
      console.error("WebSocket error", err);
    };
  }, [path, onMessage, onClose]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { ws: wsRef };
}
