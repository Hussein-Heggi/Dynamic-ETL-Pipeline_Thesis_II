const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export interface WebSocketMessage {
  type: string;
  stage?: string;
  progress?: number;
  message: string;
  timestamp: string;
  data?: any;
}

export class PipelineWebSocket {
  private ws: WebSocket | null = null;
  private runId: string;
  private onMessage: (message: WebSocketMessage) => void;
  private onError: (error: Event) => void;
  private onClose: () => void;

  constructor(
    runId: string,
    onMessage: (message: WebSocketMessage) => void,
    onError: (error: Event) => void = () => {},
    onClose: () => void = () => {}
  ) {
    this.runId = runId;
    this.onMessage = onMessage;
    this.onError = onError;
    this.onClose = onClose;
  }

  connect() {
    const url = `${WS_BASE_URL}/ws/pipeline/${this.runId}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.onError(error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
      this.onClose();
    };
  }

  send(message: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    }
  }

  requestStatus() {
    this.send('status');
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
