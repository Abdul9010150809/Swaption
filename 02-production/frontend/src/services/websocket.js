/**
 * WebSocket service for real-time communication with the backend
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectInterval = 1000; // Start with 1 second
    this.maxReconnectInterval = 30000; // Max 30 seconds
    this.subscribers = new Map();
    this.isConnected = false;
    this.url = null;
  }

  /**
   * Connect to WebSocket server
   * @param {string} url - WebSocket URL
   * @param {Object} options - Connection options
   */
  connect(url, options = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    this.url = url;
    const protocols = options.protocols || [];
    const token = options.token || localStorage.getItem('authToken');

    // Add authorization header if token exists
    const wsUrl = token ? `${url}?token=${token}` : url;

    try {
      this.ws = new WebSocket(wsUrl, protocols);

      this.ws.onopen = (event) => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectInterval = 1000;
        this.emit('connected', event);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          this.emit('error', { type: 'parse_error', error });
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        this.emit('disconnected', event);

        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', { type: 'websocket_error', error });
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.emit('error', { type: 'connection_error', error });
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.isConnected = false;
    this.reconnectAttempts = 0;
  }

  /**
   * Send message to server
   * @param {Object} message - Message to send
   */
  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        const messageString = JSON.stringify(message);
        this.ws.send(messageString);
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.emit('error', { type: 'send_error', error });
      }
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  /**
   * Subscribe to WebSocket events
   * @param {string} event - Event name
   * @param {Function} callback - Event callback
   */
  subscribe(event, callback) {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, []);
    }
    this.subscribers.get(event).push(callback);
  }

  /**
   * Unsubscribe from WebSocket events
   * @param {string} event - Event name
   * @param {Function} callback - Event callback to remove
   */
  unsubscribe(event, callback) {
    if (this.subscribers.has(event)) {
      const callbacks = this.subscribers.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to subscribers
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  emit(event, data) {
    if (this.subscribers.has(event)) {
      this.subscribers.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} callback:`, error);
        }
      });
    }
  }

  /**
   * Handle incoming messages
   * @param {Object} message - Received message
   */
  handleMessage(message) {
    const { type, data, id } = message;

    // Emit message to subscribers
    this.emit(type, data);

    // Handle specific message types
    switch (type) {
      case 'pricing_result':
        this.emit('pricingUpdate', data);
        break;
      case 'market_data':
        this.emit('marketDataUpdate', data);
        break;
      case 'error':
        this.emit('serverError', data);
        break;
      case 'heartbeat':
        // Respond to heartbeat
        this.send({ type: 'heartbeat_ack', id });
        break;
      default:
        this.emit('message', message);
    }
  }

  /**
   * Schedule reconnection attempt
   */
  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectInterval);

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      if (!this.isConnected && this.url) {
        this.connect(this.url);
      }
    }, delay);
  }

  /**
   * Get connection status
   * @returns {boolean} Connection status
   */
  get isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Set connection status
   * @param {boolean} value - Connection status
   */
  set isConnected(value) {
    this._isConnected = value;
  }

  /**
   * Get current connection status
   * @returns {boolean} Connection status
   */
  getConnectionStatus() {
    return this._isConnected;
  }
}

// Create singleton instance
const websocketService = new WebSocketService();

export default websocketService;

// Export individual methods for convenience
export const {
  connect,
  disconnect,
  send,
  subscribe,
  unsubscribe,
  getConnectionStatus
} = websocketService;