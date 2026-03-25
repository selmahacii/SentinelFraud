/**
 * =============================================================================
 * SENTINEL-FRAUD: WebSocket Service for Real-time Alerts
 * =============================================================================
 * Pushes fraud alerts to connected dashboard clients in real-time.
 * =============================================================================
 */

import { Server as HttpServer } from "http";
import { Server as SocketIOServer, Socket } from "socket.io";

const PORT = 3003;

// Fraud alert interface
interface FraudAlert {
  transaction_id: string;
  user_id: string;
  amount: number;
  currency: string;
  fraud_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  risk_factors: string[];
  timestamp: string;
  merchant_id: string;
  merchant_category: string;
  location: {
    latitude: number;
    longitude: number;
  };
}

// Statistics tracking
interface DashboardStats {
  total_transactions: number;
  fraud_detected: number;
  fraud_rate: number;
  avg_latency_ms: number;
  alerts_last_hour: number;
  total_amount_blocked: number;
}

// Create HTTP server and Socket.IO
const httpServer = new HttpServer();
const io = new SocketIOServer(httpServer, {
  cors: {
    origin: ["http://localhost:3000"],
    methods: ["GET", "POST"],
  },
});

// In-memory stats
let stats: DashboardStats = {
  total_transactions: 0,
  fraud_detected: 0,
  fraud_rate: 0,
  avg_latency_ms: 0,
  alerts_last_hour: 0,
  total_amount_blocked: 0,
};

// Alert history (last 100)
const alertHistory: FraudAlert[] = [];
const latencyHistory: number[] = [];

// Connection handling
io.on("connection", (socket: Socket) => {
  console.log(`[WS] Client connected: ${socket.id}`);

  // Send current stats and history on connect
  socket.emit("stats:update", stats);
  socket.emit("alerts:history", alertHistory.slice(-20));

  // Subscribe to alert levels
  socket.on("alerts:subscribe", (levels: string[]) => {
    console.log(`[WS] Client ${socket.id} subscribed to: ${levels.join(", ")}`);
    socket.join(levels.map((l) => `alerts:${l}`));
  });

  // Unsubscribe
  socket.on("alerts:unsubscribe", (levels: string[]) => {
    levels.forEach((l) => socket.leave(`alerts:${l}`));
  });

  // Disconnect
  socket.on("disconnect", () => {
    console.log(`[WS] Client disconnected: ${socket.id}`);
  });
});

// Simulate fraud alerts (for demo)
function generateMockAlert(): FraudAlert {
  const riskLevels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"] as const;
  const riskFactors = [
    "IMPOSSIBLE_TRAVEL",
    "UNUSUAL_AMOUNT",
    "HIGH_VELOCITY",
    "RAPID_TRANSACTIONS",
    "DISTANT_TRANSACTION",
    "PRIOR_FRAUD_HISTORY",
    "UNUSUAL_TIME",
  ];

  const riskLevel = riskLevels[Math.floor(Math.random() * riskLevels.length)];
  const numFactors = Math.floor(Math.random() * 3) + 1;

  return {
    transaction_id: `TXN_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    user_id: `USER_${Math.floor(Math.random() * 10000).toString().padStart(8, "0")}`,
    amount: parseFloat((Math.random() * 10000 + 100).toFixed(2)),
    currency: ["USD", "EUR", "GBP"][Math.floor(Math.random() * 3)],
    fraud_probability: parseFloat((Math.random() * 0.5 + 0.5).toFixed(4)),
    risk_level: riskLevel,
    risk_factors: riskFactors.sort(() => Math.random() - 0.5).slice(0, numFactors),
    timestamp: new Date().toISOString(),
    merchant_id: `MERCH_${Math.floor(Math.random() * 1000).toString().padStart(6, "0")}`,
    merchant_category: ["retail", "online", "travel", "gaming"][
      Math.floor(Math.random() * 4)
    ],
    location: {
      latitude: parseFloat((Math.random() * 180 - 90).toFixed(6)),
      longitude: parseFloat((Math.random() * 360 - 180).toFixed(6)),
    },
  };
}

// Process and broadcast alert
function processAlert(alert: FraudAlert) {
  // Update stats
  stats.total_transactions++;
  
  if (alert.risk_level === "HIGH" || alert.risk_level === "CRITICAL") {
    stats.fraud_detected++;
    stats.alerts_last_hour++;
    stats.total_amount_blocked += alert.amount;
  }
  
  stats.fraud_rate = stats.fraud_detected / stats.total_transactions;

  // Add to history
  alertHistory.push(alert);
  if (alertHistory.length > 100) {
    alertHistory.shift();
  }

  // Broadcast to all clients
  io.emit("alert:new", alert);

  // Broadcast to level-specific rooms
  io.to(`alerts:${alert.risk_level}`).emit("alert:level", alert);

  // Update stats for all
  io.emit("stats:update", stats);

  console.log(
    `[ALERT] ${alert.risk_level} - ${alert.transaction_id} - $${alert.amount}`
  );
}

// Simulate incoming alerts (demo mode)
let demoInterval: ReturnType<typeof setInterval> | null = null;

function startDemo() {
  if (demoInterval) return;

  console.log("[DEMO] Starting fraud alert simulation...");
  
  demoInterval = setInterval(() => {
    // 10% chance of fraud alert
    if (Math.random() < 0.1) {
      const alert = generateMockAlert();
      // Bias towards higher risk for demo
      if (Math.random() < 0.3) {
        alert.risk_level = "CRITICAL";
        alert.fraud_probability = 0.9 + Math.random() * 0.09;
      }
      processAlert(alert);
    }

    // Also simulate normal transactions
    stats.total_transactions++;
  }, 500);
}

// Start server
httpServer.listen(PORT, () => {
  console.log(`[WS] Sentinel-Fraud WebSocket service running on port ${PORT}`);
  
  // Start demo mode
  startDemo();
});

// API endpoint to receive alerts from inference service
// In production, this would connect to Kafka or receive webhooks
export function receiveAlert(alert: FraudAlert) {
  processAlert(alert);
}
