/**
 * =============================================================================
 * SENTINEL-FRAUD: Fraud Dashboard - Real-time Monitoring
 * =============================================================================
 * Next.js dashboard with WebSocket for real-time fraud alerts.
 * =============================================================================
 */

"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import {
  AlertTriangle,
  Shield,
  Activity,
  TrendingUp,
  Clock,
  DollarSign,
  CheckCircle2,
  XCircle,
  MapPin,
  User,
  CreditCard,
  Zap,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// =============================================================================
// TYPES
// =============================================================================

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

interface DashboardStats {
  total_transactions: number;
  fraud_detected: number;
  fraud_rate: number;
  avg_latency_ms: number;
  alerts_last_hour: number;
  total_amount_blocked: number;
}

// Dynamic import for socket.io-client (client-side only)
const io = dynamic(() => import("socket.io-client").then((mod) => mod.io), {
  ssr: false,
});

// =============================================================================
// RISK LEVEL CONFIG
// =============================================================================

const riskLevelConfig = {
  CRITICAL: {
    color: "bg-red-500",
    textColor: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500",
    label: "Critical",
  },
  HIGH: {
    color: "bg-orange-500",
    textColor: "text-orange-500",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500",
    label: "High",
  },
  MEDIUM: {
    color: "bg-yellow-500",
    textColor: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500",
    label: "Medium",
  },
  LOW: {
    color: "bg-green-500",
    textColor: "text-green-500",
    bgColor: "bg-green-500/10",
    borderColor: "border-green-500",
    label: "Low",
  },
};

// =============================================================================
// STAT CARD COMPONENT
// =============================================================================

interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
}

function StatCard({ title, value, description, icon, trend, trendValue }: StatCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}
        {trend && (
          <div className="flex items-center pt-1">
            {trend === "up" && <TrendingUp className="h-4 w-4 text-green-500 mr-1" />}
            {trend === "down" && <TrendingUp className="h-4 w-4 text-red-500 mr-1 rotate-180" />}
            <span className={`text-xs ${trend === "up" ? "text-green-500" : "text-red-500"}`}>
              {trendValue}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// ALERT ITEM COMPONENT
// =============================================================================

interface AlertItemProps {
  alert: FraudAlert;
  isExpanded?: boolean;
}

function AlertItem({ alert, isExpanded = false }: AlertItemProps) {
  const config = riskLevelConfig[alert.risk_level];
  const time = new Date(alert.timestamp).toLocaleTimeString();

  return (
    <div
      className={`p-4 rounded-lg border ${config.bgColor} ${config.borderColor} transition-all duration-300`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-full ${config.color}`}>
            <AlertTriangle className="h-4 w-4 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold">{alert.transaction_id.slice(0, 16)}...</span>
              <Badge variant="outline" className={config.textColor}>
                {config.label}
              </Badge>
            </div>
            <div className="text-sm text-muted-foreground flex items-center space-x-2 mt-1">
              <User className="h-3 w-3" />
              <span>{alert.user_id}</span>
              <span>•</span>
              <CreditCard className="h-3 w-3" />
              <span>{alert.merchant_category}</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="font-bold text-lg">
            {alert.currency} {alert.amount.toLocaleString()}
          </div>
          <div className="text-xs text-muted-foreground">{time}</div>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-border">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Fraud Probability:</span>
              <span className="ml-2 font-medium">
                {(alert.fraud_probability * 100).toFixed(2)}%
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Merchant:</span>
              <span className="ml-2 font-medium">{alert.merchant_id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Location:</span>
              <span className="ml-2 font-medium flex items-center">
                <MapPin className="h-3 w-3 mr-1" />
                {alert.location.latitude.toFixed(2)}, {alert.location.longitude.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Risk Factors:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {alert.risk_factors.map((factor, i) => (
                  <Badge key={i} variant="secondary" className="text-xs">
                    {factor.replace(/_/g, " ")}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// ALERTS TABLE COMPONENT
// =============================================================================

interface AlertsTableProps {
  alerts: FraudAlert[];
}

function AlertsTable({ alerts }: AlertsTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Transaction ID</TableHead>
          <TableHead>User</TableHead>
          <TableHead>Amount</TableHead>
          <TableHead>Risk Level</TableHead>
          <TableHead>Probability</TableHead>
          <TableHead>Time</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {alerts.map((alert) => (
          <TableRow key={alert.transaction_id}>
            <TableCell className="font-mono text-xs">
              {alert.transaction_id.slice(0, 16)}...
            </TableCell>
            <TableCell>{alert.user_id}</TableCell>
            <TableCell>
              {alert.currency} {alert.amount.toLocaleString()}
            </TableCell>
            <TableCell>
              <Badge
                variant="outline"
                className={riskLevelConfig[alert.risk_level].textColor}
              >
                {alert.risk_level}
              </Badge>
            </TableCell>
            <TableCell>{(alert.fraud_probability * 100).toFixed(2)}%</TableCell>
            <TableCell className="text-muted-foreground">
              {new Date(alert.timestamp).toLocaleTimeString()}
            </TableCell>
            <TableCell>
              <Button variant="ghost" size="sm">
                Review
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// =============================================================================
// MAIN DASHBOARD COMPONENT
// =============================================================================

export default function FraudDashboard() {
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState<DashboardStats>({
    total_transactions: 0,
    fraud_detected: 0,
    fraud_rate: 0,
    avg_latency_ms: 0,
    alerts_last_hour: 0,
    total_amount_blocked: 0,
  });
  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [selectedLevel, setSelectedLevel] = useState<string>("all");
  const socketRef = useRef<any>(null);

  // Connect to WebSocket
  useEffect(() => {
    const initSocket = async () => {
      try {
        const { io } = await import("socket.io-client");
        
        const newSocket = io("/?XTransformPort=3003", {
          transports: ["websocket"],
        });

        newSocket.on("connect", () => {
          console.log("[Dashboard] Connected to WebSocket");
          setConnected(true);
        });

        newSocket.on("disconnect", () => {
          console.log("[Dashboard] Disconnected from WebSocket");
          setConnected(false);
        });

        newSocket.on("stats:update", (newStats: DashboardStats) => {
          setStats(newStats);
        });

        newSocket.on("alert:new", (alert: FraudAlert) => {
          setAlerts((prev) => [alert, ...prev].slice(0, 100));
        });

        newSocket.on("alerts:history", (history: FraudAlert[]) => {
          setAlerts(history);
        });

        socketRef.current = newSocket;

        return () => {
          newSocket.close();
        };
      } catch (error) {
        console.error("[Dashboard] Failed to connect:", error);
      }
    };

    initSocket();

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  // Subscribe to alert levels
  useEffect(() => {
    if (socketRef.current && connected) {
      if (selectedLevel === "all") {
        socketRef.current.emit("alerts:subscribe", ["LOW", "MEDIUM", "HIGH", "CRITICAL"]);
      } else {
        socketRef.current.emit("alerts:unsubscribe", ["LOW", "MEDIUM", "HIGH", "CRITICAL"]);
        socketRef.current.emit("alerts:subscribe", [selectedLevel]);
      }
    }
  }, [connected, selectedLevel]);

  // Filter alerts by level
  const filteredAlerts =
    selectedLevel === "all"
      ? alerts
      : alerts.filter((a) => a.risk_level === selectedLevel);

  // Format currency
  const formatCurrency = (amount: number, currency: string = "USD") => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
    }).format(amount);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Shield className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">Sentinel-Fraud</h1>
                <p className="text-sm text-muted-foreground">
                  Real-time Fraud Detection Dashboard
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {connected ? (
                <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500">
                  <XCircle className="h-3 w-3 mr-1" />
                  Disconnected
                </Badge>
              )}
              <Button variant="outline" size="sm">
                <Activity className="h-4 w-4 mr-2" />
                Live Mode
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6 flex-1">
        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
          <StatCard
            title="Total Transactions"
            value={stats.total_transactions.toLocaleString()}
            description="Processed in current session"
            icon={<Activity className="h-4 w-4 text-muted-foreground" />}
          />
          <StatCard
            title="Fraud Detected"
            value={stats.fraud_detected}
            description="High & Critical risk alerts"
            icon={<AlertTriangle className="h-4 w-4 text-orange-500" />}
            trend="up"
            trendValue={`${(stats.fraud_rate * 100).toFixed(2)}% rate`}
          />
          <StatCard
            title="Amount Blocked"
            value={formatCurrency(stats.total_amount_blocked)}
            description="Total suspicious transactions"
            icon={<DollarSign className="h-4 w-4 text-muted-foreground" />}
          />
          <StatCard
            title="Avg Latency"
            value={`${stats.avg_latency_ms.toFixed(2)}ms`}
            description="Target: &lt;50ms"
            icon={<Clock className="h-4 w-4 text-muted-foreground" />}
          />
        </div>

        {/* Alerts Section */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Live Alerts Feed */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Live Alerts</CardTitle>
                  <CardDescription>Real-time fraud detection alerts</CardDescription>
                </div>
                <div className="flex flex-wrap gap-2">
                  {["all", "CRITICAL", "HIGH", "MEDIUM", "LOW"].map((level) => (
                    <Button
                      key={level}
                      variant={selectedLevel === level ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedLevel(level)}
                    >
                      {level === "all" ? "All" : level}
                    </Button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px] pr-4">
                {filteredAlerts.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground">
                    <Zap className="h-12 w-12 mb-4" />
                    <p>Waiting for alerts...</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {filteredAlerts.map((alert) => (
                      <AlertItem key={alert.transaction_id} alert={alert} />
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Sidebar Stats */}
          <div className="space-y-6">
            {/* Fraud Rate Progress */}
            <Card>
              <CardHeader>
                <CardTitle>Fraud Rate</CardTitle>
                <CardDescription>Current detection rate</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold">
                      {(stats.fraud_rate * 100).toFixed(3)}%
                    </span>
                    <Badge variant="outline">
                      {stats.fraud_rate < 0.001 ? "Normal" : "Elevated"}
                    </Badge>
                  </div>
                  <Progress value={stats.fraud_rate * 1000} max={1} className="h-2" />
                  <p className="text-xs text-muted-foreground">
                    Industry baseline: 0.1% - 0.5%
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Risk Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Risk Distribution</CardTitle>
                <CardDescription>Alert breakdown by risk level</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((level) => {
                    const count = alerts.filter((a) => a.risk_level === level).length;
                    const config = riskLevelConfig[level];
                    return (
                      <div key={level} className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <div className={`w-3 h-3 rounded-full ${config.color}`} />
                          <span className="text-sm">{config.label}</span>
                        </div>
                        <span className="font-medium">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Model Status */}
            <Card>
              <CardHeader>
                <CardTitle>Model Status</CardTitle>
                <CardDescription>Inference engine health</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">XGBoost Model</span>
                    <Badge variant="outline" className="text-green-500 border-green-500">
                      Active
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Redis Feature Store</span>
                    <Badge variant="outline" className="text-green-500 border-green-500">
                      Connected
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Kafka Consumer</span>
                    <Badge variant="outline" className="text-green-500 border-green-500">
                      Running
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Circuit Breaker</span>
                    <Badge variant="outline" className="text-green-500 border-green-500">
                      Closed
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button className="w-full" variant="outline">
                  Export Alerts
                </Button>
                <Button className="w-full" variant="outline">
                  View Reports
                </Button>
                <Button className="w-full" variant="outline">
                  Configure Rules
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Alerts Table */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Recent Transactions</CardTitle>
            <CardDescription>Detailed view of recent fraud alerts</CardDescription>
          </CardHeader>
          <CardContent>
            {filteredAlerts.length > 0 ? (
              <AlertsTable alerts={filteredAlerts.slice(0, 10)} />
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No alerts to display
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center space-x-4">
              <span>Sentinel-Fraud v1.0.0</span>
              <span>•</span>
              <span>Model: XGBoost v1.0.0</span>
              <span>•</span>
              <span>Latency Target: &lt;50ms</span>
            </div>
            <div className="flex items-center space-x-4">
              <span>
                Last updated: {new Date().toLocaleTimeString()}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
