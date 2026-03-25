"""
=============================================================================
SENTINEL-FRAUD: Transaction Producer
=============================================================================
Simulated Kafka producer for generating realistic transaction data.
=============================================================================
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("producer")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProducerConfig:
    """Producer configuration."""
    bootstrap_servers: str = "localhost:9092"
    topic: str = "transactions"
    transactions_per_second: int = 100
    fraud_rate: float = 0.001  # 0.1% fraud
    batch_size: int = 100
    simulation_mode: str = "realistic"  # realistic, stress, burst


# =============================================================================
# TRANSACTION GENERATOR
# =============================================================================

class MerchantCategory(str, Enum):
    """Merchant categories."""
    RETAIL = "retail"
    FOOD_DINING = "food_dining"
    GAS_STATION = "gas_station"
    GROCERY = "grocery"
    ONLINE = "online"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    GAMING = "gaming"


class TransactionChannel(str, Enum):
    """Transaction channels."""
    ONLINE = "online"
    OFFLINE = "offline"
    MOBILE = "mobile"
    API = "api"


@dataclass
class GeoLocation:
    """Geographic location."""
    latitude: float
    longitude: float
    city: str
    country: str


# Realistic locations
LOCATIONS = [
    GeoLocation(40.7128, -74.0060, "New York", "USA"),
    GeoLocation(51.5074, -0.1278, "London", "UK"),
    GeoLocation(48.8566, 2.3522, "Paris", "France"),
    GeoLocation(35.6762, 139.6503, "Tokyo", "Japan"),
    GeoLocation(-33.8688, 151.2093, "Sydney", "Australia"),
    GeoLocation(1.3521, 103.8198, "Singapore", "Singapore"),
    GeoLocation(55.7558, 37.6173, "Moscow", "Russia"),
    GeoLocation(19.4326, -99.1332, "Mexico City", "Mexico"),
    GeoLocation(-23.5505, -46.6333, "São Paulo", "Brazil"),
    GeoLocation(37.5665, 126.9780, "Seoul", "South Korea"),
]


@dataclass
class Transaction:
    """Generated transaction."""
    transaction_id: str
    user_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    timestamp: str
    latitude: float
    longitude: float
    city: str
    country: str
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    channel: str = "online"
    is_fraud: bool = False
    fraud_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Kafka."""
        return asdict(self)


class TransactionGenerator:
    """
    Generate realistic transaction data.
    
    Supports multiple fraud patterns:
    - Impossible travel
    - High velocity
    - Amount anomaly
    - Unusual time
    - New device/location
    """

    def __init__(self, config: ProducerConfig):
        self.config = config
        self._users: dict[str, dict[str, Any]] = {}
        self._merchants: dict[str, str] = {}
        self._fraud_types = [
            "impossible_travel",
            "velocity_attack",
            "amount_anomaly",
            "stolen_card",
            "account_takeover",
        ]
        
        # Initialize merchant pool
        for i in range(1000):
            merchant_id = f"MERCH_{i:06d}"
            self._merchants[merchant_id] = random.choice(list(MerchantCategory)).value
        
        # Initialize user pool
        for i in range(10000):
            user_id = f"USER_{i:08d}"
            self._users[user_id] = {
                "avg_amount": random.uniform(50, 500),
                "std_amount": random.uniform(20, 100),
                "home_location": random.choice(LOCATIONS),
                "usual_hours": random.sample(range(6, 23), random.randint(8, 16)),
                "preferred_channels": random.sample(
                    list(TransactionChannel), 
                    random.randint(1, 3)
                ),
                "txn_count": random.randint(0, 1000),
            }

    def generate_normal_transaction(self) -> Transaction:
        """Generate a normal, legitimate transaction."""
        user_id = random.choice(list(self._users.keys()))
        user = self._users[user_id]
        
        # Select location (usually near home)
        if random.random() < 0.9:
            home = user["home_location"]
            location = GeoLocation(
                latitude=home.latitude + random.uniform(-0.5, 0.5),
                longitude=home.longitude + random.uniform(-0.5, 0.5),
                city=home.city,
                country=home.country
            )
        else:
            location = random.choice(LOCATIONS)
        
        # Generate amount based on user profile
        amount = max(0.01, random.gauss(user["avg_amount"], user["std_amount"]))
        
        # Select merchant
        merchant_id = random.choice(list(self._merchants.keys()))
        merchant_category = self._merchants[merchant_id]
        
        # Select channel
        channel = random.choice(user["preferred_channels"]).value
        
        # Generate timestamp (usually during user's usual hours)
        hour = random.choice(user["usual_hours"])
        timestamp = datetime.now(timezone.utc).replace(
            hour=hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user_id,
            amount=round(amount, 2),
            currency=random.choice(["USD", "EUR", "GBP", "JPY"]),
            merchant_id=merchant_id,
            merchant_category=merchant_category,
            timestamp=timestamp.isoformat(),
            latitude=round(location.latitude + random.uniform(-0.01, 0.01), 6),
            longitude=round(location.longitude + random.uniform(-0.01, 0.01), 6),
            city=location.city,
            country=location.country,
            device_id=f"DEV_{random.randint(1, 100):04d}",
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            channel=channel,
            is_fraud=False
        )

    def generate_fraud_transaction(self) -> Transaction:
        """Generate a fraudulent transaction with specific patterns."""
        fraud_type = random.choice(self._fraud_types)
        
        if fraud_type == "impossible_travel":
            return self._generate_impossible_travel_fraud()
        elif fraud_type == "velocity_attack":
            return self._generate_velocity_fraud()
        elif fraud_type == "amount_anomaly":
            return self._generate_amount_anomaly_fraud()
        elif fraud_type == "stolen_card":
            return self._generate_stolen_card_fraud()
        else:
            return self._generate_account_takeover_fraud()

    def _generate_impossible_travel_fraud(self) -> Transaction:
        """Generate fraud with impossible travel pattern."""
        user_id = random.choice(list(self._users.keys()))
        user = self._users[user_id]
        
        # Location far from user's home
        distant_location = random.choice([
            loc for loc in LOCATIONS 
            if loc.country != user["home_location"].country
        ])
        
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user_id,
            amount=round(random.uniform(500, 5000), 2),
            currency="USD",
            merchant_id=random.choice(list(self._merchants.keys())),
            merchant_category=random.choice(["online", "travel", "gaming"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            latitude=distant_location.latitude,
            longitude=distant_location.longitude,
            city=distant_location.city,
            country=distant_location.country,
            device_id=f"DEV_NEW_{random.randint(1, 9999):04d}",
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            channel="online",
            is_fraud=True,
            fraud_type="impossible_travel"
        )

    def _generate_velocity_fraud(self) -> Transaction:
        """Generate high-velocity fraud pattern."""
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=random.choice(list(self._users.keys())),
            amount=round(random.uniform(100, 500), 2),
            currency="USD",
            merchant_id=random.choice(list(self._merchants.keys())),
            merchant_category=random.choice(["online", "gaming"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            latitude=random.choice(LOCATIONS).latitude,
            longitude=random.choice(LOCATIONS).longitude,
            city=random.choice(LOCATIONS).city,
            country=random.choice(LOCATIONS).country,
            device_id=f"DEV_{random.randint(1, 100):04d}",
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            channel="online",
            is_fraud=True,
            fraud_type="velocity_attack"
        )

    def _generate_amount_anomaly_fraud(self) -> Transaction:
        """Generate unusual high amount fraud."""
        user_id = random.choice(list(self._users.keys()))
        user = self._users[user_id]
        
        # Amount significantly higher than user's average
        anomalous_amount = user["avg_amount"] * random.uniform(5, 20)
        
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user_id,
            amount=round(anomalous_amount, 2),
            currency="USD",
            merchant_id=random.choice(list(self._merchants.keys())),
            merchant_category=random.choice(["online", "travel"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            latitude=user["home_location"].latitude,
            longitude=user["home_location"].longitude,
            city=user["home_location"].city,
            country=user["home_location"].country,
            device_id=f"DEV_{random.randint(1, 100):04d}",
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            channel=random.choice(["online", "mobile"]),
            is_fraud=True,
            fraud_type="amount_anomaly"
        )

    def _generate_stolen_card_fraud(self) -> Transaction:
        """Generate stolen card fraud pattern."""
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=random.choice(list(self._users.keys())),
            amount=round(random.uniform(1000, 10000), 2),
            currency="USD",
            merchant_id=random.choice(list(self._merchants.keys())),
            merchant_category=random.choice(["online", "financial"]),
            timestamp=datetime.now(timezone.utc).replace(
                hour=random.randint(1, 4)  # Unusual hours
            ).isoformat(),
            latitude=random.choice(LOCATIONS).latitude,
            longitude=random.choice(LOCATIONS).longitude,
            city=random.choice(LOCATIONS).city,
            country=random.choice(LOCATIONS).country,
            device_id=f"DEV_NEW_{random.randint(1, 9999):04d}",
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            channel="online",
            is_fraud=True,
            fraud_type="stolen_card"
        )

    def _generate_account_takeover_fraud(self) -> Transaction:
        """Generate account takeover fraud pattern."""
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=random.choice(list(self._users.keys())),
            amount=round(random.uniform(500, 2000), 2),
            currency="USD",
            merchant_id=random.choice(list(self._merchants.keys())),
            merchant_category="online",
            timestamp=datetime.now(timezone.utc).isoformat(),
            latitude=random.choice(LOCATIONS).latitude,
            longitude=random.choice(LOCATIONS).longitude,
            city=random.choice(LOCATIONS).city,
            country=random.choice(LOCATIONS).country,
            device_id=f"DEV_NEW_{random.randint(1, 9999):04d}",
            ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            channel="api",
            is_fraud=True,
            fraud_type="account_takeover"
        )

    def generate_transaction(self) -> Transaction:
        """Generate a transaction (normal or fraudulent based on fraud rate)."""
        if random.random() < self.config.fraud_rate:
            return self.generate_fraud_transaction()
        return self.generate_normal_transaction()


# =============================================================================
# KAFKA PRODUCER
# =============================================================================

class TransactionProducer:
    """Async Kafka producer for transactions."""

    def __init__(self, config: ProducerConfig):
        self.config = config
        self.generator = TransactionGenerator(config)
        self._producer: Optional[AIOKafkaProducer] = None
        self._running = False
        self._stats = {
            "total_sent": 0,
            "fraud_sent": 0,
            "errors": 0,
            "start_time": None,
        }

    async def start(self) -> None:
        """Start the producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            compression_type="lz4",
            acks="all",
            max_batch_size=16384,
            linger_ms=5,
        )
        await self._producer.start()
        self._running = True
        self._stats["start_time"] = time.time()
        logger.info(f"Producer started, targeting {self.config.transactions_per_second} TPS")

    async def stop(self) -> None:
        """Stop the producer."""
        self._running = False
        if self._producer:
            await self._producer.stop()
        logger.info(f"Producer stopped. Stats: {self._stats}")

    async def send_transaction(self, transaction: Transaction) -> bool:
        """Send a single transaction to Kafka."""
        try:
            await self._producer.send_and_wait(
                self.config.topic,
                transaction.to_dict(),
                key=transaction.user_id.encode("utf-8")
            )
            
            self._stats["total_sent"] += 1
            if transaction.is_fraud:
                self._stats["fraud_sent"] += 1
            
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to send transaction: {e}")
            return False

    async def run(self) -> None:
        """Main production loop."""
        await self.start()
        
        interval = 1.0 / self.config.transactions_per_second
        
        try:
            while self._running:
                start = time.perf_counter()
                
                # Generate and send transaction
                transaction = self.generator.generate_transaction()
                await self.send_transaction(transaction)
                
                # Log periodically
                if self._stats["total_sent"] % 1000 == 0:
                    elapsed = time.time() - self._stats["start_time"]
                    tps = self._stats["total_sent"] / elapsed
                    logger.info(
                        f"Sent {self._stats['total_sent']} transactions "
                        f"({tps:.1f} TPS, {self._stats['fraud_sent']} fraud)"
                    )
                
                # Rate limiting
                elapsed = time.perf_counter() - start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except asyncio.CancelledError:
            logger.info("Producer cancelled")
        finally:
            await self.stop()

    async def run_burst(self, burst_size: int = 1000) -> None:
        """Send a burst of transactions (for stress testing)."""
        await self.start()
        
        logger.info(f"Sending burst of {burst_size} transactions...")
        
        tasks = []
        for _ in range(burst_size):
            transaction = self.generator.generate_transaction()
            tasks.append(self.send_transaction(transaction))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.info(f"Burst complete: {success_count}/{burst_size} sent")
        
        await self.stop()


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    import os
    
    config = ProducerConfig(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        topic=os.getenv("KAFKA_TOPIC", "transactions"),
        transactions_per_second=int(os.getenv("TRANSACTIONS_PER_SECOND", "100")),
        fraud_rate=float(os.getenv("FRAUD_RATE", "0.001")),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        simulation_mode=os.getenv("SIMULATION_MODE", "realistic"),
    )
    
    producer = TransactionProducer(config)
    
    try:
        await producer.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
