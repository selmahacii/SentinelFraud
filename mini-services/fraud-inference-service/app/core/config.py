"""
=============================================================================
SENTINEL-FRAUD: Configuration Module
=============================================================================
Centralized configuration using Pydantic Settings for type-safe env vars.
=============================================================================
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Service Information
    service_name: str = Field(default="sentinel-fraud-inference")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    cors_origins: str = Field(default="*")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    redis_pool_size: int = Field(default=50)
    redis_timeout: float = Field(default=5.0)
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:"
                f"{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic: str = Field(default="transactions")
    kafka_consumer_group: str = Field(default="sentinel-inference-group")
    kafka_auto_offset_reset: str = Field(default="latest")
    kafka_enable_auto_commit: bool = Field(default=False)
    kafka_max_poll_records: int = Field(default=500)
    
    # Model Configuration
    model_path: str = Field(default="/app/ml/artifacts/xgboost_fraud_model.json")
    model_version: str = Field(default="1.0.0")
    fallback_enabled: bool = Field(default=True)
    
    # Performance Configuration
    latency_target_ms: float = Field(default=50.0)
    max_concurrent_requests: int = Field(default=1000)
    request_timeout_seconds: float = Field(default=30.0)
    
    # Feature Store Configuration
    feature_ttl_seconds: int = Field(default=86400)  # 24 hours
    transaction_history_ttl: int = Field(default=604800)  # 7 days
    
    # Fraud Thresholds
    fraud_threshold: float = Field(default=0.85)
    high_risk_threshold: float = Field(default=0.70)
    medium_risk_threshold: float = Field(default=0.50)
    
    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(default=5)
    circuit_breaker_recovery_timeout: float = Field(default=30.0)
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=8000)
    metrics_prefix: str = Field(default="sentinel")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        """Ensure CORS origins are properly formatted."""
        return v


class KafkaConfig:
    """Kafka-specific configuration builder."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def consumer_config(self) -> dict:
        """Get Kafka consumer configuration."""
        return {
            "bootstrap.servers": self.settings.kafka_bootstrap_servers,
            "group.id": self.settings.kafka_consumer_group,
            "auto.offset.reset": self.settings.kafka_auto_offset_reset,
            "enable.auto.commit": self.settings.kafka_enable_auto_commit,
            "max.poll.records": self.settings.kafka_max_poll_records,
            "session.timeout.ms": 30000,
            "heartbeat.interval.ms": 10000,
            "max.poll.interval.ms": 300000,
        }
    
    @property
    def producer_config(self) -> dict:
        """Get Kafka producer configuration."""
        return {
            "bootstrap.servers": self.settings.kafka_bootstrap_servers,
            "client.id": f"{self.settings.service_name}-producer",
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 100,
            "compression.type": "lz4",
            "batch.size": 16384,
            "linger.ms": 5,
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_kafka_config() -> KafkaConfig:
    """Get Kafka configuration."""
    return KafkaConfig(get_settings())
