from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    deepgram_api_key: str = ""
    twilio_auth_token: str = ""
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql://emma:emma@localhost:5432/emma"


settings = Settings()
