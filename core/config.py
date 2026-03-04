from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str = ""

    # WhatsApp Business API
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = "digital-ketu-verify"
    whatsapp_app_secret: str = ""

    # YouTube
    youtube_api_key: str = ""
    youtube_channel_id: str = ""

    # Server
    port: int = 8000
    auto_reply_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
