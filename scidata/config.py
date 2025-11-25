from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # openai
    openai_api_key: str
    openai_project_id: str
    openai_organization_id: str
    
    # opensearch
    opensearch_host: str
    opensearch_port: int
    opensearch_username: str
    opensearch_password: str
    
    model_config = SettingsConfigDict(env_file=".env")
    

settings = Settings()
