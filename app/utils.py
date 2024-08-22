import os


def get_env_or_throw(key: str, default: str = None) -> str:
    value = os.getenv(key, default)
    if value is None or value.strip() == "":
        raise ValueError(f"Environment variable '{key}' is not set or is empty")
    return value
