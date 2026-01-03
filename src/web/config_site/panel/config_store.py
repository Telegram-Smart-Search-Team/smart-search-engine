import json
import pathlib
import typing as tp

# django & related imports
from django.conf import settings

# redis & related imports
from redis import Redis


REDIS_CONFIG_KEY = settings.REDIS_CONFIG_KEY
REDIS_CONFIG_CHANNEL = settings.REDIS_CONFIG_CHANNEL

CONFIG_DIR = pathlib.Path(settings.PROJECT_DIR)
CONFIG_PATH = CONFIG_DIR / "tracking_config.json"


def default_sources_config() -> tp.Dict[str, bool]:
    return {
        "track_text": False,
        "track_image": False,
        "track_video": False,
        "track_audio": False,
        "track_files": False,
    }


def get_redis() -> Redis:
    return Redis.from_url(settings.REDIS_URL)


def _default_config() -> dict[str, tp.Any]:
    return {"chats": {}}


def load_config() -> dict[str, tp.Any]:
    r = get_redis()
    raw = r.get(REDIS_CONFIG_KEY)

    if raw:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            print(f"Could not read from redis '{REDIS_CONFIG_KEY}'")
            pass

    # fallback to file
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = _default_config()
    else:
        data = _default_config()

    # sync into redis
    r.set(REDIS_CONFIG_KEY, json.dumps(data, ensure_ascii=False))
    return data


def save_config_and_publish(config: dict[str, tp.Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, ensure_ascii=False)

    CONFIG_PATH.write_text(payload, encoding="utf-8")

    r = get_redis()
    r.set(REDIS_CONFIG_KEY, payload)
    r.publish(REDIS_CONFIG_CHANNEL, "updated")
