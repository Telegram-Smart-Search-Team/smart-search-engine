import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parents[2]
load_dotenv(dotenv_path=str(PROJECT_DIR / ".env"))


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
DEBUG = True
ALLOWED_HOSTS = ["*"]


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "panel",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "telegram_config_panel.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "panel" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "telegram_config_panel.wsgi.application"
ASGI_APPLICATION = "telegram_config_panel.asgi.application"

# default sqlite for this panel
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "static"
STATICFILES_DIRS = [BASE_DIR / "panel" / "static"]

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# redis
REDIS_URL = os.getenv("REDIS_URL")
REDIS_CONFIG_KEY = os.getenv("REDIS_CONFIG_KEY")
REDIS_CONFIG_CHANNEL = os.getenv("REDIS_CONFIG_CHANNEL")


# telegram client for django (separate session)
DJANGO_CLIENT_APP_API_ID = int(os.getenv("CLIENT_APP_API_ID"))
DJANGO_CLIENT_APP_API_HASH = os.getenv("CLIENT_APP_API_HASH")
DJANGO_SESSION_PATH = str((PROJECT_DIR / "config_panel_session.session").resolve())

TELEGRAM_MEDIA_SUBDIR = "telegram_media"  # under MEDIA_ROOT


CLIENT_USER_ID = int(os.getenv("CLIENT_USER_ID"))
BOT_USER_ID = int(os.getenv("BOT_TOKEN").split(":", 1)[0])
