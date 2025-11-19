import json
import typing as tp

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpRequest, HttpResponse

from .config_store import load_config, save_config_and_publish, default_sources_config
from .telegram_service import build_bootstrap_payload, fetch_batch_details


PAGE_SIZE = 20


def index(request: HttpRequest) -> HttpResponse:
    """
    Just render the shell; JS will call /api/bootstrap -> /api/batch.
    """
    return render(request, "panel/index.html", {"page_size": PAGE_SIZE})


async def api_bootstrap(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    config = load_config()
    payload = await build_bootstrap_payload(config)
    payload["page_size"] = PAGE_SIZE
    return JsonResponse(payload)


@csrf_exempt
async def api_batch(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    chat_ids: tp.List[str] = data.get("chat_ids") or []
    if not isinstance(chat_ids, list):
        return JsonResponse({"error": "chat_ids must be a list"}, status=400)

    config = load_config()
    details = await fetch_batch_details(chat_ids, config)
    return JsonResponse({"chats": details})


@csrf_exempt
def api_chat_config(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data: tp.Dict[str, tp.Any] = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    chat_key = data.get("chat_key")
    if not isinstance(chat_key, str):
        return JsonResponse({"error": "chat_key required"}, status=400)

    incoming_cfg = {
        "track_text": bool(data.get("track_text", False)),
        "track_image": bool(data.get("track_image", False)),
        "track_video": bool(data.get("track_video", False)),
        "track_audio": bool(data.get("track_audio", False)),
        "track_files": bool(data.get("track_files", False)),
    }

    config = load_config()
    current_cfg = config.get("chats", {}).get(chat_key, default_sources_config())

    if incoming_cfg == current_cfg:
        # No change, no Redis publish
        return JsonResponse({"ok": True, "changed": False})

    config.setdefault("chats", {})[chat_key] = incoming_cfg
    save_config_and_publish(config)
    return JsonResponse({"ok": True, "changed": True})
