from __future__ import annotations

"""AWS Lambda adapters for API Gateway HTTP API payload format 2.0.

Prediction and reveal are separate entry points on purpose.  The prediction
entry point prepares only truth-free runtime assets; the reveal entry point
loads only the frozen reference parquet and never initialises LightGBM.
"""

import base64
import binascii
import json
import logging
import os
import threading
from http import HTTPStatus
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs

from backend.s3_assets import AssetError, prepare_prediction_assets, prepare_reveal_assets


LOGGER = logging.getLogger(__name__)
MAX_PAYLOAD_BYTES = 64 * 1024
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


class RequestError(ValueError):
    def __init__(self, message: str, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


_prediction_service: Any | None = None
_reveal_service: Any | None = None
_prediction_service_factory: Callable[[], Any] | None = None
_reveal_service_factory: Callable[[], Any] | None = None
_prediction_lock = threading.Lock()
_reveal_lock = threading.Lock()


def _default_prediction_service_factory() -> Any:
    # Assets must be ready and exported to environment variables before
    # importing/constructing the service that reads those paths.
    prepare_prediction_assets()
    from backend.api import DemoService

    return DemoService()


def _default_reveal_service_factory() -> Any:
    prepare_reveal_assets()
    # This module has no dependency on backend.inference or LightGBM.
    from backend.reveal import RevealService

    return RevealService()


def _get_prediction_service() -> Any:
    global _prediction_service
    if _prediction_service is None:
        with _prediction_lock:
            if _prediction_service is None:
                factory = _prediction_service_factory or _default_prediction_service_factory
                _prediction_service = factory()
    return _prediction_service


def _get_reveal_service() -> Any:
    global _reveal_service
    if _reveal_service is None:
        with _reveal_lock:
            if _reveal_service is None:
                factory = _reveal_service_factory or _default_reveal_service_factory
                _reveal_service = factory()
    return _reveal_service


def _reset_services_for_tests(
    *,
    prediction_factory: Callable[[], Any] | None = None,
    reveal_factory: Callable[[], Any] | None = None,
) -> None:
    """Reset warm caches; intended for deterministic tests, not requests."""
    global _prediction_service, _reveal_service
    global _prediction_service_factory, _reveal_service_factory
    _prediction_service = None
    _reveal_service = None
    _prediction_service_factory = prediction_factory
    _reveal_service_factory = reveal_factory


def _headers(event: Mapping[str, Any]) -> dict[str, str]:
    supplied = event.get("headers") or {}
    if not isinstance(supplied, Mapping):
        return {}
    return {str(name).lower(): str(value) for name, value in supplied.items()}


def _allowed_origins() -> set[str]:
    return {
        origin.strip().rstrip("/")
        for origin in os.environ.get("UBIKE_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
        if origin.strip()
    }


def _response_headers(event: Mapping[str, Any]) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "X-Content-Type-Options": "nosniff",
    }
    origin = _headers(event).get("origin", "").rstrip("/")
    if origin and origin in _allowed_origins():
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
    return headers


def _json_response(
    event: Mapping[str, Any],
    payload: Mapping[str, Any],
    status: int = HTTPStatus.OK,
) -> dict[str, Any]:
    return {
        "statusCode": int(status),
        "headers": _response_headers(event),
        "body": json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "isBase64Encoded": False,
    }


def _empty_response(event: Mapping[str, Any], status: int) -> dict[str, Any]:
    headers = _response_headers(event)
    headers.pop("Content-Type", None)
    return {
        "statusCode": int(status),
        "headers": headers,
        "body": "",
        "isBase64Encoded": False,
    }


def _error_response(
    event: Mapping[str, Any], status: int, code: str, message: str
) -> dict[str, Any]:
    return _json_response(
        event,
        {"error": message, "code": code},
        status,
    )


def _method_and_path(event: Mapping[str, Any]) -> tuple[str, str]:
    version = event.get("version")
    if version not in {None, "2.0"}:
        raise RequestError(
            "只支援API Gateway HTTP API payload format 2.0",
            HTTPStatus.BAD_REQUEST,
            "UNSUPPORTED_EVENT",
        )
    context = event.get("requestContext") or {}
    http = context.get("http") if isinstance(context, Mapping) else {}
    http = http if isinstance(http, Mapping) else {}
    method = str(http.get("method") or event.get("httpMethod") or "").upper()
    path = str(event.get("rawPath") or event.get("path") or "/")
    if not method:
        raise RequestError(
            "無法判斷HTTP方法",
            HTTPStatus.BAD_REQUEST,
            "INVALID_EVENT",
        )
    # Custom domains and some test fixtures can include a trailing slash.
    if len(path) > 1:
        path = path.rstrip("/")
    return method, path


def _read_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    headers = _headers(event)
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise RequestError(
            "Content-Type 必須是 application/json",
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "UNSUPPORTED_MEDIA_TYPE",
        )
    declared_length = headers.get("content-length")
    if declared_length:
        try:
            parsed_length = int(declared_length)
        except ValueError as error:
            raise RequestError(
                "Content-Length格式錯誤",
                HTTPStatus.BAD_REQUEST,
                "INVALID_REQUEST",
            ) from error
        if parsed_length < 0:
            raise RequestError(
                "Content-Length格式錯誤",
                HTTPStatus.BAD_REQUEST,
                "INVALID_REQUEST",
            )
        if parsed_length > MAX_PAYLOAD_BYTES:
            raise RequestError(
                "JSON請求不得超過64KB",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "PAYLOAD_TOO_LARGE",
            )

    raw_body = event.get("body")
    if raw_body is None or raw_body == "":
        raise RequestError(
            "請提供有效的JSON請求",
            HTTPStatus.BAD_REQUEST,
            "INVALID_JSON",
        )
    if not isinstance(raw_body, str):
        raise RequestError(
            "JSON body必須是字串",
            HTTPStatus.BAD_REQUEST,
            "INVALID_JSON",
        )

    try:
        if bool(event.get("isBase64Encoded")):
            raw_bytes = base64.b64decode(raw_body, validate=True)
        else:
            raw_bytes = raw_body.encode("utf-8")
    except (UnicodeError, binascii.Error, ValueError) as error:
        raise RequestError(
            "JSON請求編碼錯誤",
            HTTPStatus.BAD_REQUEST,
            "INVALID_JSON",
        ) from error

    if len(raw_bytes) > MAX_PAYLOAD_BYTES:
        raise RequestError(
            "JSON請求不得超過64KB",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "PAYLOAD_TOO_LARGE",
        )
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RequestError(
            "請提供有效的JSON請求",
            HTTPStatus.BAD_REQUEST,
            "INVALID_JSON",
        ) from error
    if not isinstance(payload, dict):
        raise RequestError(
            "JSON最外層必須是物件",
            HTTPStatus.BAD_REQUEST,
            "INVALID_JSON",
        )
    return payload


def _query_parameter(event: Mapping[str, Any], name: str) -> str | None:
    values = event.get("queryStringParameters") or {}
    if isinstance(values, Mapping) and values.get(name) is not None:
        return str(values[name])
    raw_query = str(event.get("rawQueryString") or "")
    return parse_qs(raw_query).get(name, [None])[0]


def _prediction_health(service: Any) -> dict[str, Any]:
    if hasattr(service, "health"):
        return service.health()
    eligible = getattr(service, "eligible", ())
    return {
        "status": "ok",
        "service": "prediction",
        "model": "lgbm_full",
        "rows": int(len(eligible)),
    }


def _dispatch_prediction(
    event: Mapping[str, Any], method: str, path: str
) -> dict[str, Any]:
    if method == "OPTIONS":
        return _empty_response(event, HTTPStatus.NO_CONTENT)
    if method == "GET" and path == "/api/health":
        return _json_response(event, _prediction_health(_get_prediction_service()))
    if method == "GET" and path == "/api/options":
        requested = _query_parameter(event, "decision_time")
        mode = _query_parameter(event, "mode") or "empty"
        return _json_response(event, _get_prediction_service().options(requested, mode=mode))
    if method == "POST" and path == "/api/predict":
        payload = _read_payload(event)
        return _json_response(event, _get_prediction_service().predict(payload))
    if method == "POST" and path == "/api/explain":
        payload = _read_payload(event)
        service = _get_prediction_service()
        if not hasattr(service, "explain"):
            raise RequestError(
                "模型解釋服務尚未啟用",
                HTTPStatus.NOT_IMPLEMENTED,
                "EXPLAIN_NOT_CONFIGURED",
            )
        return _json_response(event, service.explain(payload))
    raise RequestError("找不到此端點", HTTPStatus.NOT_FOUND, "NOT_FOUND")


def _dispatch_reveal(
    event: Mapping[str, Any], method: str, path: str
) -> dict[str, Any]:
    if method == "OPTIONS":
        return _empty_response(event, HTTPStatus.NO_CONTENT)
    if method == "POST" and path == "/api/reveal":
        payload = _read_payload(event)
        return _json_response(event, _get_reveal_service().reveal(payload))
    raise RequestError("找不到此端點", HTTPStatus.NOT_FOUND, "NOT_FOUND")


def _handle(
    event: Mapping[str, Any] | None,
    context: Any,
    dispatcher: Callable[[Mapping[str, Any], str, str], dict[str, Any]],
) -> dict[str, Any]:
    del context  # Request IDs remain available to API Gateway/CloudWatch logs.
    safe_event: Mapping[str, Any] = event if isinstance(event, Mapping) else {}
    try:
        method, path = _method_and_path(safe_event)
        return dispatcher(safe_event, method, path)
    except RequestError as error:
        return _error_response(safe_event, error.status, error.code, str(error))
    except (ValueError, TypeError) as error:
        return _error_response(
            safe_event,
            HTTPStatus.BAD_REQUEST,
            "INVALID_REQUEST",
            str(error),
        )
    except AssetError:
        # Avoid leaking bucket names, keys, or checksums to a public client.
        LOGGER.exception("Frozen AWS asset preparation failed")
        return _error_response(
            safe_event,
            HTTPStatus.SERVICE_UNAVAILABLE,
            "ASSET_UNAVAILABLE",
            "凍結資產無法驗證，服務暫時無法使用",
        )
    except Exception:  # pragma: no cover - last-resort Lambda boundary
        LOGGER.exception("Unhandled Lambda request failure")
        return _error_response(
            safe_event,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "服務暫時無法處理",
        )


def prediction_handler(event: Mapping[str, Any] | None, context: Any) -> dict[str, Any]:
    """Handler for health/options/predict/explain routes (never reveal)."""
    return _handle(event, context, _dispatch_prediction)


def reveal_handler(event: Mapping[str, Any] | None, context: Any) -> dict[str, Any]:
    """Handler for the isolated truth reveal route."""
    return _handle(event, context, _dispatch_reveal)


# Conventional single-handler name defaults to the truth-free prediction API.
lambda_handler = prediction_handler


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "lambda_handler",
    "prediction_handler",
    "reveal_handler",
]
