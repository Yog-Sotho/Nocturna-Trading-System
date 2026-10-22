"""
NOCTURNA Trading System - The OpenAPI description of the API.

Each route declares its contract with ``@documented(...)``, directly under its
``route`` decorator: a summary, who may call it, the request body and query
parameters, and the ``data`` of each response as pydantic models
(``src/routes/schemas.py``). ``build_spec`` walks the URL map and turns the
declarations into an OpenAPI 3.1 document, served at
``/api/v1/openapi.json`` and written by ``flask openapi``.

The contract tests (``src/tests/test_openapi.py``) keep it true: every route
is declared, the declared access holds (401 without credentials, 403 for a
caller without the role), and real responses match the declared models.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from flask import Flask
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

Access = Literal["public", "user", "trader", "admin", "metrics"]

#: Who may call an operation, as the OpenAPI document says it
ACCESS_TEXT = {
    "public": "No credentials.",
    "user": "Any signed-in user: a bearer token, or an API key.",
    "trader": "The trader role: a bearer token of a trader or admin, or an API key with the trade scope.",
    "admin": "An admin's bearer token (API keys never carry admin rights).",
    "metrics": "Authorization: Bearer <METRICS_TOKEN>.",
}
SECURITY = {
    "public": [],
    "user": [{"bearerAuth": []}, {"apiKey": []}],
    "trader": [{"bearerAuth": []}, {"apiKey": []}],
    "admin": [{"bearerAuth": []}],
    "metrics": [{"metricsToken": []}],
}
#: Statuses an operation may answer whatever it declares: by its access, its rate
#: limit and the app's error handler. Their bodies are always the envelope.
SIGNED_IN = {401: "Missing, expired or revoked credentials"}
ROLE = {403: "The caller lacks the role"}
RATE_LIMITED = {429: "Rate limit exceeded"}
ERROR = {500: "Unexpected error"}

#: A description; (description, model of ``data``); or (description, model, data always present)
Response = str | tuple[str, type[BaseModel] | None] | tuple[str, type[BaseModel] | None, bool]


@dataclass(frozen=True)
class Operation:
    summary: str
    tag: str
    access: Access
    responses: Mapping[int, Response]
    description: str = ""
    body: type[BaseModel] | None = None
    body_required: bool = True
    query: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    #: False: the declared bodies are the model itself, not the {success, data, ...} envelope
    envelope: bool = True
    #: A non-JSON success body (Prometheus text)
    media_type: str | None = None
    #: False for the endpoints the rate limiter exempts
    rate_limited: bool = True

    def implied(self) -> dict[int, str]:
        implied = dict(ERROR)
        if self.access in ("user", "trader", "admin"):
            implied.update(SIGNED_IN)
        if self.access in ("trader", "admin"):
            implied.update(ROLE)
        if self.rate_limited:
            implied.update(RATE_LIMITED)
        return implied

    def statuses(self) -> list[int]:
        return sorted(set(self.responses) | set(self.implied()))

    def response(self, status: int) -> tuple[str, type[BaseModel] | None, bool] | None:
        """(description, model of the body or its data, data always present); None if not declared."""
        if status in self.responses:
            declared = self.responses[status]
            if isinstance(declared, str):
                return declared, None, False
            return (*declared, True) if len(declared) == 2 else declared
        if status in self.implied():
            return self.implied()[status], None, False
        return None


def documented(summary: str, *, tag: str, access: Access, responses: Mapping[int, Response],
               **details: Any) -> Callable[[Callable], Callable]:
    """
    Declare a route's contract. ``responses`` maps each status to its description,
    or to (description, model of ``data``). 500, 429 (unless ``rate_limited=False``),
    401 for signed-in access and 403 for trader or admin access are implied.
    """
    operation = Operation(summary, tag, access, responses, **details)

    def declare(view: Callable) -> Callable:
        view.__openapi__ = operation
        return view

    return declare


def operation_of(view: Callable) -> Operation | None:
    """The declaration on a view, through any wrapping decorators (functools.wraps)."""
    while view is not None:
        found = getattr(view, "__openapi__", None)
        if found is not None:
            return found
        view = getattr(view, "__wrapped__", None)
    return None


#: Endpoints that are not part of the API: the dashboard's files
UNDOCUMENTED = {"static", "serve"}
_ARGUMENT = re.compile(r"<(?:(\w+)(?:\([^)]*\))?:)?(\w+)>")


def documented_routes(app: Flask):
    """(rule, method, operation) for every API route; operation is None when undeclared."""
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        if rule.endpoint in UNDOCUMENTED:
            continue
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            yield rule, method, operation_of(app.view_functions[rule.endpoint])


def _path(rule: str) -> tuple[str, list[dict[str, Any]]]:
    parameters = []
    for converter, name in _ARGUMENT.findall(rule):
        schema = {"type": "integer"} if converter == "int" else {"type": "string"}
        parameters.append({"name": name, "in": "path", "required": True, "schema": schema})
    return _ARGUMENT.sub(lambda m: "{" + m.group(2) + "}", rule), parameters


def _envelope(data: dict[str, Any] | None, required: bool) -> dict[str, Any]:
    if data is None:
        return {"$ref": "#/components/schemas/Envelope"}
    return {"allOf": [{"$ref": "#/components/schemas/Envelope"},
                      {"type": "object", "properties": {"data": data}, "required": ["data"] if required else []}]}


def build_spec(app: Flask, version: str = "v1") -> dict[str, Any]:
    """
    Raises:
        LookupError: If a route has no ``@documented`` declaration.
    """
    routes = list(documented_routes(app))
    missing = [f"{method} {rule.rule}" for rule, method, op in routes if op is None]
    if missing:
        raise LookupError(f"routes without @documented: {missing}")

    models: set[type[BaseModel]] = set()
    for _, _, op in routes:
        if op.body is not None:
            models.add(op.body)
        models.update(m for _, m, _ in (op.response(s) for s in op.responses) if m is not None)
    ordered = sorted(models, key=lambda m: m.__name__)
    keys, definitions = models_json_schema([(m, "validation") for m in ordered],
                                           ref_template="#/components/schemas/{model}")
    schemas = dict(definitions.get("$defs", {}))
    refs = {m: keys[(m, "validation")] for m in ordered}
    schemas["Envelope"] = {
        "type": "object",
        "description": "Every JSON response of the API (but /health and /ready).",
        "properties": {
            "success": {"type": "boolean"},
            "timestamp": {"type": "string", "format": "date-time"},
            "request_id": {"type": ["string", "null"], "description": "Also in the X-Request-ID header"},
            "data": {},
            "error": {"type": "string", "description": "Why it failed (success false)"},
            "message": {"type": "string"},
        },
        "required": ["success"],
    }

    paths: dict[str, dict[str, Any]] = {}
    for rule, method, op in routes:
        path, parameters = _path(rule.rule)
        parameters += [{"name": name, "in": "query", "required": False, "schema": {"type": kind},
                        "description": text} for name, (kind, text) in op.query.items()]
        responses = {}
        for status in op.statuses():
            text, model, required = op.response(status)
            schema = refs[model] if model is not None else None
            if op.media_type and 200 <= status < 300:
                content = {op.media_type: {"schema": {"type": "string"}}}
            elif not op.envelope and status in op.responses:
                content = {"application/json": {"schema": schema or {"type": "object"}}}
            else:
                content = {"application/json": {"schema": _envelope(schema, required)}}
            responses[str(status)] = {"description": text, "content": content}
        operation: dict[str, Any] = {
            "operationId": f"{rule.endpoint.replace('.', '_')}_{method.lower()}",
            "summary": op.summary,
            "description": "\n\n".join(t for t in (op.description, f"Access: {ACCESS_TEXT[op.access]}") if t),
            "tags": [op.tag],
            "security": SECURITY[op.access],
            "x-access": op.access,
            "responses": responses,
        }
        if parameters:
            operation["parameters"] = parameters
        if op.body is not None:
            operation["requestBody"] = {"required": op.body_required,
                                        "content": {"application/json": {"schema": refs[op.body]}}}
        paths.setdefault(path, {})[method.lower()] = operation

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "NOCTURNA Trading System API",
            "version": version,
            "description": "Paper trading on Alpaca. Money and quantities are decimal strings.",
        },
        "servers": [{"url": "/"}],
        "tags": [{"name": t} for t in sorted({op.tag for _, _, op in routes})],
        "paths": paths,
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT",
                               "description": "POST /api/v1/auth/login returns it"},
                "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key",
                           "description": "POST /api/v1/api-keys creates one"},
                "metricsToken": {"type": "http", "scheme": "bearer",
                                 "description": "The METRICS_TOKEN setting; unset, /metrics answers 404"},
            },
        },
    }
