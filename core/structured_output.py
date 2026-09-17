"""Bounded, offline JSON-schema checks shared by both chat providers."""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator, SchemaError, ValidationError

MAX_SCHEMA_CHARS = 64000


def _json(text):
    def invalid_constant(value):
        raise ValueError(f"JSON에는 {value}를 사용할 수 없습니다")
    return json.loads(text, parse_constant=invalid_constant)


def parse_schema(value):
    """Accept inline Draft 2020-12 schemas; never retrieve schema URLs."""
    try:
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(raw) > MAX_SCHEMA_CHARS:
            raise ValueError("JSON 스키마는 64,000자 이하여야 합니다")
        schema = _json(raw)
        if not isinstance(schema, dict) or not schema:
            raise ValueError("JSON 스키마는 비어 있지 않은 객체여야 합니다")
        nodes = 0

        def check(node, depth=0):
            nonlocal nodes
            nodes += 1
            if depth > 32 or nodes > 4000:
                raise ValueError("JSON 스키마가 너무 복잡합니다 (최대 깊이 32)")
            if isinstance(node, dict):
                if set(node) & {'$ref', '$dynamicRef', '$recursiveRef', '$id'}:
                    raise ValueError("현재는 참조($ref/$id) 없는 인라인 스키마를 사용해 주세요")
                if '$schema' in node and node['$schema'] not in (
                    'https://json-schema.org/draft/2020-12/schema',
                    'https://json-schema.org/draft/2020-12/schema#',
                ):
                    raise ValueError("JSON Schema Draft 2020-12를 사용해 주세요 ($schema 생략 가능)")
                for child in node.values():
                    check(child, depth + 1)
            elif isinstance(node, list):
                for child in node:
                    check(child, depth + 1)

        check(schema)
        Draft202012Validator.check_schema(schema)
        return schema
    except (SchemaError, RecursionError) as exc:
        raise ValueError(f"올바르지 않은 JSON 스키마: {getattr(exc, 'message', str(exc))[:400]}") from None
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 스키마를 읽을 수 없습니다: {str(exc)[:400]}") from None


def validate_output(content, schema):
    """Raise on incomplete/non-conforming output; never repair or strip text."""
    try:
        instance = _json(content)
        Draft202012Validator(schema).validate(instance)
    except (ValueError, ValidationError, RecursionError) as exc:
        raise ValueError(f"응답이 JSON 스키마와 일치하지 않습니다. 받은 내용은 유지됩니다: {getattr(exc, 'message', str(exc))[:400]}") from None
