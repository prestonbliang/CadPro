from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from cadpro.enrichment import (
    EnrichmentConfig,
    EnrichmentServiceError,
    EnrichmentUnavailableError,
    enrich_references,
    enrich_references_async,
    sanitize_query,
)


SOURCE_URL = "https://manufacturer.example/products/widget-42"


def _images(directory: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = directory / f"view-{index:02d}.png"
        Image.new("RGB", (1_600, 1_200), (index * 11 % 255, 80, 140)).save(path)
        paths.append(path)
    return paths


def _payload() -> dict:
    return {
        "object_identity": {
            "common_name": "Widget bracket",
            "manufacturer": "Example Manufacturing",
            "model_number": "W-42",
            "confidence": 0.82,
            "evidence": "The label and housing match a manufacturer page.",
            "source_urls": [SOURCE_URL, "javascript:alert(1)"],
        },
        "candidate_dimensions": [
            {
                "name": "overall width",
                "value": 42.0,
                "unit": "mm",
                "basis": "published_reference",
                "confidence": 0.75,
                "source_url": SOURCE_URL,
                "caveat": "Catalog value for the likely model.",
            },
            {
                "name": "boss diameter",
                "value": 8.0,
                "unit": "mm",
                "basis": "visual_estimate",
                "confidence": 0.31,
                "source_url": None,
                "caveat": "Estimated from proportions.",
            },
        ],
        "specification_facts": [
            {
                "name": "material",
                "value": "die-cast aluminum",
                "confidence": 0.7,
                "source_url": SOURCE_URL,
            }
        ],
        "cad_feature_observations": [
            {
                "name": "mounting boss",
                "description": "One cylindrical boss is visible on the front face.",
                "evidence_basis": "visible_image",
                "confidence": 0.9,
                "source_url": None,
            }
        ],
        "uncertainties": ["The rear face is partly occluded."],
        "source_urls": [SOURCE_URL, "file:///etc/passwd"],
    }


class FakeResponse:
    def __init__(self, payload: dict | str):
        self.output_text = payload if isinstance(payload, str) else json.dumps(payload)

    def model_dump(self):
        return {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"sources": [{"url": SOURCE_URL}]},
                }
            ]
        }


class FakeClient:
    def __init__(self, response: FakeResponse | Exception):
        self.response = response
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_default_is_local_only_and_deterministic(tmp_path):
    client = FakeClient(FakeResponse(_payload()))

    first = enrich_references([], " widget ", client=client)
    second = enrich_references([], " widget ", client=client)

    assert first == second
    assert first.status == "disabled"
    assert first.source_urls == ()
    assert first.to_dict()["candidate_dimensions"] == []
    assert client.calls == []


def test_explicit_enable_without_key_stays_disabled_and_does_not_read_images(tmp_path):
    result = enrich_references(
        [tmp_path / "missing.png"],
        config=EnrichmentConfig(enabled=True),
    )

    assert result.status == "disabled"
    assert "OPENAI_API_KEY" in result.warnings[0]


def test_provider_request_is_stateless_bounded_and_validated(tmp_path):
    paths = _images(tmp_path, 20)
    client = FakeClient(FakeResponse(_payload()))
    config = EnrichmentConfig(
        enabled=True,
        api_key="sk-test-secret",
        max_images=4,
        max_image_edge=256,
        max_image_bytes=30_000,
        max_total_image_bytes=80_000,
        max_query_chars=24,
    )

    report = enrich_references(
        paths,
        "  Widget\n\tmodel 42 with a very long hostile hint  ",
        config=config,
        client=client,
    )

    assert report.status == "completed"
    assert report.query == "Widget model 42 with a v"
    assert report.object_identity["source_urls"] == [SOURCE_URL]
    assert report.source_urls == (SOURCE_URL,)
    assert report.candidate_dimensions[0]["basis"] == "published_reference"
    estimate = report.candidate_dimensions[1]
    assert "not a measured manufacturing dimension" in estimate["caveat"]

    request = client.calls[0]
    assert request["store"] is False
    assert request["tools"] == [{"type": "web_search"}]
    assert request["include"] == ["web_search_call.action.sources"]
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    images = [
        item
        for item in request["input"][0]["content"]
        if item["type"] == "input_image"
    ]
    assert len(images) == 4
    assert sum(
        len(base64.b64decode(item["image_url"].partition(",")[2])) for item in images
    ) <= 80_000


def test_uncited_reference_claims_are_dropped(tmp_path):
    path = _images(tmp_path, 1)[0]
    payload = _payload()
    payload["candidate_dimensions"][0]["source_url"] = "https://hallucinated.invalid/dim"
    payload["specification_facts"][0]["source_url"] = "https://hallucinated.invalid/spec"
    payload["cad_feature_observations"].append(
        {
            "name": "hidden channel",
            "description": "Claimed by a reference.",
            "evidence_basis": "published_reference",
            "confidence": 0.8,
            "source_url": "https://hallucinated.invalid/feature",
        }
    )

    report = enrich_references(
        [path],
        config=EnrichmentConfig(enabled=True, api_key="sk-test"),
        client=FakeClient(FakeResponse(payload)),
    )

    assert [item["name"] for item in report.candidate_dimensions] == ["boss diameter"]
    assert report.specification_facts == ()
    assert [item["name"] for item in report.cad_feature_observations] == [
        "mounting boss"
    ]
    assert sum("Dropped an uncited" in item for item in report.uncertainties) == 3


def test_invalid_measurement_basis_and_provider_errors_are_sanitized(tmp_path):
    path = _images(tmp_path, 1)[0]
    payload = _payload()
    payload["candidate_dimensions"][0]["basis"] = "measured"

    with pytest.raises(EnrichmentServiceError) as invalid:
        enrich_references(
            [path],
            config=EnrichmentConfig(enabled=True, api_key="sk-secret"),
            client=FakeClient(FakeResponse(payload)),
        )
    assert "sk-secret" not in str(invalid.value)
    assert invalid.value.__cause__ is None

    with pytest.raises(EnrichmentServiceError) as provider:
        enrich_references(
            [path],
            config=EnrichmentConfig(enabled=True, api_key="sk-secret"),
            client=FakeClient(RuntimeError("request Authorization: Bearer sk-secret")),
        )
    assert "sk-secret" not in str(provider.value)
    assert provider.value.__cause__ is None


@pytest.mark.parametrize("value", [0, -12.5])
def test_nonpositive_candidate_dimensions_are_rejected(tmp_path, value):
    path = _images(tmp_path, 1)[0]
    payload = _payload()
    payload["candidate_dimensions"][0]["value"] = value

    with pytest.raises(EnrichmentServiceError, match="local reconstruction can continue"):
        enrich_references(
            [path],
            config=EnrichmentConfig(enabled=True, api_key="sk-secret"),
            client=FakeClient(FakeResponse(payload)),
        )


def test_async_wrapper_and_config_do_not_expose_secret(tmp_path):
    path = _images(tmp_path, 1)[0]
    config = EnrichmentConfig(enabled=True, api_key="sk-never-print")
    result = asyncio.run(
        enrich_references_async(
            [path], config=config, client=FakeClient(FakeResponse(_payload()))
        )
    )

    assert result.status == "completed"
    assert "sk-never-print" not in repr(config)
    assert EnrichmentConfig.from_env(
        {
            "CADPRO_AI_ENRICHMENT": "yes",
            "OPENAI_API_KEY": " test-key ",
            "CADPRO_AI_MODEL": "gpt-test",
        }
    ).available


def test_optional_sdk_is_not_required_for_local_use(tmp_path, monkeypatch):
    path = _images(tmp_path, 1)[0]
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(EnrichmentUnavailableError, match="local reconstruction"):
        enrich_references(
            [path],
            config=EnrichmentConfig(enabled=True, api_key="sk-secret"),
        )


def test_query_sanitizer_rejects_non_text_and_bounds_length():
    assert sanitize_query(" a\n\tb\x00c ", 5) == "a b c"
    assert sanitize_query("abcdefgh", 4) == "abcd"
    with pytest.raises(ValueError, match="text"):
        sanitize_query(123)  # type: ignore[arg-type]
