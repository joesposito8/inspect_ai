import hashlib
import logging
from typing import Any, Iterator

import pytest

from inspect_ai.dataset import (
    MemoryDataset,
    RecordToSample,
    Sample,
    align_variants,
    by_language,
    language_field_record,
    load_multilingual_config_per_language,
    variant_anchor,
    variants_of,
)


@pytest.fixture(autouse=True)
def _ensure_log_propagation() -> Iterator[None]:
    """Counter init_logger() setting propagate=False on inspect_ai logger."""
    lgr = logging.getLogger("inspect_ai")
    old_propagate = lgr.propagate
    lgr.propagate = True
    yield
    lgr.propagate = old_propagate


def map_records(mapper: RecordToSample, records: list[dict[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    for record in records:
        result = mapper(record)
        samples.extend([result] if isinstance(result, Sample) else result)
    return samples


def sample_for(language: str, anchor: str, input: str | None = None) -> Sample:
    return Sample(
        input=input or f"question {anchor} in {language}",
        target="answer",
        id=anchor,
        metadata={"language": language},
    )


def parallel_samples(languages: list[str], anchors: list[str]) -> list[Sample]:
    return [sample_for(lang, anchor) for anchor in anchors for lang in languages]


def anchor_from_id(sample: Sample) -> str:
    return str(sample.id)


# 1. happy path: 3 languages, tier-1 ids, 100% pairing


def test_align_variants_happy_path() -> None:
    samples = parallel_samples(["en", "de", "ar"], ["abstract_algebra/test/0"])
    report = align_variants(
        samples, name_prefix="global_mmlu", anchor_key=anchor_from_id
    )
    assert report.pairing_rate == 1.0
    assert report.paired_groups == 1
    for sample, lang in zip(samples, ["en", "de", "ar"]):
        assert sample.id == f"global_mmlu:abstract_algebra/test/0:{lang}"
        assert sample.metadata is not None
        assert sample.metadata["variant_of"] == "global_mmlu:abstract_algebra/test/0"
        assert sample.metadata["language"] == lang


# 2. #668 regression: per-language hash of translated input -> 0% -> hard fail


def test_align_variants_rejects_language_dependent_hash_anchors() -> None:
    samples = [
        sample_for(lang, anchor=f"q{i}", input=f"question {i} translated to {lang}")
        for i in range(3)
        for lang in ["en", "es"]
    ]

    def hashed_anchor(sample: Sample) -> str:
        assert isinstance(sample.input, str)
        return hashlib.sha256(sample.input.encode()).hexdigest()[:8]

    with pytest.raises(ValueError, match="language-dependent"):
        align_variants(samples, name_prefix="mgsm", anchor_key=hashed_anchor)


# 3. row-index tier: equal-length parallel configs pair; unequal lengths fail loudly


def test_row_index_anchor_parallel_configs() -> None:
    samples = [
        sample_for(lang, anchor=variant_anchor({}, index=i))
        for lang in ["en", "fr"]
        for i in range(4)
    ]
    report = align_variants(samples, name_prefix="mgsm", anchor_key=anchor_from_id)
    assert report.pairing_rate == 1.0
    assert samples[0].id == "mgsm:row0:en"
    assert samples[4].metadata is not None
    assert samples[4].metadata["variant_of"] == "mgsm:row0"


def test_row_index_anchor_unequal_lengths_fail_loudly() -> None:
    samples = [
        sample_for("en", anchor=variant_anchor({}, index=i)) for i in range(6)
    ] + [sample_for("fr", anchor=variant_anchor({}, index=i)) for i in range(4)]
    with pytest.raises(ValueError, match="min_pairing_rate"):
        align_variants(
            samples,
            name_prefix="mgsm",
            anchor_key=anchor_from_id,
            min_pairing_rate=1.0,
        )


# 4. within-language duplicate anchors: error (default) / drop


def test_within_language_duplicate_raises_by_default() -> None:
    samples = parallel_samples(["en", "de"], ["q1"]) + [sample_for("en", "q1")]
    with pytest.raises(ValueError, match="more than once"):
        align_variants(samples, name_prefix="ds", anchor_key=anchor_from_id)


def test_within_language_duplicate_drop_removes_group_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    samples = (
        parallel_samples(["en", "de"], ["q1"])
        + [sample_for("en", "q1")]
        + parallel_samples(["en", "de"], ["q2"])
    )
    with caplog.at_level(logging.WARNING):
        report = align_variants(
            samples, name_prefix="ds", anchor_key=anchor_from_id, on_duplicate="drop"
        )
    assert report.dropped_groups == 1
    assert report.total_samples == 5
    assert report.paired_groups == 1
    assert len(samples) == 2
    assert {sample.id for sample in samples} == {"ds:q2:en", "ds:q2:de"}
    assert "dropped 1 group" in caplog.text
    # partition invariant: paired + unpaired + dropped == total
    assert report.pairing_rate == 2 / 5


def test_drop_requires_a_list() -> None:
    samples = tuple(parallel_samples(["en", "de"], ["q1"]))
    with pytest.raises(TypeError, match="requires a list"):
        align_variants(
            samples, name_prefix="ds", anchor_key=anchor_from_id, on_duplicate="drop"
        )


# 5. partial coverage: pairs where possible, variant_of=None otherwise


def test_partial_language_coverage() -> None:
    samples = parallel_samples(["en", "de", "ar"], ["q1"]) + [
        sample_for("en", "q2"),
        sample_for("de", "q2"),
        sample_for("ar", "q3"),
    ]
    report = align_variants(samples, name_prefix="ds", anchor_key=anchor_from_id)
    assert report.paired_groups == 2
    assert report.unpaired_samples == 1
    assert report.pairing_rate == 5 / 6
    q2_en = next(s for s in samples if s.id == "ds:q2:en")
    assert q2_en.metadata is not None
    assert q2_en.metadata["variant_of"] == "ds:q2"
    q3 = next(s for s in samples if s.id == "q3")  # unpaired keeps its id
    assert q3.metadata is not None
    assert q3.metadata["variant_of"] is None


# 6. peer set without source language: pairing succeeds; loader warns only


def test_peer_set_without_source_language_pairs() -> None:
    samples = parallel_samples(["de", "fr", "ar"], ["q1", "q2"])
    report = align_variants(samples, name_prefix="ds", anchor_key=anchor_from_id)
    assert report.pairing_rate == 1.0
    assert report.languages == ["ar", "de", "fr"]


def test_loader_warns_when_source_language_absent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_hf_dataset(path: str, **kwargs: Any) -> MemoryDataset:
        language = kwargs["name"]
        return MemoryDataset([sample_for(language, anchor) for anchor in ["q1", "q2"]])

    monkeypatch.setattr(
        "inspect_ai.dataset._sources.multilingual.hf_dataset", fake_hf_dataset
    )
    with caplog.at_level(logging.WARNING):
        dataset = load_multilingual_config_per_language(
            path="org/peer-set",
            languages=["de", "fr"],
            record_to_sample=lambda record: Sample(input="unused"),
            anchor_key=anchor_from_id,
            name_prefix="peer",
        )
    assert "source language 'en'" in caplog.text
    assert len(dataset) == 4
    assert dataset[0].id == "peer:q1:de"
    assert dataset[0].metadata is not None
    assert dataset[0].metadata["language"] == "de"
    assert dataset[0].metadata["variant_of"] == "peer:q1"


# 7. namespacing: same raw anchor, different name_prefix -> distinct variant_of


def test_name_prefix_namespaces_anchors() -> None:
    samples_a = parallel_samples(["en", "de"], ["q1"])
    samples_b = parallel_samples(["en", "de"], ["q1"])
    align_variants(samples_a, name_prefix="dataset_a", anchor_key=anchor_from_id)
    align_variants(samples_b, name_prefix="dataset_b", anchor_key=anchor_from_id)
    assert samples_a[0].metadata is not None
    assert samples_b[0].metadata is not None
    assert samples_a[0].metadata["variant_of"] == "dataset_a:q1"
    assert samples_b[0].metadata["variant_of"] == "dataset_b:q1"
    assert samples_a[0].id != samples_b[0].id


# 8. language_field_record with anchor_field (MORU pattern)


def test_language_field_record_with_anchor() -> None:
    mapper = language_field_record(
        lambda record: Sample(input=str(record["prompt"])),
        language_field="language",
        languages=["en", "de"],
        anchor_field="record_id",
        name_prefix="moru",
    )
    records = [
        {"prompt": "hello", "language": "en", "record_id": 17},
        {"prompt": "hallo", "language": "de", "record_id": 17},
        {"prompt": "bonjour", "language": "fr", "record_id": 17},  # filtered
    ]
    samples = map_records(mapper, records)
    assert len(samples) == 2  # fr dropped
    for sample, lang in zip(samples, ["en", "de"]):
        assert sample.id == f"moru:17:{lang}"
        assert sample.metadata is not None
        assert sample.metadata["language"] == lang
        assert sample.metadata["variant_of"] == "moru:17"
    # re-validation on the same guard path is a fixpoint
    report = align_variants(
        samples,
        name_prefix="",
        anchor_key=lambda s: str(s.id).rsplit(":", 1)[0],
    )
    assert report.pairing_rate == 1.0
    assert samples[0].id == "moru:17:en"
    assert samples[0].metadata is not None
    assert samples[0].metadata["variant_of"] == "moru:17"


def test_language_field_record_requires_prefix_with_anchor() -> None:
    with pytest.raises(ValueError, match="name_prefix is required"):
        language_field_record(
            lambda record: Sample(input="x"),
            language_field="language",
            anchor_field="record_id",
        )


# 9. language_field_record without anchor: variant_of=None, never fabricated


def test_language_field_record_no_anchor() -> None:
    mapper = language_field_record(
        lambda record: Sample(input=str(record["prompt"])),
        language_field="language",
    )
    records = [
        {"prompt": "hello", "language": "en"},
        {"prompt": "hallo", "language": "de"},
    ]
    samples = map_records(mapper, records)
    assert len(samples) == 2
    for sample in samples:
        assert sample.metadata is not None
        assert "variant_of" in sample.metadata
        assert sample.metadata["variant_of"] is None


# 10. align_variants standalone: language-dependent long-shape anchor -> hard fail


def test_align_variants_standalone_misconfigured_anchor() -> None:
    # anchor_field was language-dependent: every anchor unique per language
    samples = [
        sample_for(lang, anchor=f"{lang}-{i}")
        for lang in ["en", "de"]
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="language-dependent"):
        align_variants(samples, name_prefix="", anchor_key=anchor_from_id)


# 11. align_variants standalone: hand-rolled samples, exact PairingReport


def test_align_variants_standalone_hand_rolled() -> None:
    samples = parallel_samples(["en", "zh-Hans"], ["item-1", "item-2", "item-3"])
    report = align_variants(samples, name_prefix="wide", anchor_key=anchor_from_id)
    assert report.total_samples == 6
    assert report.num_groups == 3
    assert report.paired_groups == 3
    assert report.unpaired_samples == 0
    assert report.dropped_groups == 0
    assert report.languages == ["en", "zh-Hans"]
    assert report.pairing_rate == 1.0
    assert samples[0].id == "wide:item-1:en"
    assert samples[1].id == "wide:item-1:zh-Hans"


# edge cases


def test_align_variants_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="no samples"):
        align_variants([], name_prefix="ds", anchor_key=anchor_from_id)


def test_align_variants_monolingual_input_is_legitimate() -> None:
    samples = [sample_for("en", f"q{i}") for i in range(3)]
    report = align_variants(samples, name_prefix="ds", anchor_key=anchor_from_id)
    assert report.paired_groups == 0
    assert report.pairing_rate == 0.0
    for sample in samples:
        assert sample.metadata is not None
        assert sample.metadata["variant_of"] is None


def test_align_variants_rejects_bad_anchor() -> None:
    samples = [Sample(input="x", id=None, metadata={"language": "en"})]
    with pytest.raises(ValueError, match="non-empty strings"):
        align_variants(
            samples,
            name_prefix="ds",
            anchor_key=lambda s: s.id,  # type: ignore[arg-type,return-value]
        )


def test_variant_anchor_dispatch() -> None:
    record = {"sample_id": "alg/0", "link": "https://x.y/p", "q": 3}
    assert variant_anchor(record, id_field="sample_id") == "alg/0"
    assert variant_anchor(record, id_field=("link", "q")) == "https://x.y/p|3"
    assert variant_anchor(record, index=7) == "row7"
    assert variant_anchor(record, index=7, prefix="mgsm") == "mgsm:row7"
    assert variant_anchor(record, id_field="sample_id", prefix="ds") == "ds:alg/0"
    with pytest.raises(ValueError, match="not both"):
        variant_anchor(record, id_field="sample_id", index=0)
    with pytest.raises(ValueError, match="no content-hash fallback"):
        variant_anchor(record)
    with pytest.raises(ValueError, match="is None"):
        variant_anchor({"link": None, "q": 3}, id_field=("link", "q"))


def test_variants_of_and_by_language() -> None:
    samples = parallel_samples(["en", "de"], ["q1", "q2"]) + [sample_for("fr", "q3")]
    align_variants(samples, name_prefix="ds", anchor_key=anchor_from_id)
    dataset = MemoryDataset(samples)

    group = variants_of(dataset, "ds:q1:en")
    assert {sample.id for sample in group} == {"ds:q1:en", "ds:q1:de"}
    # unpaired sample: group is just itself
    assert [sample.id for sample in variants_of(dataset, "q3")] == ["q3"]
    with pytest.raises(ValueError, match="no sample with id"):
        variants_of(dataset, "missing")

    assert [sample.id for sample in by_language(dataset, "de")] == [
        "ds:q1:de",
        "ds:q2:de",
    ]
    assert by_language(dataset, "ja") == []


def test_loader_rejects_empty_languages() -> None:
    with pytest.raises(ValueError, match="languages is empty"):
        load_multilingual_config_per_language(
            path="org/ds",
            languages=[],
            record_to_sample=lambda record: Sample(input="x"),
            anchor_key=anchor_from_id,
            name_prefix="ds",
        )
