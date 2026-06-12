import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from typing_extensions import NotRequired, TypedDict

from .._dataset import (
    Dataset,
    DatasetRecord,
    MemoryDataset,
    RecordToSample,
    Sample,
)
from .hf import hf_dataset

logger = logging.getLogger(__name__)


class LanguageMetadata(TypedDict):
    """Editor hints for the multilingual metadata convention. Not enforced at runtime.

    Multilingual datasets carry two `Sample.metadata` keys:

    - `language`: BCP 47 code of the sample's language (`"en"`, `"de"`, `"zh-Hans"`).
    - `variant_of`: language-free group anchor shared by every member of an
      N-way variant set, formatted `"{prefix}:{anchor}"`; `None` = unpaired.

    Paired samples additionally use the id format `"{prefix}:{anchor}:{lang}"` —
    unique per language variant, with the group anchor recoverable by truncating
    the final `:`-separated segment.
    """

    language: str
    variant_of: NotRequired[str | None]


@dataclass
class PairingReport:
    """Outcome of an `align_variants()` run. Always logged; returned for assertions/CI."""

    total_samples: int
    """Samples passed in (includes samples later dropped)."""

    num_groups: int
    """Anchor groups observed (includes dropped groups)."""

    paired_groups: int
    """Groups with >= 2 languages, each appearing exactly once."""

    unpaired_samples: int
    """Samples left with `variant_of = None`."""

    dropped_groups: int
    """Groups removed under `on_duplicate="drop"` (0 otherwise)."""

    languages: list[str]
    """Sorted language codes observed across the samples."""

    pairing_rate: float
    """Paired samples / total samples."""


def default_language_key(sample: Sample) -> str:
    """Read the convention key `sample.metadata["language"]`.

    The default `language_key` for `align_variants()`.

    Args:
        sample: Sample to read the language from.

    Returns:
        The sample's BCP 47 language code.

    Raises:
        ValueError: If the sample has no `metadata["language"]` string.
    """
    language = (sample.metadata or {}).get("language")
    if not isinstance(language, str):
        raise ValueError(
            f"Sample {sample.id!r} has no metadata['language'] (found {language!r}). "
            "Set metadata['language'] when constructing samples, or pass an "
            "explicit language_key to align_variants()."
        )
    return language


def _set_variant(sample: Sample, variant_of: str | None) -> None:
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["variant_of"] = variant_of


def align_variants(
    samples: Sequence[Sample],
    *,
    name_prefix: str,
    anchor_key: Callable[[Sample], str],
    language_key: Callable[[Sample], str] = default_language_key,
    on_duplicate: Literal["error", "drop"] = "error",
    min_pairing_rate: float = 0.0,
    set_id: bool = True,
) -> PairingReport:
    """Apply the alignment guard to already-constructed samples, in place.

    Groups samples by `anchor_key` — a **language-invariant** anchor — and pairs
    each group observed in two or more languages (each language exactly once),
    setting on every member:

    - `sample.metadata["variant_of"] = f"{name_prefix}:{anchor}"` (the language-free
      group anchor; just `anchor` when `name_prefix` is empty)
    - `sample.id = f"{name_prefix}:{anchor}:{lang}"` (when `set_id` is True;
      unpaired samples keep their original ids)

    Groups observed in a single language are kept with `variant_of = None`
    (excluded from paired analysis, present in the dataset). Pairing precision is
    prioritized over recall throughout: a false pairing silently corrupts every
    paired statistic computed over it, while a missing pairing merely excludes an
    item — so failures are loud and pairings are never fabricated.

    This is the public guard: `load_multilingual_config_per_language()` calls it
    internally, and `language_field_record()` output or hand-rolled wide-shape
    loaders can validate their pairing on the same code path (for samples that
    already carry prefixed ids, the recipe
    `align_variants(samples, name_prefix="", anchor_key=lambda s: str(s.id).rsplit(":", 1)[0])`
    re-validates without altering ids). Run over a dataset's loaded samples in CI,
    it turns silent pairing collapse into a hard error: anchors hashed from
    translated text differ per language (the `inspect_evals#668` bug), drive the
    pairing rate to 0%, and fail here instead of corrupting downstream statistics.

    Mutates samples in place and is not idempotent when `set_id=True` with an
    id-derived `anchor_key` and a non-empty `name_prefix` — run once per load.

    Args:
        samples: Samples to align, each carrying `metadata["language"]` (or a
            language readable via `language_key`). Must be a `list` when
            `on_duplicate="drop"` (dropped samples are removed in place).
        name_prefix: Dataset namespace (e.g. `"global_mmlu"`) prepended to
            anchors, preventing cross-dataset collisions. Pass `""` to use
            anchors that are already namespaced.
        anchor_key: Returns the language-invariant anchor for a sample (e.g.
            `lambda s: str(s.id)` when ids are shared across language configs).
            Must never derive from translated text.
        language_key: Returns the language code for a sample. Defaults to
            reading `metadata["language"]`.
        on_duplicate: What to do when the same anchor appears more than once
            within one language: `"error"` (default) raises; `"drop"` removes
            the whole group and logs it.
        min_pairing_rate: Minimum acceptable pairing rate in [0.0, 1.0]. The
            default 0.0 only enforces the unconditional 0% check; callers with
            fully parallel corpora can demand e.g. `0.99`.
        set_id: Whether to rewrite paired samples' ids to
            `"{name_prefix}:{anchor}:{lang}"`.

    Returns:
        A `PairingReport` describing the input processed (counts partition as
        `paired + unpaired_samples + dropped == total_samples`).

    Raises:
        ValueError: If `samples` is empty; if an anchor is not a non-empty
            string; on within-language duplicate anchors (under `"error"`); if
            0% of groups pair across >= 2 observed languages (the signature of a
            language-dependent anchor); or if the pairing rate falls below
            `min_pairing_rate`.
        TypeError: If `on_duplicate="drop"` and `samples` is not a `list`.
    """
    if not samples:
        raise ValueError("align_variants: no samples provided.")
    if not 0.0 <= min_pairing_rate <= 1.0:
        raise ValueError(
            f"align_variants: min_pairing_rate must be in [0.0, 1.0], "
            f"got {min_pairing_rate}."
        )
    if on_duplicate == "drop" and not isinstance(samples, list):
        raise TypeError(
            "align_variants: on_duplicate='drop' removes samples from the input "
            f"in place and requires a list, not {type(samples).__name__}."
        )

    # group by anchor (anchor_key and language_key called exactly once per sample)
    keyed: list[tuple[str, Sample]] = []
    for sample in samples:
        anchor = anchor_key(sample)
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(
                f"align_variants: anchor_key returned {anchor!r} for sample "
                f"{sample.id!r} (anchors must be non-empty strings)."
            )
        keyed.append((anchor, sample))
    groups: dict[str, list[tuple[str, Sample]]] = {}
    for anchor, sample in keyed:
        groups.setdefault(anchor, []).append((language_key(sample), sample))

    # guard table per group
    languages: set[str] = set()
    paired_groups = 0
    unpaired_samples = 0
    dropped_samples = 0
    dropped_anchors: set[str] = set()
    for anchor, group in groups.items():
        counts = Counter(language for language, _ in group)
        languages.update(counts)
        if any(n > 1 for n in counts.values()):
            duplicated = sorted(language for language, n in counts.items() if n > 1)
            if on_duplicate == "error":
                raise ValueError(
                    f"align_variants: anchor {anchor!r} appears more than once "
                    f"within language(s) {duplicated}. Fix the anchor source, or "
                    "pass on_duplicate='drop' to remove such groups."
                )
            dropped_anchors.add(anchor)
            dropped_samples += len(group)
        elif len(counts) >= 2:
            paired_groups += 1
            variant_of = f"{name_prefix}:{anchor}" if name_prefix else anchor
            for language, sample in group:
                _set_variant(sample, variant_of)
                if set_id:
                    sample.id = f"{variant_of}:{language}"
        else:
            unpaired_samples += len(group)
            for _, sample in group:
                _set_variant(sample, None)

    if dropped_anchors:
        assert isinstance(samples, list)  # validated above
        samples[:] = [
            sample for anchor, sample in keyed if anchor not in dropped_anchors
        ]
        logger.warning(
            "align_variants[%s]: dropped %d group(s) (%d sample(s)) with "
            "within-language duplicate anchors.",
            name_prefix,
            len(dropped_anchors),
            dropped_samples,
        )

    total_samples = len(keyed)
    paired_samples = total_samples - unpaired_samples - dropped_samples
    report = PairingReport(
        total_samples=total_samples,
        num_groups=len(groups),
        paired_groups=paired_groups,
        unpaired_samples=unpaired_samples,
        dropped_groups=len(dropped_anchors),
        languages=sorted(languages),
        pairing_rate=paired_samples / total_samples,
    )
    logger.info(
        "align_variants[%s]: pairing rate %.1f%% (%d/%d samples in %d/%d groups; "
        "languages: %s)",
        name_prefix,
        report.pairing_rate * 100,
        paired_samples,
        total_samples,
        paired_groups,
        report.num_groups,
        ", ".join(report.languages),
    )

    if paired_groups == 0 and len(report.languages) >= 2:
        raise ValueError(
            f"align_variants: 0% of {report.num_groups} anchor groups paired "
            f"across {len(report.languages)} languages. This almost always means "
            "anchor_key returns language-dependent values — e.g. an id hashed "
            "from translated sample text (the inspect_evals#668 bug) — so the "
            "same question never shares an anchor across languages. Use a "
            "language-invariant anchor: an explicit id shared across language "
            "configs, a composite structural key, or (opt-in) the row index."
        )
    if report.pairing_rate < min_pairing_rate:
        raise ValueError(
            f"align_variants: pairing rate {report.pairing_rate:.1%} is below "
            f"the required min_pairing_rate={min_pairing_rate:.1%} "
            f"({paired_samples}/{total_samples} samples paired)."
        )
    return report


def variant_anchor(
    record: DatasetRecord,
    *,
    id_field: str | Sequence[str] | None = None,
    index: int | None = None,
    prefix: str = "",
) -> str:
    """Language-invariant anchor for one record.

    Anchor sources, in decreasing order of trust:

    - `id_field` (str): use `record[id_field]` verbatim — an explicit id that is
      identical across language configs (e.g. Global-MMLU `sample_id`).
    - `id_field` (sequence): join the named fields with `"|"` — a composite
      structural key (e.g. Belebele `("link", "question_number")`).
    - `index`: `f"row{index}"` — parallel row order. Opt-in and fragile: it
      cannot detect reordered or subsetted configs (only length/coverage drift
      surfaces, via the `align_variants()` pairing rate).
    - Neither: raise `ValueError`. There is deliberately no content-hash
      fallback — hashing sample text produces language-dependent anchors that
      silently break cross-language pairing (the `inspect_evals#668` bug).

    Leave `prefix=""` when the anchor feeds `align_variants()` or the bundled
    loader: they apply `name_prefix` themselves.

    Args:
        record: Raw dataset record.
        id_field: Field name (or sequence of field names) holding the
            language-invariant id.
        index: Row index for parallel-row-order datasets (mutually exclusive
            with `id_field`).
        prefix: Optional namespace prepended as `f"{prefix}:{anchor}"`.

    Returns:
        The anchor string.

    Raises:
        ValueError: If both or neither of `id_field`/`index` are given, or if
            the referenced fields are missing, `None`, or empty.

    Examples:
        >>> variant_anchor({"sample_id": "algebra/test/0"}, id_field="sample_id")
        'algebra/test/0'
        >>> variant_anchor({"link": "https://x.y/p", "q": 3}, id_field=("link", "q"))
        'https://x.y/p|3'
        >>> variant_anchor({}, index=7, prefix="mgsm")
        'mgsm:row7'
    """
    if id_field is not None and index is not None:
        raise ValueError("variant_anchor: pass id_field or index, not both.")
    if isinstance(id_field, str):
        anchor = str(record[id_field]) if record[id_field] is not None else ""
    elif id_field is not None:
        parts: list[str] = []
        for field_name in id_field:
            value = record[field_name]
            if value is None:
                raise ValueError(
                    f"variant_anchor: record field {field_name!r} is None."
                )
            parts.append(str(value))
        anchor = "|".join(parts)
    elif index is not None:
        anchor = f"row{index}"
    else:
        raise ValueError(
            "variant_anchor: no language-invariant anchor specified — pass "
            "id_field (an explicit id or composite key) or index (parallel row "
            "order, opt-in). There is deliberately no content-hash fallback: "
            "hashing sample text produces language-dependent anchors that "
            "silently break cross-language pairing (the inspect_evals#668 bug)."
        )
    if not anchor:
        raise ValueError(
            f"variant_anchor: empty anchor from id_field={id_field!r} "
            f"(record fields must be non-empty)."
        )
    return f"{prefix}:{anchor}" if prefix else anchor


def load_multilingual_config_per_language(
    path: str,
    languages: Sequence[str],
    record_to_sample: RecordToSample,
    *,
    anchor_key: Callable[[Sample], str],
    source_language: str | None = "en",
    split: str = "test",
    name_prefix: str,
    revision: str | None = None,
    on_duplicate: Literal["error", "drop"] = "error",
    min_pairing_rate: float = 0.0,
    **hf_kwargs: Any,
) -> MemoryDataset:
    """Load one Hugging Face config per language and align variants.

    Covers the dominant multilingual HF shape — one dataset config per language
    (Global-MMLU, MMLU-ProX, Belebele, MGSM, MMMLU, FLORES). For each language,
    calls `hf_dataset()` with `name=language`, sets `metadata["language"]` on
    every sample, then hands all samples to `align_variants()` — the single
    guard code path — which pairs variant groups, sets
    `metadata["variant_of"]`/`sample.id`, and fails loudly on pairing collapse.

    For datasets with no explicit shared id but parallel row order (e.g. MGSM),
    pass `auto_id=True` (positional ids are assigned per config before
    alignment) with `anchor_key=lambda s: str(s.id)` and leave `shuffle` off.

    Example (Global-MMLU; `sample_id` is identical across its 42 configs):

    ```python
    dataset = load_multilingual_config_per_language(
        path="CohereLabs/Global-MMLU",
        languages=["en", "de", "ar"],
        record_to_sample=record_to_sample,  # sets id=record["sample_id"]
        anchor_key=lambda s: str(s.id),
        name_prefix="global_mmlu",
        revision="<40-char dataset SHA>",
    )
    # id:       "global_mmlu:abstract_algebra/test/0:de"
    # metadata: {"language": "de", "variant_of": "global_mmlu:abstract_algebra/test/0"}
    ```

    Args:
        path: Path or name of the Hugging Face dataset.
        languages: Language configs to load (BCP 47 codes matching the
            dataset's config names).
        record_to_sample: Maps a raw record to a `Sample` (passed to
            `hf_dataset()` as `sample_fields`).
        anchor_key: Returns the language-invariant anchor for a constructed
            sample (e.g. `lambda s: str(s.id)`).
        source_language: Informational source/reference language. When not
            among `languages` a warning is logged (peer sets and source-absent
            slices are legitimate); pairing is unaffected.
        split: Dataset split to load.
        name_prefix: Dataset namespace for `variant_of`/`id` (e.g.
            `"global_mmlu"`).
        revision: Dataset revision (pin to a 40-char SHA for reproducibility).
        on_duplicate: Within-language duplicate anchor handling (see
            `align_variants()`).
        min_pairing_rate: Minimum acceptable pairing rate (see
            `align_variants()`).
        **hf_kwargs: Additional arguments passed through to `hf_dataset()`.

    Returns:
        One `MemoryDataset` wrapping the aligned samples from all languages.

    Raises:
        ValueError: If `languages` is empty, or on any `align_variants()`
            guard failure.
    """
    if not languages:
        raise ValueError("load_multilingual_config_per_language: languages is empty.")
    if source_language is not None and source_language not in languages:
        logger.warning(
            "load_multilingual_config_per_language[%s]: source language %r is "
            "not among the loaded languages %s; pairing proceeds over a peer set.",
            name_prefix,
            source_language,
            list(languages),
        )

    samples: list[Sample] = []
    for language in languages:
        dataset = hf_dataset(
            path,
            split=split,
            name=language,
            revision=revision,
            sample_fields=record_to_sample,
            **hf_kwargs,
        )
        for sample in dataset:
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["language"] = language
            samples.append(sample)

    align_variants(
        samples,
        name_prefix=name_prefix,
        anchor_key=anchor_key,
        on_duplicate=on_duplicate,
        min_pairing_rate=min_pairing_rate,
    )
    return MemoryDataset(samples, name=name_prefix, location=path)


def language_field_record(
    record_to_sample_inner: RecordToSample,
    *,
    language_field: str,
    languages: Sequence[str] | None = None,
    anchor_field: str | None = None,
    name_prefix: str = "",
) -> RecordToSample:
    """`RecordToSample` factory for the long language-as-field shape.

    For datasets with one row per language variant and the language in a column
    (Aya `language`, LinguaSafe `lang`). The returned mapper wraps
    `record_to_sample_inner`, per record:

    1. Reads `record[language_field]`; when `languages` is set and the value is
       not in it, drops the record (returns `[]`).
    2. Maps the record with the inner mapper and sets `metadata["language"]`.
    3. When `anchor_field` names a language-invariant shared record id, sets
       `metadata["variant_of"] = f"{name_prefix}:{record[anchor_field]}"` and
       `id = f"{name_prefix}:{record[anchor_field]}:{language}"`. When
       `anchor_field` is `None`, sets `variant_of = None` — a pairing is never
       fabricated.

    This factory tags each record independently; it cannot assert that anchors
    actually pair across languages. To close that gap, validate the loaded
    samples on the same guard code path:
    `align_variants(samples, name_prefix="", anchor_key=lambda s: str(s.id).rsplit(":", 1)[0])`
    — which hard-fails if `anchor_field` turns out to be language-dependent and
    leaves correctly-tagged samples unchanged.

    The wide shape (N parallel language columns in one row, e.g. M-ALERT
    en/de/fr/es/it) is not this factory's job: write a `record_to_sample` that
    returns one `Sample` per language column, then validate with
    `align_variants()` directly — the row itself is the language-invariant
    anchor (`variant_anchor()` with `id_field` or `index`).

    Args:
        record_to_sample_inner: Inner mapper from record to `Sample`.
        language_field: Record field holding the language code.
        languages: Languages to keep; `None` keeps all.
        anchor_field: Record field holding a language-invariant shared record
            id; `None` emits `variant_of = None`.
        name_prefix: Dataset namespace; required when `anchor_field` is set.

    Returns:
        A `RecordToSample` suitable for `hf_dataset(sample_fields=...)` and the
        other dataset readers.

    Raises:
        ValueError: At construction when `anchor_field` is set without a
            `name_prefix`; at mapping time when the anchor field is `None` or
            the inner mapper returns multiple samples for an anchored record
            (they would share an identical id).
    """
    if anchor_field is not None and not name_prefix:
        raise ValueError(
            "language_field_record: name_prefix is required when anchor_field "
            "is set (it namespaces variant_of against other datasets)."
        )

    def record_to_sample(record: DatasetRecord) -> list[Sample]:
        language = str(record[language_field])
        if languages is not None and language not in languages:
            return []
        result = record_to_sample_inner(record)
        result_samples = [result] if isinstance(result, Sample) else result
        if anchor_field is not None and len(result_samples) > 1:
            raise ValueError(
                f"language_field_record: the inner mapper returned "
                f"{len(result_samples)} samples for one record; with "
                "anchor_field set they would share an identical id. Map one "
                "sample per record, or apply align_variants() directly."
            )
        for sample in result_samples:
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["language"] = language
            if anchor_field is not None:
                anchor_value = record[anchor_field]
                if anchor_value is None:
                    raise ValueError(
                        f"language_field_record: record field {anchor_field!r} is None."
                    )
                variant_of = f"{name_prefix}:{anchor_value}"
                sample.metadata["variant_of"] = variant_of
                sample.id = f"{variant_of}:{language}"
            else:
                sample.metadata["variant_of"] = None
        return result_samples

    return record_to_sample


def variants_of(dataset: Dataset, sample_id: str) -> list[Sample]:
    """All samples sharing the given sample's `variant_of` group (including itself).

    Args:
        dataset: Dataset to search.
        sample_id: Id of any member of the variant group.

    Returns:
        The variant group's samples in dataset order. For an unpaired sample
        (`variant_of` is `None`), just that sample.

    Raises:
        ValueError: If no sample has the given id.
    """
    target = next((sample for sample in dataset if sample.id == sample_id), None)
    if target is None:
        raise ValueError(f"variants_of: no sample with id {sample_id!r} in dataset.")
    variant_of = (target.metadata or {}).get("variant_of")
    if variant_of is None:
        return [target]
    return [
        sample
        for sample in dataset
        if (sample.metadata or {}).get("variant_of") == variant_of
    ]


def by_language(dataset: Dataset, code: str) -> list[Sample]:
    """All samples with `metadata["language"] == code`.

    Args:
        dataset: Dataset to search.
        code: BCP 47 language code.

    Returns:
        Matching samples in dataset order.
    """
    return [
        sample for sample in dataset if (sample.metadata or {}).get("language") == code
    ]
