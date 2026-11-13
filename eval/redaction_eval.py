"""Redaction precision/recall harness — measures the redactor against a
labeled sample set covering every document type and identifier the app uses.

Labeled samples include the exact identifier formats the corpus produces
(synthetic contacts in ticket subjects, agreement/SLA doc ids, lookup ids)
plus standard PII fixtures (SSN/card/phone/email). Preserved identifiers —
the citation/lookup keys the app depends on — are explicitly labeled and
must *not* be masked.

Usage: ``copilot eval redaction`` → JSON report under eval/reports/.

Author: venkateshwara doijode
"""

import json
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.dependencies import get_pii_redactor
from app.security.pii import REDACT_ENTITIES

# (text, expected_redacted_entities, expected_preserved_spans)
# expected_redacted_entities: {entity: count} the redactor must detect+mask
# expected_preserved_spans: substrings that must survive redaction untouched
LABELED_SAMPLES: list[tuple[str, dict[str, int], list[str]]] = [
    # synthetic ticket-subject PII (matches corpus SYNTHETIC_PEOPLE format)
    (
        "Dispute of wire transfer fee — contact Maria Delgado "
        "(maria.delgado@acme.example.com, +1-212-555-0148)",
        {"PERSON": 1, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1},
        [],
    ),
    (
        "Statement not received — contact Priya Nair "
        "(priya.nair@bluepeak.example.com, +1-415-555-0199)",
        {"PERSON": 1, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1},
        [],
    ),
    # standard PII fixtures
    ("Applicant SSN 123-45-6789 verified.", {"US_SSN": 1}, []),
    ("Card 4111 1111 1111 1111 was declined.", {"CREDIT_CARD": 1}, []),
    # business-specific identifiers — sensitive → must be redacted
    ("Wire to account ACC-123456 was returned.", {"ACCOUNT_NUMBER": 1}, ["ACC"]),
    ("Access request for staff member EMP-1024.", {"EMPLOYEE_ID": 1}, ["EMP"]),
    # preserved identifiers — citation/lookup keys, must NOT be masked
    ("Ticket TKT-00037 is currently open.", {}, ["TKT-00037"]),
    ("Trade TRD-00012 settled on 2025-03-14.", {}, ["TRD-00012"]),
    ("See agreement AGR-001 §2 for the fee schedule.", {}, ["AGR-001"]),
    ("SLA-006 defines the wire cutoff.", {}, ["SLA-006"]),
    # mixed: PII plus preserved ids in one string
    (
        "Ticket TKT-00042 — contact John Smith (john.smith@corp.example.com)",
        {"PERSON": 1, "EMAIL_ADDRESS": 1},
        ["TKT-00042"],
    ),
    # non-PII control sample — nothing should be masked
    ("The outgoing domestic wire fee is $25.00 per item.", {}, ["$25.00"]),
]


def run_redaction_eval(output_dir: Path | None = None) -> dict:
    """Run the redaction precision/recall measurement and write a JSON report.

    Precision = detected-redactable spans that were truly PII / all masked
    spans. Recall = expected PII spans masked / all expected PII spans.
    Preserved-span survival is reported separately — any masked preserve
    id is a retrieval-breaking regression.
    """
    redactor = get_pii_redactor()
    per_sample = []
    tp = fp = fn = 0
    preserve_violations = 0

    for text, expected, preserved in LABELED_SAMPLES:
        findings = redactor.analyze(text)
        redacted = redactor.redact(text)

        # recall: every expected redactable entity must appear in findings
        # with the right type; false positives = redactable findings beyond
        # expected count (approximation without span-level labels)
        found_counts: dict[str, int] = {}
        for f in findings:
            if f["entity"] in REDACT_ENTITIES:
                found_counts[f["entity"]] = found_counts.get(f["entity"], 0) + 1
        for entity, count in expected.items():
            got = found_counts.get(entity, 0)
            tp += min(got, count)
            fn += max(count - got, 0)
            fp += max(got - count, 0)
        # preserved spans must survive the redaction unchanged
        violations = [s for s in preserved if s not in redacted]
        preserve_violations += len(violations)

        per_sample.append(
            {
                "text": text[:80],
                "expected": expected,
                "found": found_counts,
                "preserve_violations": violations,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    report = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "samples": len(LABELED_SAMPLES),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "preserve_violations": preserve_violations,
        "per_sample": per_sample,
    }
    out_dir = output_dir or (get_settings().eval_dir / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"redaction_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(path)
    return report


if __name__ == "__main__":
    r = run_redaction_eval()
    print(json.dumps({k: v for k, v in r.items() if k != "per_sample"}, indent=2))
