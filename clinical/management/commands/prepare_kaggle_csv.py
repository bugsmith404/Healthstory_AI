"""
Convert common Kaggle diabetes CSVs into the 11-feature format expected by train_risk_model.

Supported input formats
-----------------------
1. pima  — Pima Indians Diabetes (Kaggle: diabetes.csv)
           Columns: Pregnancies, Glucose, BloodPressure, SkinThickness,
                    Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

2. heart — UCI Heart Disease / Kaggle heart.csv
           Uses age, trestbps, chol, thalach mapped to our schema.

Usage
-----
python manage.py prepare_kaggle_csv --input path/to/diabetes.csv --format pima
python manage.py prepare_kaggle_csv --input diabetes.csv --format pima --output clinical/data/training_ready.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from clinical.services.ml_engine import ChronicDiseaseRiskEngine

FEATURE_ORDER = list(ChronicDiseaseRiskEngine._feature_order)


def _glucose_to_hba1c_estimate(glucose_mg_dl: float) -> float:
    """Rough ADA-style mapping when HbA1c is unavailable (single time-point datasets)."""
    return round((glucose_mg_dl + 46.7) / 28.7, 2)


def _estimate_ldl(bmi: float, glucose: float) -> float:
    """Proxy LDL when cholesterol column is missing (cross-sectional datasets)."""
    return round(90 + 0.8 * max(bmi - 25, 0) + 0.05 * max(glucose - 100, 0), 1)


def convert_pima_row(row: dict) -> dict:
    """Map one Pima Indians Diabetes row to training features."""
    glucose = float(row.get("Glucose") or row.get("glucose") or 0)
    bp = float(row.get("BloodPressure") or row.get("blood_pressure") or row.get("Blood Pressure") or 120)
    bmi = float(row.get("BMI") or row.get("bmi") or 27)
    age = float(row.get("Age") or row.get("age") or 50)
    label = int(float(row.get("Outcome") or row.get("outcome") or row.get("label") or 0))

    hba1c = _glucose_to_hba1c_estimate(glucose)
    ldl = _estimate_ldl(bmi, glucose)

    return {
        "age_years": age,
        "systolic_bp_avg": bp,
        "systolic_bp_slope_12m": 0.0,
        "bmi_avg": bmi,
        "bmi_slope_12m": 0.0,
        "fasting_glucose_avg": glucose,
        "fasting_glucose_slope_12m": 0.0,
        "hba1c_avg": hba1c,
        "hba1c_slope_12m": 0.0,
        "ldl_avg": ldl,
        "ldl_slope_12m": 0.0,
        "label": label,
    }


def convert_heart_row(row: dict) -> dict:
    """Map UCI/Kaggle heart disease row to training features."""
    age = float(row.get("age") or row.get("Age") or 55)
    bp = float(row.get("trestbps") or row.get("resting_blood_pressure") or 130)
    chol = float(row.get("chol") or row.get("cholesterol") or 200)
    # Heart dataset: target 0 = no disease, 1+ = disease — binarize
    raw_target = row.get("target") or row.get("Target") or row.get("num") or 0
    label = 1 if int(float(raw_target)) > 0 else 0

    bmi = 26.0  # not in classic heart CSV; use population prior
    glucose = 95.0

    return {
        "age_years": age,
        "systolic_bp_avg": bp,
        "systolic_bp_slope_12m": 0.0,
        "bmi_avg": bmi,
        "bmi_slope_12m": 0.0,
        "fasting_glucose_avg": glucose,
        "fasting_glucose_slope_12m": 0.0,
        "hba1c_avg": _glucose_to_hba1c_estimate(glucose),
        "hba1c_slope_12m": 0.0,
        "ldl_avg": min(chol, 250),
        "ldl_slope_12m": 0.0,
        "label": label,
    }


CONVERTERS = {
    "pima": convert_pima_row,
    "heart": convert_heart_row,
}


class Command(BaseCommand):
    help = "Convert Kaggle/UCI CSV to 11-feature training format for train_risk_model."

    def add_arguments(self, parser):
        parser.add_argument("--input", "-i", required=True, help="Path to raw Kaggle CSV.")
        parser.add_argument(
            "--format",
            "-f",
            choices=list(CONVERTERS.keys()),
            default="pima",
            help="Input dataset format (default: pima).",
        )
        parser.add_argument(
            "--output",
            "-o",
            default="clinical/data/training_ready.csv",
            help="Output path for converted CSV.",
        )

    def handle(self, *args, **options):
        input_path = Path(options["input"])
        output_path = Path(options["output"])

        if not input_path.exists():
            raise CommandError(f"Input file not found: {input_path}")

        converter = CONVERTERS[options["format"]]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0

        with input_path.open(newline="", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            rows_out = []
            for row in reader:
                try:
                    features = converter(row)
                    if features["label"] not in (0, 1):
                        skipped += 1
                        continue
                    rows_out.append(features)
                    converted += 1
                except (ValueError, TypeError):
                    skipped += 1

        if converted < 20:
            raise CommandError(
                f"Only {converted} valid rows converted. Check --format matches your CSV columns."
            )

        fieldnames = FEATURE_ORDER + ["label"]
        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

        pos_rate = sum(r["label"] for r in rows_out) / len(rows_out)
        self.stdout.write(self.style.SUCCESS(f"Wrote {converted} rows to {output_path}"))
        self.stdout.write(f"  Skipped:     {skipped}")
        self.stdout.write(f"  Pos. rate:   {pos_rate:.1%}")
        self.stdout.write(
            f"\nNext step:\n"
            f"  python manage.py train_risk_model --csv {output_path} --label-column label --evaluate"
        )
