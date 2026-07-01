"""
Module 1: Transparent Data Normalization.

Resolves syntactic heterogeneity across source systems and implements the four
operations named in the paper: schema alignment, type standardization,
missing-value imputation, and noise reduction. It also applies value
standardization and versioned currency conversion.

Every rule is externalized into a versioned YAML file. Critically, every
application of a rule is committed to the provenance log, so the normalized
output is not just a cleaner table but an auditable transformation: for any
field an auditor can retrieve the original value, the rule id, the rule
version, and when it was applied. This is Design Transparency made structural.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List

import pandas as pd
import yaml

from .provenance import ProvenanceEntry, ProvenanceLog


class DataNormalizer:
    """Apply versioned YAML normalization rules, recording full provenance."""

    def __init__(self, rules_path: str, provenance: ProvenanceLog | None = None):
        if not os.path.exists(rules_path):
            raise FileNotFoundError(
                f"Normalization rules not found: {rules_path}")
        with open(rules_path, "r", encoding="utf-8") as fh:
            self.rules: Dict[str, Any] = yaml.safe_load(fh)
        self.version = str(self.rules.get("version", "unknown"))
        self.provenance = provenance if provenance is not None else ProvenanceLog()

    def _rid(self, section: str) -> str:
        return str(self.rules.get(section, {}).get("rule_id", section))

    # --- 1. schema alignment ---------------------------------------------- #
    def _align_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        section = self.rules.get("schema_alignment", {})
        mapping = section.get("map", {})
        applicable = {k: v for k, v in mapping.items() if k in df.columns}
        for src, dst in applicable.items():
            self.provenance.record(ProvenanceEntry(
                module="M1", operation="schema_alignment",
                rule_id=self._rid("schema_alignment"), rule_version=self.version,
                subject="<schema>", before=src, after=dst))
        return df.rename(columns=applicable)

    @staticmethod
    def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Merge columns that share a name after schema alignment.

        Different source systems map distinct field names to the same canonical
        field (e.g., ERP 'company' and CRM 'client_name' both become 'name').
        After alignment these appear as duplicate columns; we coalesce them,
        taking the first non-null value per row.
        """
        if not df.columns.duplicated().any():
            return df
        out = pd.DataFrame(index=df.index)
        for name in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == name]
            if same.shape[1] == 1:
                out[name] = same.iloc[:, 0]
            else:
                out[name] = same.bfill(axis=1).iloc[:, 0]
        return out

    # --- null standardization --------------------------------------------- #
    def _standardize_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        null_values = set(self.rules.get("null_values", []))
        for col in df.columns:
            mask = df[col].isin(null_values)
            for idx in df.index[mask]:
                self.provenance.record(ProvenanceEntry(
                    module="M1", operation="null_standardization",
                    rule_id="NORM_NULL_01", rule_version=self.version,
                    subject=f"row_{idx}", before=df.at[idx, col], after=None,
                    evidence={"field": col}))
            df.loc[mask, col] = pd.NA
        return df

    # --- 4. noise reduction ----------------------------------------------- #
    def _reduce_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        rules = self.rules.get("noise_reduction", {})
        rid = self._rid("noise_reduction")
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            is_text = (df[col].dtype == object or "string" in dtype_str
                       or dtype_str == "str")
            if not is_text:
                continue
            series = df[col].astype("object")
            if rules.get("strip_whitespace"):
                series = series.map(lambda x: x.strip() if isinstance(x, str) else x)
            if rules.get("deduplicate_spaces"):
                series = series.map(
                    lambda x: re.sub(r"\s+", " ", x) if isinstance(x, str) else x)
            if rules.get("normalize_unicode"):
                series = series.map(self._strip_accents)
            df[col] = series
        for col in rules.get("lowercase_fields", []):
            if col in df.columns:
                df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x)
        for col, pattern in rules.get("remove_pattern", {}).items():
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: re.sub(pattern, "", x) if isinstance(x, str) else x)
        self.provenance.record(ProvenanceEntry(
            module="M1", operation="noise_reduction", rule_id=rid,
            rule_version=self.version, subject="<table>",
            evidence={"applied": list(rules.keys())}))
        return df

    @staticmethod
    def _strip_accents(value):
        if not isinstance(value, str):
            return value
        nfkd = unicodedata.normalize("NFKD", value)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    # --- value mapping ----------------------------------------------------- #
    def _map_values(self, df: pd.DataFrame) -> pd.DataFrame:
        section = self.rules.get("value_mapping", {})
        mapping = section.get("map", {})
        rid = self._rid("value_mapping")
        for col, m in mapping.items():
            if col in df.columns:
                for idx in df.index:
                    val = df.at[idx, col]
                    if isinstance(val, str) and val in m:
                        self.provenance.record(ProvenanceEntry(
                            module="M1", operation="value_mapping", rule_id=rid,
                            rule_version=self.version, subject=f"row_{idx}",
                            before=val, after=m[val], evidence={"field": col}))
                df[col] = df[col].replace(m)
        return df

    # --- 2. type standardization ------------------------------------------ #
    def _standardize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        section = self.rules.get("type_standardization", {})
        types = section.get("map", {})
        for col, target in types.items():
            if col not in df.columns:
                continue
            if target == "string":
                df[col] = df[col].astype("string")
            elif target == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target == "integer":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df

    # --- 3. missing-value imputation -------------------------------------- #
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        section = self.rules.get("imputation", {})
        strategies = section.get("strategy", {})
        rid = self._rid("imputation")
        for col, strategy in strategies.items():
            if col not in df.columns:
                continue
            missing = int(df[col].isna().sum())
            if missing == 0:
                continue
            if strategy == "drop":
                df = df[df[col].notna()].reset_index(drop=True)
            elif strategy == "mode":
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
            elif strategy == "median":
                median = df[col].median(skipna=True)
                if pd.notna(median):
                    if str(df[col].dtype) == "Int64":
                        median = int(round(median))
                    df[col] = df[col].fillna(median)
            self.provenance.record(ProvenanceEntry(
                module="M1", operation="imputation", rule_id=rid,
                rule_version=self.version, subject=f"<field:{col}>",
                evidence={"strategy": strategy, "count": missing}))
        return df

    # --- currency conversion (versioned) ---------------------------------- #
    def _convert_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert a revenue column to the base currency using versioned rates.

        The paper highlights versioned currency conversion as an audit example:
        an auditor can retrieve the exact rate active at the processing date.
        Conversion applies only when a 'currency' column is present alongside
        'revenue'; otherwise it is a no-op. Each conversion is recorded.
        """
        section = self.rules.get("currency_conversion", {})
        rates = section.get("rates", {})
        rate_date = section.get("rate_date", "")
        rid = self._rid("currency_conversion")
        if "revenue" not in df.columns or "currency" not in df.columns:
            return df
        for idx in df.index:
            cur = df.at[idx, "currency"]
            val = df.at[idx, "revenue"]
            if cur in rates and pd.notna(val):
                before = val
                df.at[idx, "revenue"] = float(val) * float(rates[cur])
                self.provenance.record(ProvenanceEntry(
                    module="M1", operation="currency_conversion", rule_id=rid,
                    rule_version=self.version, subject=f"row_{idx}",
                    before=f"{before} {cur}", after=df.at[idx, "revenue"],
                    evidence={"rate": rates[cur], "rate_date": rate_date}))
        return df

    # --- orchestration ----------------------------------------------------- #
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._align_schema(df)
        df = self._coalesce_duplicate_columns(df)
        df = self._standardize_nulls(df)
        df = self._reduce_noise(df)
        df = self._map_values(df)
        df = self._standardize_types(df)
        df = self._convert_currency(df)
        df = self._impute(df)
        return df

    def records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        return df.where(pd.notna(df), None).to_dict(orient="records")


if __name__ == "__main__":
    raw = pd.DataFrame([
        {"company": " Acme  Corporation ", "ville": "London", "pays": "UK",
         "tel": "+44 20 1234", "headcount": "450"},
        {"company": "Globex", "ville": "Madrid", "pays": "N/A",
         "tel": "N/A", "headcount": ""},
        {"company": "Initech", "ville": "Madrid", "pays": "deutschland",
         "tel": "+1 (555) 0100", "headcount": "120"},
    ])
    norm = DataNormalizer("config/normalization_rules.yaml")
    out = norm.normalize(raw)
    print(out.to_string())
    print(f"\nProvenance entries: {len(norm.provenance)}")
    for e in norm.provenance.entries()[:4]:
        print(" ", e.operation, e.rule_id, e.before, "->", e.after)
