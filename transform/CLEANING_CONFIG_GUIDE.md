# Data Cleaning Configuration Guide

## Overview

The data cleaning system uses pattern-based configuration to handle different column types flexibly. This guide explains all available options and their interactions.

---

## Configuration Structure

```json
{
  "$schema": "./cleaning_config.schema.json",
  "version": 1,
  "global_settings": { ... },
  "column_rules": [ ... ],
  "relationship_validations": [ ... ]
}
```

---

## Global Settings

Default values applied to all columns unless overridden in column rules.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_null_threshold` | number (0.0-1.0) | 0.5 | Ratio of null values that triggers column deletion |
| `default_allow_column_deletion` | boolean | true | Whether columns can be deleted |
| `default_imputation_strategy` | string | "auto" | Default imputation method |
| `remove_duplicates` | boolean | false | Whether to drop exact duplicate rows |

---

## Column Rules

Rules are processed in order. **First matching pattern wins.**

### Required Fields

- **`pattern`** (string): Regex pattern to match column names
  - Examples: `"^ticker$"`, `"^balance_sheet_.*"`, `".*"`

### Optional Fields

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `dtype` | string | `auto`, `string`, `float`, `int`, `datetime` | Target data type |
| `null_threshold` | number | 0.0 to 1.0 | Override global threshold |
| `allow_column_deletion` | boolean | true/false | Override global deletion policy |
| `imputation_strategy` | string | See below | How to handle missing values |
| `imputation_value` | any | - | Value for `constant` strategy |
| `validations` | array | See below | Row-level validation rules |
| `comment` | string | - | Human-readable description |

---

## Null Handling Behavior

The system determines what to do with null values based on three settings:

### Decision Flow

```
For each column with null values:
│
├─ Calculate null_ratio = (null_count / total_rows)
│
├─ Is null_ratio > null_threshold?
│  │
│  ├─ YES: Is allow_column_deletion = true?
│  │  ├─ YES → DELETE COLUMN
│  │  └─ NO  → IMPUTE (using imputation_strategy)
│  │
│  └─ NO: Is null_count > 0?
│     ├─ YES → IMPUTE (using imputation_strategy)
│     └─ NO  → NO ACTION
```

### Key Points

1. **`null_threshold`** determines when deletion is *considered*
2. **`allow_column_deletion`** determines if deletion is *allowed*
3. **`imputation_strategy`** determines *how* to fill nulls (if not deleted)

---

## Imputation Strategies

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **`auto`** | Auto-detect based on dtype:<br>• Numeric → `normal_distribution`<br>• Datetime → `unix_epoch`<br>• Other → `constant` | Default for most cases |
| **`normal_distribution`** | Sample from N(μ, σ) of non-null values | Numeric data |
| **`unix_epoch`** | Use 1970-01-01 (with timezone if applicable) | Datetime columns |
| **`constant`** | Use value from `imputation_value` field | Categorical data |
| **`none`** | **Skip imputation entirely (leave nulls as-is)** | When you want to preserve nulls |

---

## Common Configuration Patterns

### 1. **Critical Column (Never Delete, Always Impute)**

```json
{
  "pattern": "^ticker$",
  "dtype": "string",
  "null_threshold": 0.0,
  "allow_column_deletion": false,
  "imputation_strategy": "constant",
  "imputation_value": "UNKNOWN"
}
```

**Behavior:**
- ✓ Threshold 0.0 means even 1 null would trigger deletion *if allowed*
- ✓ But deletion is disabled, so nulls are always imputed
- ✓ Imputes with "UNKNOWN" constant

---

### 2. **Optional Column (Keep if Enough Data, Impute if Sparse, Delete if Too Sparse)**

```json
{
  "pattern": "^vwap$",
  "dtype": "float",
  "null_threshold": 0.8,
  "allow_column_deletion": true,
  "imputation_strategy": "normal_distribution"
}
```

**Behavior:**
- ✗ If >80% nulls → Delete column
- ✓ If ≤80% nulls → Keep and impute using normal distribution

---

### 3. **Preserve Nulls (No Deletion, No Imputation)**

```json
{
  "pattern": "^optional_metric$",
  "dtype": "float",
  "null_threshold": 1.0,
  "allow_column_deletion": false,
  "imputation_strategy": "none"
}
```

**Behavior:**
- ✓ Never delete (threshold 1.0 means only delete if 100% null, but deletion disabled anyway)
- ✓ Never impute (strategy "none")
- ✓ Nulls remain as NaN/None in the dataframe

---

### 4. **Delete if Too Sparse, Otherwise Leave Nulls**

```json
{
  "pattern": "^experimental_.*",
  "dtype": "float",
  "null_threshold": 0.7,
  "allow_column_deletion": true,
  "imputation_strategy": "none"
}
```

**Behavior:**
- ✗ If >70% nulls → Delete column
- ✓ If ≤70% nulls → Keep but don't impute (nulls preserved)

---

### 5. **Always Delete if Any Nulls**

```json
{
  "pattern": "^strict_column$",
  "dtype": "float",
  "null_threshold": 0.0,
  "allow_column_deletion": true,
  "imputation_strategy": "auto"
}
```

**Behavior:**
- ✗ Even 1 null (>0%) triggers deletion
- Note: `imputation_strategy` is irrelevant here since column gets deleted

---

## Validation Rules

Row-level validations that filter out invalid rows.

| Validation | Applies To | Behavior |
|------------|-----------|----------|
| `positive` | Numeric | Keep rows where `value > 0` |
| `non_negative` | Numeric | Keep rows where `value >= 0` |
| `no_future_dates` | Datetime | Keep rows where `ts <= now(UTC)` |

**Example:**
```json
{
  "pattern": "^(open|high|low|close)$",
  "dtype": "float",
  "validations": ["positive"]
}
```
Rows with `open ≤ 0` are dropped.

---

## Relationship Validations

Cross-column checks that validate relationships between multiple columns.

### Available Checks

**1. `high_low_relationship`** (Stock data)
```json
{
  "name": "stock_high_low_check",
  "required_columns": ["high", "low", "open", "close"],
  "check_type": "high_low_relationship",
  "action_on_failure": "drop_rows"
}
```
Validates: `high >= max(open, close)` AND `low <= min(open, close)`

**2. `vwap_in_range`** (Stock data)
```json
{
  "name": "vwap_bounds_check",
  "required_columns": ["vwap", "low", "high"],
  "check_type": "vwap_in_range",
  "action_on_failure": "set_null"
}
```
Validates: `low <= vwap <= high`

### Graceful Handling

- If required columns are missing → **Warning** (skip check)
- If check fails → **Error** in report, but processing continues
- `action_on_failure`:
  - `drop_rows`: Remove invalid rows
  - `set_null`: Set violating values to NaN

---

## FAQ

### Q: What's the difference between `null_threshold` and `allow_column_deletion`?

**A:** They work together:
- `null_threshold` sets the *condition* for deletion (null_ratio > threshold)
- `allow_column_deletion` is the *permission* to delete

Both must be true for deletion to occur.

---

### Q: Can I keep a column but skip imputation?

**A:** Yes! Set `imputation_strategy: "none"`:
```json
{
  "pattern": "^my_column$",
  "imputation_strategy": "none",
  "allow_column_deletion": false
}
```

---

### Q: How do I make a column never delete, even with 100% nulls?

**A:** Set `allow_column_deletion: false`:
```json
{
  "pattern": "^critical_col$",
  "allow_column_deletion": false
}
```

---

### Q: What if I want to delete ONLY if 100% null, otherwise keep nulls as-is?

**A:**
```json
{
  "pattern": "^my_column$",
  "null_threshold": 0.99,
  "allow_column_deletion": true,
  "imputation_strategy": "none"
}
```
(Threshold 0.99 means delete if >99% null, which effectively means 100%)

---

### Q: Are column rules evaluated in order?

**A:** Yes! **First matching pattern wins.** Put more specific patterns before general ones:
```json
"column_rules": [
  {"pattern": "^ticker$", ...},           // Specific
  {"pattern": "^balance_sheet_.*", ...},  // Medium specific
  {"pattern": ".*", ...}                  // Catch-all (should be last)
]
```

---

### Q: Is `null_threshold` optional?

**A:** Yes, it uses `global_settings.default_null_threshold` (default: 0.5) if not specified.

---

### Q: What happens to columns that don't match any pattern?

**A:** The catch-all pattern `".*"` should be the last rule to handle unmatched columns. If no catch-all exists, columns are skipped (but this is not recommended).

---

## Example: Full Configuration

```json
{
  "$schema": "./cleaning_config.schema.json",
  "version": 1,
  "global_settings": {
    "default_null_threshold": 0.5,
    "default_allow_column_deletion": true,
    "default_imputation_strategy": "auto",
    "remove_duplicates": false
  },
  "column_rules": [
    {
      "comment": "Critical identifier - never delete",
      "pattern": "^ticker$",
      "dtype": "string",
      "null_threshold": 0.0,
      "allow_column_deletion": false,
      "imputation_strategy": "constant",
      "imputation_value": "UNKNOWN"
    },
    {
      "comment": "Financial data - high tolerance, preserve some nulls",
      "pattern": "^balance_sheet_.*",
      "dtype": "float",
      "null_threshold": 0.7,
      "allow_column_deletion": true,
      "imputation_strategy": "none"
    },
    {
      "comment": "Catch-all for unmatched columns",
      "pattern": ".*",
      "dtype": "auto",
      "null_threshold": 0.5,
      "imputation_strategy": "auto"
    }
  ],
  "relationship_validations": [
    {
      "name": "stock_high_low_check",
      "description": "Ensure high/low bounds are valid",
      "required_columns": ["high", "low", "open", "close"],
      "check_type": "high_low_relationship",
      "action_on_failure": "drop_rows"
    }
  ]
}
```

---

## Quick Reference Table

| Goal | `null_threshold` | `allow_column_deletion` | `imputation_strategy` |
|------|------------------|-------------------------|----------------------|
| Delete if >50% null, else impute | 0.5 | true | auto/normal_distribution |
| Never delete, always impute | any | **false** | auto/normal_distribution |
| Never delete, preserve nulls | any | **false** | **none** |
| Delete if >70% null, else preserve | 0.7 | true | **none** |
| Delete if any nulls | 0.0 | true | any (irrelevant) |
| Never do anything (keep as-is) | 1.0 | **false** | **none** |

---

## Schema Validation

Your IDE should provide autocomplete and validation if it supports JSON Schema. The schema file is located at:
```
transform/cleaning_config.schema.json
```

Reference it in your config with:
```json
{
  "$schema": "./cleaning_config.schema.json",
  ...
}
```
