"""
Data splitter for the 17-class subset (excludes MTRN, MEGRT, OTHERS, OTHR).
Reuses all logic from DataSplitter, just points to the filtered JSON.
"""
from pathlib import Path
from scripts.DataSplitter import get_data_splits, _print_split_summary, _write_csv_from_list_of_dicts

DEFAULT_JSON = Path("data/metadata_17classes.json")
DEFAULT_TRAIN_CSV = DEFAULT_JSON.parent / "split_train.csv"
DEFAULT_VAL_CSV   = DEFAULT_JSON.parent / "split_val.csv"
DEFAULT_TEST_CSV  = DEFAULT_JSON.parent / "split_test.csv"

if __name__ == "__main__":
    train_records, val_records, test_records, info = get_data_splits(json_path=DEFAULT_JSON)
    _print_split_summary(info)

    _write_csv_from_list_of_dicts(DEFAULT_TRAIN_CSV, train_records)
    _write_csv_from_list_of_dicts(DEFAULT_VAL_CSV, val_records)
    _write_csv_from_list_of_dicts(DEFAULT_TEST_CSV, test_records)

    print(
        "[done] wrote:\n"
        f"  {DEFAULT_TRAIN_CSV}\n"
        f"  {DEFAULT_VAL_CSV}\n"
        f"  {DEFAULT_TEST_CSV}"
    )
