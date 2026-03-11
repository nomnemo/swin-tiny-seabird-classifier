# Data Loading Times

Benchmarks for loading dataset onto compute instances, for reference and reproducibility.

## S3 to Lambda Cloud Instance

| Date       | Source                                         | Destination        | Instance Type  | Region    | Command      | Wall Clock Time | Notes                          |
|------------|------------------------------------------------|--------------------|----------------|-----------|--------------|-----------------|--------------------------------|
| 2026-03-11 | S3 bucket (`saahil/classification/data/crops/`) | `data/crops/`      | gpu_1x_a10     | us-west-1 | `aws s3 cp --recursive` | **45m 58s**     | 32 species folders             |
