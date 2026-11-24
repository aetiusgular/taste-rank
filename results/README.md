# results/

Run artifacts (BUILD_OUTLINE.md section 6). One JSON object per line in `results/*.jsonl`. Every row carries:

| Field | What |
|---|---|
| `run_id` | unique per run |
| `git_sha` | commit the run used |
| `machine` | read at run time (`nvidia-smi`, `/proc/cpuinfo`, `/proc/meminfo`; on the box, `$FORGE_MACHINE_FILE`), never copied from a notes file |
| `model_revision` | Hugging Face commit or vendor revision; `null` when no model runs |
| `quant` | quantization; `null` when none |
| `runtime` | serving runtime and version; `null` when none |
| `date` | ISO 8601 |

Any number in a README, PR, `DEFENSE.md` or outreach draft points to a row here. Negative results stay.
