# Fresh-runtime resource initialization

BudgetMeter/runtime_profile/run_logged were imported only in cell 1 but referenced
in resource cell 2. They are now explicitly imported in cell 2 before use. Cell 1
sets TRAINING_PREFLIGHT_OK only after checkout and CPU preflight succeed. Cell 2
requires that marker and sets RESOURCE_SETUP_OK only on completion. Staging imports
its own tempfile/shutil/Path and requires resource setup; profiling requires completed
staging. Starting a prerequisite cell clears its success flag to prevent failed
reruns being mistaken for success.

Audit: cell 1 owns REPO/CONFIG/OUT and verified Drive paths; resource setup owns
meter/monitored/CU environment; staging owns LOCAL; profile uses those after guards.
The runner stays pinned at da4ffe964869a0c368a9b2136b575eca221dc41e so existing
checkpoint code hashes remain compatible. No prepared arrays or extraction change.

Regression executes the entire resource cell with a fresh namespace containing
only preflight success and repository path, mocked GPU/Drive/subprocess and temporary
output paths. No preexisting BudgetMeter symbol is supplied. Out-of-order cells
fail with actionable assertions before external work. These are software tests,
not an actual L4 or private-Drive execution.
