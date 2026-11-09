# Fresh GPU runtime dependencies

PREP previously lived in a separate prepared-input check cell, not Section A only.
Skipping that cell after GPU setup produced the reported NameError. Setup now defines
RAW, CACHE, WORK, PREP and TOKEN_DIR explicitly. The second setup cell includes the
required saved-report/header preflight and sets PREPARED_INPUTS_CHECKED only after
success. The probe rejects incomplete setup before touching model revision files.

Dependency trace: runtime setup provides paths, CU/session environment and imports;
code setup provides REPO, meter, monitored_run and prepared-input readiness;
probe provides revision, extract_args and cost estimates; extraction provides LOCAL;
training profile provides OUT, train_args and training estimates. Later cells now
check their predecessor variables with actionable errors. No Section A globals are
required. On reconnect run both setup cells, then the probe. CPU preparation is not
called. Local tests execute path initialization in an empty namespace and check
setup validation placement, failure guards and compilation. GPU/model/Drive calls
are not executed by those tests; those still require Colab.
