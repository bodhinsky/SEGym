MAX_RETRIES = 5
TIMEOUT_SECONDS = 30
DEFAULT_SAVE_PATH = "./temp"
DOCKER_TAG = "pytest-env"
GIT_APPLY_PATCH = "git apply --ignore-space-change --ignore-whitespace --verbose --recount --inaccurate-eof ./file.patch"
GIT_CHECK_PATCH = "git apply --check --ignore-space-change --ignore-whitespace --verbose --recount --inaccurate-eof ./file.patch"
GIT_DISCARD_CHANGES = "git reset --hard HEAD"
GIT_DIFF = "git diff"
MODEL_NAME = "llama3:8b"
FUZZY_MATCH_THRESHOLD = 0.8
#issue artist Overview as of now
ISSUE_DESC = "Calculate percentage of releases owned per artist. Implement data visualization for artist release percentage. Update data model and storage as needed. Enhance UI with artist release percentage visualization. Ensure secure API communication and data protection. Follow best practices for maintainability, testing, documentation. Optimize performance for data processing and visualization."