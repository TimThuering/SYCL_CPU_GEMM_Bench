# Third-party header-only dependencies, fetched at configure time.
# Pinned to release tags for reproducibility. Shallow git clones (no release assets needed).

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# nlohmann/json: JSON output
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install    OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0
  GIT_SHALLOW    TRUE
)

# cxxopts: command line parsing
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CXXOPTS_BUILD_TESTS    OFF CACHE INTERNAL "")
set(CXXOPTS_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(CXXOPTS_ENABLE_WARNINGS OFF CACHE INTERNAL "")
FetchContent_Declare(cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.3.1
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(nlohmann_json cxxopts)
