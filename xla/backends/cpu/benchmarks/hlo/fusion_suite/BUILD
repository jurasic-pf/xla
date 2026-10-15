# HLO modules from JAX scientific-computing and ML libraries, used to evaluate
# XLA:CPU fusion and cost-model changes. See README.md.

load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = CPU_BENCHMARKS_VISIBILITY,
    licenses = ["notice"],
)

FUSION_SUITE_HLO_FILES = glob(["*/*.hlo"])

exports_files(FUSION_SUITE_HLO_FILES)

filegroup(
    name = "hlo_files",
    srcs = FUSION_SUITE_HLO_FILES,
)
