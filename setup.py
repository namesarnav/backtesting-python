"""Build the optional C++ extension (Phase 7).

    python setup.py build_ext --inplace

Produces `engine/_fastloop.<abi>.so`, which `engine/event_driven.py` picks up
automatically. The extension is *optional* by design: without it the engine
falls back to the pure-Python bar loop and every number in the repo is
unchanged -- only the runtime differs. A clone without a compiler still runs
the full pipeline.

There is no `install_requires` here and this is not a distributable package.
It exists solely to compile one translation unit; dependencies live in
requirements.txt, which is what Docker and CI install.
"""

from setuptools import Extension, setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext = Pybind11Extension(
        "engine._fastloop",
        ["cpp/event_loop.cpp"],
        cxx_std=17,
        # -O3 matters here: the bar loop is the thing being benchmarked, and
        # measuring an unoptimized build would understate the port and make
        # the README's speedup number meaningless.
        extra_compile_args=["-O3"],
    )
    cmdclass = {"build_ext": build_ext}
except ImportError:  # pragma: no cover - only hit if pybind11 is missing
    # Fail with a sentence rather than a traceback about a missing module.
    raise SystemExit(
        "pybind11 is required to build the C++ extension.\n"
        "  pip install -r requirements.txt\n"
        "The extension is optional -- the engine runs without it."
    ) from None

setup(
    name="backtest-fastloop",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass=cmdclass,
    zip_safe=False,
)
