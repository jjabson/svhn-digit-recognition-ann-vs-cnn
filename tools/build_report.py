import subprocess
import sys


from config.project_paths import (
    PROJECT_ROOT,
    REPORT_OUTPUT,
    REPORT_SOURCE,
)


def run_python_module(module_name: str) -> None:
    """
    Run a Python module using the current Python interpreter.
    """
    print(f"\nRunning: python -m {module_name}")

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )


def compile_typst_report() -> None:
    """
    Compile the Typst report into a PDF.
    """
    print("\nCompiling Typst report")

    subprocess.run(
        [
            "typst",
            "compile",
            "--root",
            str(PROJECT_ROOT),
            str(REPORT_SOURCE),
            str(REPORT_OUTPUT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    """
    Regenerate all report data, figures, and the final PDF.
    """
    print("Starting SVHN report build")

    run_python_module("tools.create_report_data")
    run_python_module("tools.figures.create_eda_figures")
    run_python_module("tools.figures.create_preprocessing_figures")
    run_python_module("tools.figures.create_model_figures")
    run_python_module("tools.figures.create_evaluation_figures")
    compile_typst_report()

    print(f"\nReport build completed successfully: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()