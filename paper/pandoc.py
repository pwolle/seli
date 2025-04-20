import argparse
import logging
import os
import subprocess
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_mod_time(filepath: str) -> float | None:
    """Get the last modification time of a file."""
    try:
        return os.path.getmtime(filepath)
    except FileNotFoundError:
        logger.error(f"File not found - {filepath}")
        return None


def run_pandoc(
    md_file: str,
    template: str,
    bib: str,
    output: str,
    engine: str,
) -> subprocess.Popen | None:
    """Run the Pandoc compilation command in the background."""
    command = [
        "pandoc",
        md_file,
        "--template",
        template,
        "--bibliography",
        bib,
        "--citeproc",
        "--pdf-engine",
        engine,
        "-o",
        output,
    ]
    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"Started compilation process with PID: {process.pid}")
        return process
    except FileNotFoundError:
        logger.error(
            "'pandoc' command not found. "
            "Make sure Pandoc is installed and in your PATH."
        )
        return None
    except Exception as e:
        logger.error(f"An error occurred starting the compilation: {e}")
        return None


def _handle_process_output(proc: subprocess.Popen) -> int:
    logger.info(f"Waiting for compilation process (PID: {proc.pid}) to finish")
    stdout, stderr = proc.communicate()
    logger.info(
        f"Compilation finished with exit code {proc.returncode}.",
    )
    if stdout:
        logger.info("--- stdout ---")
        for line in stdout.strip().splitlines():
            logger.info(line)
        logger.info("--- end stdout ---")

    if stderr:
        logger.warning("--- stderr ---")
        for line in stderr.strip().splitlines():
            logger.warning(line)
        logger.warning("--- end stderr ---")

    return proc.returncode


def _check_and_handle_file_change(
    last_mod_time: float | None,
    current_process: subprocess.Popen | None,
    markdown_file: str,
    template_file: str,
    bib_file: str,
    output_pdf: str,
    pdf_engine: str,
) -> tuple[subprocess.Popen | None, float | None]:
    new_process = current_process
    updated_last_mod_time = last_mod_time

    current_mod_time = get_mod_time(markdown_file)

    if current_mod_time is not None and current_mod_time != last_mod_time:
        logger.info(f"Detected change in {markdown_file}.")
        updated_last_mod_time = current_mod_time

        if current_process and current_process.poll() is None:
            _handle_process_output(current_process, description=" previous")
            new_process = None

        if new_process is None:
            logger.info("Starting new compilation...")
            new_process = run_pandoc(
                markdown_file,
                template_file,
                bib_file,
                output_pdf,
                pdf_engine,
            )
            if new_process is None:
                updated_last_mod_time = None

    return new_process, updated_last_mod_time


def watch_and_compile(
    markdown_file: str,
    template_file: str,
    bib_file: str,
    output_pdf: str,
    pdf_engine: str,
    check_interval_seconds: float,
) -> None:
    last_mod_time = get_mod_time(markdown_file)

    if last_mod_time is None:
        logger.error(f"Could not get modification time for {markdown_file}.")
        return

    current_process: subprocess.Popen | None = None
    logger.info(f"Watching {markdown_file} for changes...")

    while True:
        try:
            if current_process and current_process.poll() is not None:
                _handle_process_output(current_process)
                current_process = None

            new_process, last_mod_time = _check_and_handle_file_change(
                last_mod_time,
                current_process,
                markdown_file,
                template_file,
                bib_file,
                output_pdf,
                pdf_engine,
            )
            if new_process is not current_process:
                current_process = new_process

            time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nStopping watcher.")

            if current_process and current_process.poll() is None:
                logger.info(
                    f"Terminating running compilation process "
                    f"(PID: {current_process.pid})..."
                )
                current_process.terminate()

                try:
                    stdout, stderr = current_process.wait(timeout=5)
                    logger.info("Process terminated.")

                    if stdout:
                        logger.info("--- final stdout ---")
                        for line in stdout.strip().splitlines():
                            logger.info(line)
                        logger.info("--- end final stdout ---")

                    if stderr:
                        logger.warning("--- final stderr ---")
                        for line in stderr.strip().splitlines():
                            logger.warning(line)
                        logger.warning("--- end final stderr ---")

                except subprocess.TimeoutExpired:
                    logger.error("Process did not terminate, killing.")
                    current_process.kill()

                except Exception as e:
                    logger.error(f"Error during termination: {e}")
            break

        except Exception as e:
            logger.error(
                f"An unexpected error occurred in the watch loop: {e}",
                exc_info=True,
            )
            time.sleep(check_interval_seconds * 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile Markdown file to PDF using Pandoc upon change."
    )
    parser.add_argument(
        "--markdown-file",
        default="paper.md",
        help="Path to the input Markdown file.",
    )
    parser.add_argument(
        "--template-file",
        default="template.tex",
        help="Path to the Pandoc LaTeX template file.",
    )
    parser.add_argument(
        "--bib-file",
        default="references.bib",
        help="Path to the bibliography file.",
    )
    parser.add_argument(
        "--output-pdf",
        default="paper.pdf",
        help="Path for the output PDF file.",
    )
    parser.add_argument(
        "--pdf-engine",
        default="pdflatex",
        help="PDF engine for Pandoc (e.g., pdflatex, xelatex, lualatex).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Check interval in seconds for file changes.",
    )

    args = parser.parse_args()

    watch_and_compile(
        markdown_file=args.markdown_file,
        template_file=args.template_file,
        bib_file=args.bib_file,
        output_pdf=args.output_pdf,
        pdf_engine=args.pdf_engine,
        check_interval_seconds=args.interval,
    )
