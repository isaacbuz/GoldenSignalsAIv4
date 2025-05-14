# create_project.py
# Purpose: Main script to orchestrate the creation of the GoldenSignalsAI project
# directory, generating all files under the new architecture with improvements for options
# trading. Calls modular part scripts to create files, ensuring modularity and completeness.

import logging
import os
import zipfile
from pathlib import Path

# Configure logging with JSON-like format for consistency
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def zip_directory(directory: Path, zip_path: Path):
    """Zip the specified directory into a zip file.

    Args:
        directory (Path): Directory to zip.
        zip_path (Path): Path for the output zip file.
    """
    logger.info({"message": f"Zipping directory {directory} to {zip_path}"})
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(directory)
                    zipf.write(file_path, arcname)
        logger.info({"message": f"Project zipped successfully at: {zip_path}"})
    except Exception as e:
        logger.error({"message": f"Failed to zip directory: {str(e)}"})


def main():
    """Main function to create the GoldenSignalsAI project directory."""
    logger.info({"message": "Starting project creation"})

    # Import part scripts
    from create_part1 import create_part1
    from create_part2a import create_part2a
    from create_part2b import create_part2b
    from create_part2c import create_part2c
    from create_part3a import create_part3a
    from create_part3b import create_part3b
    from create_part3c import create_part3c
    from create_part3d import create_part3d
    from create_part3e import create_part3e
    from create_part3f import create_part3f
    from create_part3g import create_part3g
    from create_part3h import create_part3h
    from create_part3i import create_part3i
    from create_part4 import create_part4
    from create_part5 import create_part5

    # Create project directory
    project_dir = Path.cwd()
    logger.info({"message": f"Using project directory {project_dir}"})

    # Run all part scripts to create the project structure
    try:
        create_part1()  # Root files
        create_part2a()  # application/ai_service/
        create_part2b()  # application/services/
        create_part2c()  # application/workflows/
        create_part3a()  # domain/
        create_part3b()  # infrastructure/
        create_part3c()  # presentation/
        create_part3d()  # agents/
        create_part3e()  # notifications/
        create_part3f()  # orchestration/
        create_part3g()  # optimization/
        create_part3h()  # governance/
        create_part3i()  # monitoring/
        create_part4()  # .github/workflows/ and k8s/
        create_part5()  # backtesting/

        # Zip the project directory
        zip_path = Path("GoldenSignalsAI_project_new.zip")
        zip_directory(project_dir, zip_path)
        logger.info({"message": "Project creation completed successfully"})
    except Exception as e:
        logger.error({"message": f"Project creation failed: {str(e)}"})
        raise


if __name__ == "__main__":
    main()
