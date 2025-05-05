import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of scripts to run in order, now in the build_scripts/ folder
scripts = [
    "build_scripts/build_root_files.py",
    "build_scripts/build_presentation_files.py",
    "build_scripts/build_application_files.py",
    "build_scripts/build_domain_files.py",
    "build_scripts/build_infrastructure_files.py",
    "build_scripts/build_docs_files.py"
]

for script in scripts:
    logger.info(f"Running {script}...")
    try:
        result = subprocess.run(["python", script], capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script}: {e}")
        logger.error(e.stderr)
        break
    except FileNotFoundError as e:
        logger.error(f"Script {script} not found: {e}")
        break

logger.info("Project assembly complete.")