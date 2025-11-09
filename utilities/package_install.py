import subprocess
import os
import sys
import logging
import importlib.util
import re
 
# get python version
venv_python = sys.executable
python_version = sys.version_info

APP_ROOT = os.getenv("APP_ROOT", "/app")

# add the path to the local pip modules
pip_modules_paths = []
pip_modules_paths.append(f"{APP_ROOT}/pip_modules/bin")
pip_modules_paths.append(f"{APP_ROOT}/pip_modules/lib/python{python_version.major}/site-packages")
pip_modules_paths.append(f"{APP_ROOT}/pip_modules/lib/python{python_version.major}/dist-packages")
pip_modules_paths.append(f"{APP_ROOT}/pip_modules/lib/python{python_version.major}.{python_version.minor}/site-packages")
pip_modules_paths.append(f"{APP_ROOT}/pip_modules/lib/python{python_version.major}.{python_version.minor}/dist-packages")

for path in pip_modules_paths:
    if path not in sys.path:
        sys.path.append(path)

# also append to PYTHONPATH
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ":" + ":".join(pip_modules_paths)

# also append bin path to PATH
os.environ['PATH'] = os.environ.get('PATH', '') + ":" + pip_modules_paths[0]

logging.basicConfig(level=logging.INFO)

def install(package):
    # parse name and version (==, >=, etc.) from for example transformers>=4.45.1, transformers==4.45.1, transformers
    package_name = None
    package_condition = None
    package_version = None

    match = re.match(r"([a-zA-Z0-9_-]+)([<>=!]+)([0-9.]+)?", package)
    if match is not None:
        package_name = match.group(1)
        package_condition = match.group(2)
        package_version = match.group(3)
    else:
        package_name = package
    spec = importlib.util.find_spec(package_name)
    # matched is only when the package is already installed with the satisfying version
    matched = False
    if spec is not None:
        matched = True
        if package_condition is not None:
            matched = False
            # load module and check version
            module = importlib.import_module(package_name)
            spec.version = module.__version__
            if package_version is not None:
                if package_condition == "==":
                    matched = spec.version == package_version
                elif package_condition == ">=":
                    matched = spec.version >= package_version
                elif package_condition == "<=":
                    matched = spec.version <= package_version
                elif package_condition == ">":
                    matched = spec.version > package_version
                elif package_condition == "<":
                    matched = spec.version < package_version
                elif package_condition == "!=":
                    matched = spec.version != package_version

    if not matched:
        logging.warning(f"Installing: {package}")
        subprocess.check_call([venv_python, "-m", "pip", "install", package, "--prefix", f"{APP_ROOT}/pip_modules"])
