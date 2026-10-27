from __future__ import annotations

import logging
import re
import traceback
import docker
import docker.errors
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import random
import requests

from swebench.harness.constants import (
    DOCKER_USER,
    BASE_IMAGE_BUILD_DIR,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    MAP_REPO_VERSION_TO_SPECS,
)
from swebench.harness.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec
)
from swebench.harness.docker_utils import (
    cleanup_container,
    remove_image,
    find_dependent_images
)

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _runtime_proxy_env() -> dict[str, str]:
    """Proxy env vars to inject into *runtime* containers.

    Notes:
    - This affects network calls made inside the running container (tests, pip, etc).
    - It does NOT configure the Docker daemon itself (daemon proxy must be set separately).
    """

    # Prefer explicit env vars from the host process.
    keys = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "no_proxy",
        "NO_PROXY",
    ]
    out: dict[str, str] = {k: v for k in keys if (v := os.environ.get(k))}

    # If the user hasn't provided any proxy vars, fall back to the common local proxy
    # used in this workspace.
    if not any(k in out for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY")):
        default_proxy = os.environ.get("SWEB_FIXED_PROXY", "http://172.19.135.130:5000").strip()
        if default_proxy:
            out.update(
                {
                    "http_proxy": default_proxy,
                    "https_proxy": default_proxy,
                    "HTTP_PROXY": default_proxy,
                    "HTTPS_PROXY": default_proxy,
                }
            )

    # Sane default NO_PROXY to keep GitHub reachable when a global proxy is set.
    default_no_proxy = "github.com,raw.githubusercontent.com,api.github.com"
    if "NO_PROXY" not in out and "no_proxy" not in out:
        out["NO_PROXY"] = default_no_proxy
    return out


# Injected into conda-based env setup scripts to reduce long "Solving environment" stalls.
_CONDA_SOLVER_SNIPPET = """\
conda config --set channel_priority strict || true
conda config --set solver libmamba || true
"""

_PIN_PYTHON_IN_ENV_YAML_SNIPPET = """\
# Ensure the initial solve uses the desired Python version.
if grep -qE '^name:\s*testbed\s*$' environment.yml; then
    python - <<'PY'
from pathlib import Path
import re

p = Path('environment.yml')
lines = p.read_text().splitlines(True)

if not any(re.match(r"^\s*-\s*python\s*=", ln) for ln in lines):
    out = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if (not inserted) and re.match(r"^\s*dependencies:\s*$", ln):
            # NOTE: keep this as a *single* Python string literal. When this snippet
            # is embedded into bash (via heredoc), accidental newlines inside the
            # quotes can lead to SyntaxError.
            out.append("  - python=3.11\n")
            inserted = True
    if not inserted:
        import sys
        sys.exit('failed to find dependencies: line in environment.yml')
    p.write_text(''.join(out))
PY
fi
"""


def _docker_proxy_buildargs() -> dict:
    """Return docker buildargs for proxy env vars.

    We *only* pass through proxy variables if they are set in the host environment.
    This avoids hard-coding a proxy that may break GitHub `git clone` during instance
    builds (e.g., HTTP 503 from proxy after CONNECT).
    """
    keys = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "no_proxy",
        "NO_PROXY",
    ]
    out = {k: v for k in keys if (v := os.environ.get(k))}

    # Provide a sane default NO_PROXY to keep GitHub reachable when the user has
    # a global proxy in their environment. User-provided NO_PROXY still wins.
    default_no_proxy = "github.com,raw.githubusercontent.com,api.github.com"
    if "NO_PROXY" not in out and "no_proxy" not in out:
        out["NO_PROXY"] = default_no_proxy
    return out


def _patch_setup_repo_script(script: str) -> str:
    """Apply small, safe patches to instance setup scripts.

    - Inject proxy exports (user constraint)
    - Remove `--no-use-pep517` which may be unsupported depending on pip version
        - Make editable installs (`pip install -e ...`) robust when build backend
            doesn't support PEP 660 (`build_editable` hook)
    """

    # Ensure proxy is applied inside the build container before any network operation.
    proxy_exports = (
        'export HTTP_PROXY="http://172.19.135.130:5000" && '
        'export HTTPS_PROXY="http://172.19.135.130:5000" && '
        'export http_proxy="http://172.19.135.130:5000" && '
        'export https_proxy="http://172.19.135.130:5000"\n'
    )

    if "172.19.135.130:5000" not in script:
        # Insert after the shebang if present; otherwise prepend.
        if script.startswith("#!/"):
            first_nl = script.find("\n")
            if first_nl != -1:
                script = script[: first_nl + 1] + proxy_exports + script[first_nl + 1 :]
            else:
                script = script + "\n" + proxy_exports
        else:
            script = proxy_exports + script

    if "--no-use-pep517" in script:
        script = script.replace("--no-use-pep517", "")
        script = re.sub(r"[ \t]{2,}", " ", script)

    # Some repos (e.g. pylint) use a build backend that can't do editable installs
    # (missing the PEP 660 `build_editable` hook). In docker build we want a robust
    # install step rather than failing the entire image build.
    #
    # Convert:
    #   python -m pip install -e .
    # into:
    #   python -m pip install -e . || python -m pip install .
    # (and similarly for extras).
    def _editable_fallback(m: re.Match) -> str:
        prefix = m.group("prefix")
        arg = m.group("arg")
        # Preserve trailing newline if present.
        nl = m.group("nl") or ""
        return f"{prefix}{arg} || {prefix}{arg.replace('-e ', '')}{nl}"

    script = re.sub(
        r"(?m)^(?P<prefix>\s*python\s+-m\s+pip\s+install\s+)(?P<arg>-e\s+[^\n#]+)(?P<nl>\n?)$",
        _editable_fallback,
        script,
    )

    return script


def _try_add_local_repo_to_build_context(build_dir: Path, repo: str) -> bool:
    """If a local repo cache exists, copy it into build context.

    We expect a cache like `/data/tianzhao/denoise/swebench/repos/<repo_name>`.
    `repo` is like `matplotlib/matplotlib` or `pallets/flask`.

    Important: the cache directory is keyed by *repo name* (e.g. `flask`), not owner
    (e.g. `pallets`).
    """
    local_root = Path("/data/tianzhao/denoise/swebench/repos")

    owner = ""
    repo_name = repo
    if "/" in repo:
        owner, repo_name = repo.split("/", 1)

    # Prefer repo_name; fall back to owner for the rare cases where the cache may
    # be stored under owner (e.g. `matplotlib/matplotlib` works either way).
    candidates = [repo_name]
    if owner and owner != repo_name:
        candidates.append(owner)

    src = None
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        maybe = local_root / candidate
        if maybe.exists() and maybe.is_dir():
            src = maybe
            break

    if src is None:
        return False

    dst = build_dir / "local_repo"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    try:
        (build_dir / ".local_repo_source").write_text(str(src), encoding="utf-8")
    except Exception:
        pass
    return True


def _patch_setup_repo_script_for_local_repo(script: str, repo_directory: str = "/testbed") -> str:
    """Patch setup_repo.sh to use a pre-copied local repo at `repo_directory`.

    This removes network-dependent `git clone` and (optional) `git remote remove origin`.
    We keep `git reset --hard <sha>` so the commit mismatch is NOT a problem.
    """
    lines = script.splitlines(True)
    out: list[str] = []
    for ln in lines:
        if ln.lstrip().startswith("git clone "):
            continue
        # Some scripts do `cd /testbed` already; keep it.
        if ln.strip() == "git remote remove origin":
            continue
        out.append(ln)
    script2 = "".join(out)

    # Ensure git commands run inside the copied repo.
    # When we remove `git clone`, some setup scripts no longer `cd` into the repo.
    if not re.search(r"(?m)^\s*cd\s+" + re.escape(repo_directory) + r"\s*$", script2):
        insert_at = 0
        if script2.startswith("#!"):
            nl = script2.find("\n")
            insert_at = 0 if nl == -1 else nl + 1

        m = re.search(r"(?m)^\s*set\s+-[^\n]*\n", script2)
        if m:
            insert_at = m.end()

        script2 = script2[:insert_at] + f"cd {repo_directory}\n" + script2[insert_at:]

    return script2


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(instance_id: str, log_file: Path, mode="w"):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
        image_name: str,
        setup_scripts: dict,
        dockerfile: str,
        platform: str,
        client: docker.DockerClient,
        build_dir: Path,
        nocache: bool = False
    ):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
    # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            # If this looks like a conda env creation script, inject a faster/more stable solver.
            if setup_script_name == "setup_env.sh" and "conda env create" in setup_script:
                if "solver libmamba" not in setup_script and "channel_priority" not in setup_script:
                    setup_script = setup_script.replace(
                        "set -euxo pipefail\n",
                        "set -euxo pipefail\n" + _CONDA_SOLVER_SNIPPET,
                        1,
                    )

                # Pin python version inside environment.yml before creating env to avoid
                # later conflicts when attempting to downgrade Python.
                # Do this even if the solver snippet was already injected in a previous run.
                if "environment.yml" in setup_script and "- python=3.11" not in setup_script:
                    setup_script = setup_script.replace(
                        "conda env create --file environment.yml\n",
                        _PIN_PYTHON_IN_ENV_YAML_SNIPPET + "\nconda env create --file environment.yml\n",
                        1,
                    )

                # If the script later tries to install python=3.11 again, it may conflict with
                # other deps solved for a different Python. Once we pin python in environment.yml,
                # this extra step is unnecessary.
                setup_script = setup_script.replace(
                    "conda activate testbed && conda install python=3.11 -y\n",
                    "conda activate testbed\n",
                )

            # Instance image setup scripts: patch common incompatibilities.
            if setup_script_name == "setup_repo.sh":
                setup_script = _patch_setup_repo_script(setup_script)

            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        logger.info(
            f"Building docker image {image_name} in {build_dir} with platform {platform}"
        )

        # Network mode during docker build.
        # Default to host network to avoid common DNS issues inside Docker build containers
        # (e.g., `Temporary failure in name resolution` when conda/pip fetching repodata).
        # You can override with env var SWEB_DOCKER_BUILD_NETWORK:
        #   - "host" (default)
        #   - "bridge" (Docker default)
        #   - "none" / other supported modes
        build_network = os.environ.get("SWEB_DOCKER_BUILD_NETWORK", "host").strip() or "host"
        logger.info(f"Using docker build network mode: {build_network}")

        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
            network_mode=build_network,
            buildargs=_docker_proxy_buildargs(),
        )

        # 去掉内部进度条：只计数并写日志，避免在终端输出大量 [build] 行
        buildlog = ""
        for chunk in response:
            if "stream" in chunk:
                chunk_stream = ansi_escape.sub("", chunk["stream"])
                logger.info(chunk_stream.strip())
                buildlog += chunk_stream
            elif "errorDetail" in chunk:
                logger.error(
                    f"Error: {ansi_escape.sub('', chunk['errorDetail']['message'])}"
                )
                raise docker.errors.BuildError(
                    chunk["errorDetail"]["message"], buildlog
                )

        logger.info("Image built successfully!")
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)  # functions that create loggers should close them



def build_base_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False
    ):
    """
    Builds the base images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    # Get the base images to build from the dataset
    test_specs = get_test_specs_from_dataset(dataset)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }
    if force_rebuild:
        for key in base_images:
            remove_image(client, key, "quiet")

    # Build the base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            # Check if the base image already exists
            client.images.get(image_name)
            if force_rebuild:
                # Remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        except docker.errors.ImageNotFound:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")


def get_env_configs_to_build(
        client: docker.DockerClient,
        dataset: list,
    ):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    test_specs = get_test_specs_from_dataset(dataset)

    for test_spec in test_specs:
        # Check if the base image exists
        try:
            if test_spec.base_image_key not in base_images:
                base_images[test_spec.base_image_key] = client.images.get(
                    test_spec.base_image_key
                )
            base_image = base_images[test_spec.base_image_key]
        except docker.errors.ImageNotFound:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                "Please build the base images first."
            )

        # Check if the environment image exists
        image_exists = False
        try:
            env_image = client.images.get(test_spec.env_image_key)
            image_exists = True

            if env_image.attrs["Created"] < base_image.attrs["Created"]:
                # Remove the environment image if it was built after the base_image
                for dep in find_dependent_images(client, test_spec.env_image_key):
                    # Remove instance images that depend on this environment image
                    remove_image(client, dep, "quiet")
                remove_image(client, test_spec.env_image_key, "quiet")
                image_exists = False
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts


def build_env_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False,
    max_workers: int = 4
    ):
    """
    Builds the environment images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Get the environment images to build from the dataset
    if force_rebuild:
        env_image_keys = {x.env_image_key for x in get_test_specs_from_dataset(dataset)}
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    build_base_images(client, dataset, force_rebuild)
    configs_to_build = get_env_configs_to_build(client, dataset)
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return [], []
    print(f"Total environment images to build: {len(configs_to_build)}")

    # Building conda env images can spend a long time in dependency solving; running many in
    # parallel often makes it slower or causes solver to thrash.
    # Default to serial env builds; override by setting SWEB_ENV_BUILD_WORKERS.
    try:
        max_workers = int(os.environ.get("SWEB_ENV_BUILD_WORKERS", "1"))
    except ValueError:
        max_workers = 1

    # Build the environment images
    successful, failed = list(), list()
    with tqdm(
        total=len(configs_to_build), smoothing=0, desc="Building environment images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = dict()
            for image_name, config in configs_to_build.items():
                # 安全地在 FROM 之后插入 ENV/ARG
                lines = config["dockerfile"].splitlines()
                from_line_index = None
                for i, line in enumerate(lines):
                    if line.strip().upper().startswith("FROM"):
                        from_line_index = i
                        break
                if from_line_index is None:
                    raise ValueError(f"No FROM line found in Dockerfile for {image_name}")

                # 构建新的 Dockerfile
                dockerfile_lines = (
                    lines[:from_line_index + 1] +
                    lines[from_line_index + 1:]
                )
                config["dockerfile"] = "\n".join(dockerfile_lines)

                # 提交 build_image 任务
                futures[executor.submit(
                    build_image,
                    image_name,
                    {"setup_env.sh": config["setup_script"]},
                    config["dockerfile"],
                    config["platform"],
                    client,
                    ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
                )] = image_name

            # 等待每个 future 完成
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                except Exception:
                    print("Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])

    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    return successful, failed




def build_instance_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4
    ):
    """
    Builds the instance images required for the dataset if they do not already exist.

    Args:
        dataset (list): List of test specs or dataset to build images for
        client (docker.DockerClient): Docker client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Build environment images (and base images as needed) first
    test_specs = list(map(make_test_spec, dataset))
    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")
    _, env_failed = build_env_images(client, test_specs, force_rebuild, max_workers)

    if len(env_failed) > 0:
        # Don't build images for instances that depend on failed-to-build env images
        dont_run_specs = [spec for spec in test_specs if spec.env_image_key in env_failed]
        test_specs = [spec for spec in test_specs if spec.env_image_key not in env_failed]
        print(f"Skipping {len(dont_run_specs)} instances - due to failed env image builds")
    print(f"Building instance images for {len(test_specs)} instances")
    successful, failed = list(), list()

    # Build the instance images
    with tqdm(
        total=len(test_specs), smoothing=0, desc="Building instance images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each image to build
            futures = {
                executor.submit(
                    build_instance_image,
                    test_spec,
                    client,
                    None,  # logger is created in build_instance_image, don't make loggers before you need them
                    False,
                ): test_spec
                for test_spec in test_specs
            }

            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if image built successfully
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception:
                    print("Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue

    # Show how many images failed to build
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_image(
        test_spec: TestSpec,
        client: docker.DockerClient,
        logger: logging.Logger|None,
        nocache: bool,
    ):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """
    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key
    dockerfile = test_spec.instance_dockerfile

    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(env_image_name)
    except docker.errors.ImageNotFound as e:
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment image {env_image_name} not found for {test_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name} for {test_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        instance_image = client.images.get(image_name)
        if instance_image.attrs["Created"] < env_image.attrs["Created"]:
            # the environment image is newer than the instance image, meaning the instance image may be outdated
            remove_image(client, image_name, "quiet")
            image_exists = False
        else:
            image_exists = True
    except docker.errors.ImageNotFound:
        pass

    # Prefer using a local repo cache to avoid network access during docker build.
    # User has `/data/tianzhao/denoise/swebench/repos/<repo_name>` already cloned.
    use_local_repo = _try_add_local_repo_to_build_context(build_dir, test_spec.repo)
    if use_local_repo:
        logger.info(
            f"Using local repo cache for {test_spec.repo} (copied into build context as ./local_repo)"
        )
    else:
        logger.info(
            f"No local repo cache found for {test_spec.repo}; falling back to network clone during build"
        )
    if use_local_repo:
        dockerfile = dockerfile.replace(
            "COPY ./setup_repo.sh /root/\nRUN /bin/bash /root/setup_repo.sh\n",
            "COPY ./local_repo /testbed\nCOPY ./setup_repo.sh /root/\nRUN /bin/bash /root/setup_repo.sh\n",
            1,
        )

    def _cleanup_local_repo_copy() -> None:
        """Remove the copied repo from the docker build context to save disk.

        We always copy fresh from `/data/tianzhao/denoise/swebench/repos` when needed,
        so keeping the build-context copy provides little value and consumes a lot of space.
        """
        if not use_local_repo:
            return
        try:
            shutil.rmtree(build_dir / "local_repo")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to cleanup local_repo in build context: {e}")
        try:
            (build_dir / ".local_repo_source").unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # Build the instance image
    if not image_exists:
        try:
            build_image(
                image_name=image_name,
                setup_scripts={
                    "setup_repo.sh": (
                        _patch_setup_repo_script_for_local_repo(test_spec.install_repo_script)
                        if use_local_repo
                        else test_spec.install_repo_script
                    ),
                },
                dockerfile=dockerfile,
                platform=test_spec.platform,
                client=client,
                build_dir=build_dir,
                nocache=nocache,
            )
        finally:
            _cleanup_local_repo_copy()
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")
        _cleanup_local_repo_copy()

    if new_logger:
        close_logger(logger)


def build_container(
        test_spec: TestSpec,
        client: docker.DockerClient,
        run_id: str,
        logger: logging.Logger,
        nocache: bool,
        force_rebuild: bool = False
    ):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    build_instance_image(test_spec, client, logger, nocache)

    container = None
    try:
        # Get configurations for how container should be created
        config = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
        user = DOCKER_USER if not config.get("execute_test_as_nonroot", False) else "nonroot"
        nano_cpus = config.get("nano_cpus")

        # Create the container.
        # When the daemon is busy (image layer unpacking / overlay2 pressure), container
        # creation can take longer than the docker SDK default request timeout (60s).
        # We retry a few times with backoff.
        logger.info(f"Creating container for {test_spec.instance_id}...")
        container = None
        for attempt in range(1, 4):
            try:
                container = client.containers.create(
                    image=test_spec.instance_image_key,
                    name=test_spec.get_instance_container_name(run_id),
                    user=user,
                    detach=True,
                    command="tail -f /dev/null",
                    nano_cpus=nano_cpus,
                    platform=test_spec.platform,
                    environment=_runtime_proxy_env(),
                )
                break
            except requests.exceptions.ReadTimeout as e:
                if attempt >= 3:
                    raise
                sleep_s = (5 * attempt) + random.uniform(0, 2)
                logger.warning(
                    f"Docker API timeout while creating container for {test_spec.instance_id} "
                    f"(attempt {attempt}/3). Sleeping {sleep_s:.1f}s then retrying. Error: {e}"
                )
                time.sleep(sleep_s)

        assert container is not None
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e
