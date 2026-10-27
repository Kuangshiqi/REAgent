# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import argparse
import io
import re
import os
import json
import shutil
import subprocess
import tarfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import time
from docker import DockerClient, from_env
from docker.errors import ImageNotFound
from docker.models.containers import Container
from tqdm import tqdm

from .utils import BENCHMARK_CONFIG, docker_exec


class BenchmarkEvaluation:
    """
    Main class for running experiments and evaluations.
    Handles Docker image management, environment preparation, patch generation, and evaluation.
    """

    def __init__(
        self,
        benchmark: str,
        working_dir: str,
        trae_config_file_name: str,
        ite: int = 0,
        dataset: str = "SWE-bench_Pro",
        docker_env_config: str = "",
        benchmark_harness_path: str = "/app",
        run_id: str = "trae-agent",
        max_workers: int = 4,
        instance_ids: list[str] | None = None,
    ):
        """
        Initialize the BenchmarkEvaluation class.

        Args:
            benchmark: Benchmark name.
            working_dir: Path for workspace (used for temp files and artifacts).
            trae_config_file_name: Path to Trae config file.
            ite: Iteration index for selecting external requirement/feedback resources.
            dataset: Dataset name.
            docker_env_config: Path to Docker environment config file.
            benchmark_harness_path: Path to benchmark harness (for evaluation).
            run_id: Unique run identifier.
            max_workers: Maximum number of parallel workers.
            instance_ids: List of instance IDs to run (optional).
        """
        assert benchmark in BENCHMARK_CONFIG, f"Invalid benchmark name: {benchmark}"
        self.config = BENCHMARK_CONFIG[benchmark]
        self.dataset_name = dataset
        assert self.dataset_name in self.config.valid_datasets, (
            f"Invalid dataset name: {self.dataset_name}"
        )

        self.benchmark = benchmark
        self.ite = ite
        self.dataset = self.config.load_dataset(self.dataset_name)
        self.docker_client: DockerClient = from_env()
        self.image_status: dict[Any, Any] = {}

        self.working_dir = Path(working_dir)
        self.benchmark_harness_path = benchmark_harness_path
        self.run_id = run_id
        self.max_workers = max_workers
        if instance_ids is None:
            self.instance_ids = [instance["instance_id"] for instance in self.dataset]
        else:
            self.instance_ids = instance_ids

        if docker_env_config != "":
            with open(docker_env_config, "r") as f:
                self.docker_env_config: dict[str, dict[str, str]] = json.load(f)
        else:
            self.docker_env_config = {}

        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.trae_config_file_name = trae_config_file_name
        shutil.copyfile(self.trae_config_file_name, self.working_dir / "trae_config.yaml")

        self.results_dir = Path("results")
        self.task_id = f"{self.benchmark}_{self.dataset_name}_{self.run_id}".replace("/", "_")
        self.task_results_dir = self.results_dir / self.task_id
        self.task_results_dir.mkdir(parents=True, exist_ok=True)

    def _image_name(self, instance_id: str) -> str:
        """
        完全使用本地 sweap 镜像，不再使用 benchmark 默认镜像规则
        """
        if not instance_id.startswith("instance_"):
            raise ValueError(f"Invalid instance_id: {instance_id}")

        body = instance_id[len("instance_"):]
        if "__" not in body:
            raise ValueError(f"Invalid instance_id: {instance_id}")

        repo_base, rest = body.split("__", 1)
        if "-" not in rest:
            raise ValueError(f"Invalid instance_id: {instance_id}")

        repo_name, hash_part = rest.split("-", 1)

        tag = f"{repo_base}.{repo_name}-{repo_base}__{repo_name}-{hash_part}"
        
        if len(tag) > 128:
            tag = tag[:128]

        return f"jefzda/sweap-images:{tag}"

    def _retry_action(self, action, attempts: int = 3, delay: float = 1.0, *,
                      check_success=None):
        """
        Generic retry helper. action is a zero-arg callable. If check_success is
        provided it's called with the action result and should return True on
        success; otherwise any exception is treated as failure and will be retried.
        """
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                result = action()
                if check_success is None:
                    return result
                try:
                    if check_success(result):
                        return result
                    else:
                        last_exc = RuntimeError("check_success returned False")
                except Exception as e:
                    last_exc = e
            except Exception as e:
                last_exc = e
            if attempt < attempts:
                time.sleep(delay)
        # all attempts failed
        raise last_exc

    def prepare_trae_agent(self):
        """
        Build Trae Agent and UV inside a base Ubuntu container.
        Save built artifacts to workspace for later use in experiment containers.
        """
        tars = ["trae-agent.tar", "python-musl.tar.gz", "python-gnu.tar.gz",]
        all_exist = all((self.working_dir / tar).exists() for tar in tars)
        local_python_musl = Path("python/python-musl.tar.gz")
        local_python_gnu  = Path("python/python-gnu.tar.gz")

        if not local_python_musl.exists() or not local_python_gnu.exists():
            raise RuntimeError("Missing local python musl/gnu tarballs")
        
        try:
            image = self.docker_client.images.get("ubuntu:22.04")
        except:
            raise RuntimeError("ubuntu:22.04 not found locally, please docker load it first")

        repo_root_path = Path(__file__).parent.parent

        container = self.docker_client.containers.run(
            image=image,
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.absolute().as_posix(): {"bind": "/trae-workspace", "mode": "rw"},
                repo_root_path.absolute().as_posix(): {"bind": "/trae-src", "mode": "ro"},
            },
        )
        shutil.copy(local_python_musl, self.working_dir / "python-musl.tar.gz")
        shutil.copy(local_python_gnu,  self.working_dir / "python-gnu.tar.gz")

        # If artifacts exist: only install pyerfa then exit
        # if all_exist:

        #     container.stop()
        #     container.remove()
        #     return

        build_commands = [
            "apt-get update && apt-get install -y curl",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "rm -rf /trae-workspace/trae-agent && mkdir /trae-workspace/trae-agent",
            "cp -r -t /trae-workspace/trae-agent/ /trae-src/trae_agent /trae-src/.python-version /trae-src/pyproject.toml /trae-src/uv.lock /trae-src/README.md",
            "cd /trae-workspace/trae-agent && $HOME/.local/bin/uv sync",
            "export PATH=$PATH:$HOME/.local/bin",
            "cd /trae-workspace/trae-agent && source .venv/bin/activate && $HOME/.local/bin/uv pip install docker",
            "cd /trae-workspace/trae-agent && source .venv/bin/activate && $HOME/.local/bin/uv pip install datasets",
            "cd /trae-workspace/trae-agent && source .venv/bin/activate && $HOME/.local/bin/uv pip install pyerfa",
            "cd /trae-workspace/trae-agent && source .venv/bin/activate && $HOME/.local/bin/uv pip install pexpect",
        ]

        for command in tqdm(build_commands, desc="Building trae-agent inside base Docker container"):
            new_command = f'/bin/bash -c "{command}"'
            rc, out = docker_exec(container, new_command)
            if rc != 0:
                print(f"[ERROR] command failed: {command}")
                print(f"[ERROR] return_code: {rc}")
                print(f"[ERROR] output:\n{out}")
                container.stop()
                container.remove()
                raise RuntimeError("build trae-agent failed")

        for tar_name, src_path in [
            ("trae-agent.tar", "/trae-workspace/trae-agent"),
        ]:
            with open(self.working_dir / tar_name, "wb") as f:
                bits, _ = container.get_archive(src_path)
                for chunk in bits:
                    f.write(chunk)

        container.stop()
        container.remove()

    def prepare_experiment_container(self, instance: dict[str, str]) -> Container:
        image_name = self._image_name(instance["instance_id"])
        instance_result_dir = self.task_results_dir / instance["instance_id"]
        instance_result_dir.mkdir(parents=True, exist_ok=True)

        # 写入 problem_statement.txt 到挂载目录 /instance-data
        self.config.problem_statement(instance, instance_result_dir, self.ite)

        # 关键：覆盖 entrypoint，PID1 直接 sleep，容器不会因为 openhands entrypoint 报参而退出
        container: Container = self.docker_client.containers.run(
            image_name,
            entrypoint="/bin/sh",
            command=["-c", "sleep 3600"],
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                instance_result_dir.absolute().as_posix(): {"bind": "/instance-data", "mode": "rw"},
            },
            working_dir="/trae-workspace",
            environment=self.docker_env_config.get("experiment_env", None),
        )

        # 把产物 tar 和 config 放进容器 /trae-workspace（不解压、不执行）
        for fname in [
            "trae-agent.tar",
            "python-musl.tar.gz",
            "python-gnu.tar.gz",
            "trae_config.yaml",
        ]:
            src_path = self.working_dir / fname
            if not src_path.exists():
                raise FileNotFoundError(f"Missing artifact: {src_path}")

            # 制作一个只包含该文件的 tar 流，然后 put_archive 到 /trae-workspace
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(src_path.as_posix(), arcname=fname)
            tar_stream.seek(0)
            container.put_archive("/trae-workspace", tar_stream.getvalue())

        setup_commands = [
            "tar xf trae-agent.tar",
            "mkdir -p /opt/python-musl && tar -xzf python-musl.tar.gz -C /opt/python-musl --strip-components=1",
            "mkdir -p /opt/python-gnu  && tar -xzf python-gnu.tar.gz  -C /opt/python-gnu  --strip-components=1",
        ]
        for command in setup_commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                try:
                    self._retry_action(lambda: docker_exec(container, new_command), check_success=lambda r: r is not None and r[0] == 0)
                except Exception as e:
                    print(f"Docker exec error when running setup command after retries: {e}")
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break

        # print(f"[DEBUG] container created: id={container.id} name={container.name} image={image_name}")
        # print(f"[DEBUG] files copied to /trae-workspace: trae-agent.tar uv.tar uv_shared.tar trae_config.yaml")
        # print(f"[DEBUG] enter: docker exec -it {container.id} /bin/bash")
        # time.sleep(3500)

        return container


    def run_one_instance(self, instance_id: str):
        """
        Run patch generation for a single instance.
        All outputs are written directly to the mounted results directory.
        Args:
            instance_id: Instance identifier.
        """
        instance = next((inst for inst in self.dataset if inst["instance_id"] == instance_id), None)
        if instance is None:
            print(f"Instance {instance_id} not found.")
            return

        working_dir = self.config.working_dir(instance_id)

        container_problem_statement_path = "/instance-data/problem_statement.txt"
        container_patch_file_path = f"/instance-data/{instance_id}.patch"
        container_traj_path = f"/instance-data/{instance_id}.json"

        container = self.prepare_experiment_container(instance)

        setup_commands_musl = (
            "unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL && "
            "rm -f ~/.pip/pip.conf ~/.config/pip/pip.conf /etc/pip.conf 2>/dev/null || true && "
            "export PATH=/opt/python-musl/bin:$PATH && "
            "rm -rf /trae-workspace/trae-agent/.venv && "
            "/opt/python-musl/bin/python3.12 -m venv --copies /trae-workspace/trae-agent/.venv && "
            ". /trae-workspace/trae-agent/.venv/bin/activate && "
            "pip install --upgrade pip setuptools wheel && "
            "pip install -e /trae-workspace/trae-agent && "
            "pip install docker pexpect pyerfa datasets"
        )

        setup_commands_glibc = (
            "unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL && "
            "rm -f ~/.pip/pip.conf ~/.config/pip/pip.conf /etc/pip.conf 2>/dev/null || true && "
            "export PATH=/opt/python-gnu/bin:$PATH && "
            "rm -rf /trae-workspace/trae-agent/.venv && "
            "/opt/python-gnu/bin/python3.12 -m venv --copies /trae-workspace/trae-agent/.venv && "
            ". /trae-workspace/trae-agent/.venv/bin/activate && "
            "python -m pip install --upgrade pip setuptools wheel && "
            "python -m pip install -e /trae-workspace/trae-agent && "
            "python -m pip install docker pexpect pyerfa datasets"
        )
        try:
            probe_cmd = r'''
            if [ -e /lib/ld-musl-x86_64.so.1 ]; then
            echo MUSL
            elif [ -e /lib64/ld-linux-x86-64.so.2 ] || [ -e /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 ]; then
            echo GLIBC
            else
            ldd --version 2>&1 | grep -qi musl && echo MUSL || echo GLIBC
            fi
            '''
            rc, probe_out = docker_exec(container, f"/bin/bash -lc '{probe_cmd}'")
            probe_out = (probe_out or "").strip().upper()
            if "MUSL" in probe_out:
                print("Musl Docker!")
                chosen_setup = setup_commands_musl
            elif "GLIBC" in probe_out:
                print("Glibc Docker!")
                chosen_setup = setup_commands_glibc
            else:
                chosen_setup = setup_commands_musl

            new_command = f'/bin/bash -c "{chosen_setup}"'
            try:
                rc_out = self._retry_action(
                    lambda: docker_exec(container, new_command),
                    check_success=lambda r: r is not None and r[0] == 0,
                )
                return_code, output = rc_out
            except Exception as e:
                return_code, output = None, str(e)
            if return_code is None or return_code != 0:
                if chosen_setup is setup_commands_musl:
                    # print("[WARN] Musl setup failed, falling back to glibc. Output:\n", output)
                    new_command = f'/bin/bash -c "{setup_commands_glibc}"'
                    try:
                        rc_out = self._retry_action(
                            lambda: docker_exec(container, new_command),
                            check_success=lambda r: r is not None and r[0] == 0,
                        )
                        return_code, output = rc_out
                    except Exception as e2:
                        return_code, output = None, str(e2)
                        print(f"[ERROR] Glibc setup also failed. Return: {return_code} Output:\n{output}")
                else:
                    print(f"[ERROR] Setup failed. Return: {return_code} Output:\n{output}")
        except Exception:
            print(f"Setup commands execution failed.")
            print(traceback.format_exc())

        # Build the inner run command (same for musl/glibc)
        inner_run = (
            f"trae-cli run --file {container_problem_statement_path} "
            f'--working-dir="{working_dir}" '
            f"--config-file trae_config.yaml "
            f"--patch-path {container_patch_file_path} --trajectory-file {container_traj_path}"
        )

        # Execute activation + run in the same shell so activation takes effect
        # Use the venv activation path (works for both musl-created venv and glibc venv)
        activate_cmd = "source /trae-workspace/trae-agent/.venv/bin/activate"
        full_cmd = f"{activate_cmd} && {inner_run}"

        new_command = f"/bin/bash -lc '{full_cmd}'"
        try:
            return_code, output = docker_exec(container, new_command)
            if return_code is None or return_code != 0:
                print(f"Docker exec error3. Error message: {output} Error instance:{instance_id}")
                container.stop()
                container.remove()
        except Exception as e:
            print("Docker exec exception when running trae-cli:", e)

        container.stop()
        container.remove()

    def run_all(self):
        """
        Run patch generation for all instances in the dataset, with parallelism controlled by max_workers.
        """
        instance_ids = [instance["instance_id"] for instance in self.dataset]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.run_one_instance, instance_id): instance_id
                for instance_id in instance_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running all instances"
            ):
                instance_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Instance {instance_id} failed: {e}")

    def run_eval(self):
        """
        Run evaluation using the benchmark harness.
        Evaluation results and predictions.json are stored in the task results directory.
        """
        self.config.evaluate_harness_before(
            self.task_results_dir, self.dataset_name, self.max_workers
        )

        benchmark_harness_path = Path(self.benchmark_harness_path)
        cmd = self.config.evaluate_harness(
            self.dataset_name, self.task_results_dir, self.task_id, self.max_workers
        )
        process = subprocess.run(cmd, capture_output=True, cwd=benchmark_harness_path.as_posix())
        print(process.stdout.decode())
        print(process.stderr.decode())

        result_filename = "results.json"
        result_path = self.task_results_dir / result_filename
        print(f"Evaluation completed and file saved to {result_path}")

        self.config.evaluate_harness_after(self.benchmark_harness_path, self.task_id)

    def get_all_preds(self, instance_ids: list[str] | None = None):
        """
        Collect all generated patches and write predictions.json to results directory.

        Args:
            instance_ids: List of instance IDs to collect (optional).
        """
        preds: list[dict[str, str]] = []
        if not instance_ids:
            instance_ids = [instance["instance_id"] for instance in self.dataset]
        for instance_id in instance_ids:
            patch_path = self.task_results_dir / instance_id / f"patch.patch"
            if not patch_path.exists():
                continue
            with open(patch_path, "r") as f:
                patch = f.read()
            preds.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": "trae-agent",
                    "model_patch": patch,
                }
            )
        with open(self.task_results_dir / "predictions.json", "w") as f:
            json.dump(preds, f)

def main():
    """
    Main entry point for benchmark evaluation script.
    Parses command-line arguments and runs patch generation and/or evaluation.
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--benchmark", type=str, default="SWE-bench", help="Benchmark name."
    )
    argument_parser.add_argument(
        "--dataset", type=str, default="SWE-bench", help="Dataset name."
    )
    argument_parser.add_argument(
        "--ite", type=int, default=0, help="Iteration index passed to utils problem statement builder."
    )
    argument_parser.add_argument(
        "--working-dir", type=str, default="./trae-workspace", help="Workspace directory."
    )
    argument_parser.add_argument(
        "--config-file", type=str, default="trae_config.yaml", help="Trae agent config file path."
    )
    argument_parser.add_argument(
        "--docker-env-config", type=str, default="", required=False, help="Docker env config file."
    )
    argument_parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated).",
    )
    argument_parser.add_argument(
        "--benchmark-harness-path",
        type=str,
        default="",
        required=False,
        help="Path to benchmark harness (for evaluation).",
    )
    argument_parser.add_argument(
        "--run-id",
        type=str,
        required=False,
        default="trae-agent",
        help="Run ID for benchmark evaluation.",
    )
    argument_parser.add_argument(
        "--max_workers", type=int, default=4, help="Maximum number of parallel workers."
    )

    args = argument_parser.parse_args()
    evaluation = BenchmarkEvaluation(
        args.benchmark,
        args.working_dir,
        args.config_file,
        args.ite,
        args.dataset,
        args.docker_env_config,
        args.benchmark_harness_path,
        args.run_id,
        args.max_workers,
        args.instance_ids,
    )

    evaluation.prepare_trae_agent()

    if args.instance_ids:
        print(f"Running specified instances: {args.instance_ids}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(evaluation.run_one_instance, instance_id): instance_id
                for instance_id in args.instance_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running instances"
            ):
                instance_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Instance {instance_id} failed: {e}")
    else:
        print("Running all instances in dataset.")
        evaluation.run_all()


if __name__ == "__main__":
    main()
