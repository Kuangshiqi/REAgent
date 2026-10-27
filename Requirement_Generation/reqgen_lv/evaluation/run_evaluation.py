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
        dataset: str = "SWE-bench_Verified",
        docker_env_config: str = "",
        benchmark_harness_path: str = "",
        run_id: str = "trae-agent",
        max_workers: int = 4,
        instance_ids: list[str] | None = None,
        ite: int = 0,
    ):
        assert benchmark in BENCHMARK_CONFIG, f"Invalid benchmark name: {benchmark}"
        self.config = BENCHMARK_CONFIG[benchmark]
        self.dataset_name = dataset
        assert self.dataset_name in self.config.valid_datasets, (
            f"Invalid dataset name: {self.dataset_name}"
        )

        self.benchmark = benchmark
        self.dataset = self.config.load_dataset(self.dataset_name)
        self.docker_client: DockerClient = from_env()
        self.image_status: dict[Any, Any] = {}

        self.working_dir = Path(working_dir)
        self.benchmark_harness_path = benchmark_harness_path
        self.run_id = run_id
        self.max_workers = max_workers
        self.ite = ite
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
        Get the Docker image name for a given instance ID.

        Args:
            instance_id: Instance identifier.

        Returns:
            Docker image name string.
        """
        return self.config.image_name(instance_id)


    def prepare_trae_agent(self):
        """
        Build Trae Agent and UV inside a base Ubuntu container.
        Save built artifacts to workspace for later use in experiment containers.
        """
        tars = ["trae-agent.tar", "uv.tar", "uv_shared.tar"]
        all_exist = all((self.working_dir / tar).exists() for tar in tars)

        try:
            image = self.docker_client.images.get("ubuntu:22.04")
        except Exception:
            image = self.docker_client.images.pull("ubuntu:22.04")

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

        # ✅ If artifacts exist: only install pyerfa then exit
        if all_exist:
            container.stop()
            container.remove()
            return

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
            return_code, output = docker_exec(container, new_command)
            if return_code and return_code != 0:
                print("Docker exec error. Error message: {}".format(output))
                container.stop()
                container.remove()
                exit(-1)

        for tar_name, src_path in [
            ("trae-agent.tar", "/trae-workspace/trae-agent"),
            ("uv.tar", "/root/.local/bin/uv"),
            ("uv_shared.tar", "/root/.local/share/uv"),
        ]:
            with open(self.working_dir / tar_name, "wb") as f:
                bits, _ = container.get_archive(src_path)
                for chunk in bits:
                    f.write(chunk)

        container.stop()
        container.remove()

    def prepare_experiment_container(self, instance: dict[str, str]) -> Container:
        """
        Prepare experiment Docker container for a given instance.
        The container mounts the results directory for this instance,
        so all outputs are directly accessible on the host.
        Args:
            instance: Instance dictionary.
        Returns:
            Docker container object.
        """

        image_name = self._image_name(instance["instance_id"])
        instance_result_dir = self.task_results_dir / instance["instance_id"]
        instance_result_dir.mkdir(parents=True, exist_ok=True)

        self.config.problem_statement(instance, instance_result_dir, self.ite)

        container: Container = self.docker_client.containers.run(
            image_name,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                instance_result_dir.absolute().as_posix(): {"bind": "/instance-data", "mode": "rw"},
            },
            working_dir="/trae-workspace",
            environment=self.docker_env_config.get("experiment_env", None),
            stream=True,
        )

        for fname in ["trae-agent.tar", "uv.tar", "uv_shared.tar", "trae_config.yaml"]:
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(self.working_dir / fname, arcname=fname)
            tar_stream.seek(0)
            container.put_archive("/trae-workspace", tar_stream.getvalue())

        setup_commands = [
            "tar xf trae-agent.tar",
            "tar xf uv.tar",
            "mkdir -p /root/.local/bin",
            "mv uv /root/.local/bin/",
            "tar xf uv_shared.tar",
            "mkdir -p /root/.local/share",
            "mv uv /root/.local/share/",
        ]
        for command in setup_commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
                if return_code is not None and return_code != 0:
                    print("Docker exec error. Error message: {}".format(output))
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break
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
        container_traj_path = f"/instance-data/{instance_id}.json"

        container = self.prepare_experiment_container(instance)
        
        command = (
            f"source trae-agent/.venv/bin/activate && "
            f"trae-cli run --file {container_problem_statement_path} "
            f'--working-dir="{working_dir}" '
            f"--config-file trae_config.yaml "
            f"--trajectory-file {container_traj_path}"
        )

        new_command = f"/bin/bash -c '{command}'"
        try:
            return_code, output = docker_exec(container, new_command)
            if return_code is not None and return_code != 0:
                print("Docker exec error3. Error message: {}".format(output))
        except Exception:
            print(f"{command} failed.")
            print(traceback.format_exc())

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
            patch_path = self.task_results_dir / instance_id / f"{instance_id}.patch"
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
        "--dataset", type=str, default="SWE-bench_Verified", help="Dataset name."
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
        "--mode",
        type=str,
        choices=["e2e", "expr", "eval"],
        default="expr",
        help="e2e: both patch generation and evaluation; expr: only patch generation; eval: only evaluation.",
    )
    argument_parser.add_argument(
        "--max_workers", type=int, default=4, help="Maximum number of parallel workers."
    )
    argument_parser.add_argument(
        "--ite", type=int, default=0
    )

    args = argument_parser.parse_args()
    evaluation = BenchmarkEvaluation(
        args.benchmark,
        args.working_dir,
        args.config_file,
        args.dataset,
        args.docker_env_config,
        args.benchmark_harness_path,
        args.run_id,
        args.max_workers,
        args.instance_ids,
        args.ite,
    )

    evaluation.prepare_trae_agent()

    # Patch generation (expr/e2e mode)
    if args.mode in ("e2e", "expr"):
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

    # Evaluation (eval/e2e mode)
    if args.mode in ("e2e", "eval"):
        evaluation.get_all_preds(args.instance_ids)
        evaluation.run_eval()


if __name__ == "__main__":
    main()
 