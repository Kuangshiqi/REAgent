import argparse
import concurrent.futures
import io
import json
import os
import platform as py_platform
import tarfile

try:
    import modal  # Lazy/optional: only required when not using --use_local_docker
except Exception:
    modal = None
try:
    import docker  # Optional: used when --use_local_docker is set
except Exception:
    docker = None
import pandas as pd
from tqdm import tqdm

from helper_code.image_uri import get_dockerhub_image_uri

# Credit: prabhuteja12
def load_base_docker(iid):
    with open(f"dockerfiles/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()

def instance_docker(iid):
    with open(f"dockerfiles/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()

def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    with open(script_path, 'r') as f:
        return f.read()


def create_entryscript(sample):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(sample["instance_id"])
    instance_dockerfile = instance_docker(sample["instance_id"])
    
    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)
    
    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
echo "=== APPLY PATCH ==="
git apply -v /workspace/patch.diff > /workspace/apply.log 2>&1

if [ $? -eq 0 ]; then
  echo "SUCCESS" > /workspace/apply_status.txt
else
  echo "FAIL" > /workspace/apply_status.txt
  exit 1
fi
{before_repo_set_cmd}

bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log

python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def create_dockerhub_tag(uid, repo_name=""):
    if repo_name:
        repo_base, repo_name_only = repo_name.lower().split("/")
        hsh = uid.replace("instance_", "")
        return f"{repo_base}.{repo_name_only}-{hsh}"
    else:
        image_name = "default"

    if "__" in uid and len(uid) > 9:
        tag_part = uid[9:]  # Skip the first 9 characters (e.g., "django__")
    else:
        tag_part = uid

    return f"{image_name}-{tag_part}"

def prepare_run(uid, output_dir, prefix, redo):
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)
    output_path = os.path.join(uid_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        print(f"Skipping {uid} - output already exists")
        with open(output_path, "r") as f:
            return json.load(f), output_path, os.path.join(uid_dir, "workspace")
    workspace_dir = os.path.join(uid_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    return None, output_path, workspace_dir


def write_patch_snapshot(output_dir, uid, prefix, patch):
    with open(os.path.join(output_dir, uid, f"{prefix}_patch.diff"), "w") as f:
        f.write(patch)


def assemble_workspace_files(uid, scripts_dir, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(sample)

    files = {
        "patch.diff": patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def write_files_modal(sandbox, files):
    for rel_path, content in files.items():
        with sandbox.open(f"/workspace/{rel_path}", "w") as f:
            f.write(content)


def write_files_local(workspace_dir, files):
    for rel_path, content in files.items():
        dst = os.path.join(workspace_dir, rel_path)
        with open(dst, "w") as f:
            f.write(content)


def save_entryscript_copy(output_dir, uid, prefix, entryscript_content):
    with open(os.path.join(output_dir, uid, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def collect_outputs_modal(sandbox, output_dir, uid, prefix):
    # Save logs first (best-effort)
    try:
        with sandbox.open("/workspace/stdout.log", "r") as f_in:
            with open(os.path.join(output_dir, uid, f"{prefix}_stdout.log"), "w") as f:
                stdout_content = f_in.read()
                f.write(stdout_content if stdout_content is not None else "")
    except FileNotFoundError:
        pass
    try:
        with sandbox.open("/workspace/stderr.log", "r") as f_in:
            with open(os.path.join(output_dir, uid, f"{prefix}_stderr.log"), "w") as f:
                stderr_content = f_in.read()
                f.write(stderr_content if stderr_content is not None else "")
    except FileNotFoundError:
        pass

    # Then try to read output.json
    try:
        with sandbox.open("/workspace/output.json", "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        # print(
        #     f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        # )
        return None


def collect_outputs_local(workspace_dir, output_dir, uid, prefix):
    def _copy_safe(src_name, dest_name):
        src_path = os.path.join(workspace_dir, src_name)
        dest_path = os.path.join(output_dir, uid, dest_name)
        try:
            with open(src_path, "r") as f_in:
                content = f_in.read()
        except FileNotFoundError:
            content = ""
        with open(dest_path, "w") as f_out:
            f_out.write(content if content is not None else "")

    _copy_safe("stdout.log", f"{prefix}_stdout.log")
    _copy_safe("stderr.log", f"{prefix}_stderr.log")
    _copy_safe("apply.log", f"{prefix}_apply.log")
    _copy_safe("apply_status.txt", f"{prefix}_apply_status.txt")

    # Then try to read output.json
    try:
        with open(os.path.join(workspace_dir, "output.json"), "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        # print(
        #     f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        # )
        return None


def _make_tar_archive(files_dict):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        dir_info = tarfile.TarInfo(name="workspace")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)
        for name, content in files_dict.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"workspace/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _extract_file_from_container(container, container_path):
    """Read a single file from a (stopped) container. Returns str or None."""
    try:
        stream, _ = container.get_archive(container_path)
        raw = b"".join(stream)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            member = tar.getmembers()[0]
            f = tar.extractfile(member)
            if f is None:
                return None
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def eval_with_docker(patch, sample, output_dir, dockerhub_username, scripts_dir, prefix="", redo=False, docker_platform=None):
    if docker is None:
        raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    container = None
    try:
        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))

        client = docker.from_env(timeout=3600)
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            try:
                client.images.get(dockerhub_image_uri)
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
                with open("missing_images.txt", "a") as f:
                    f.write(dockerhub_image_uri + "\n")
                return None

        # --- docker cp approach (works inside Docker-in-Docker) ---
        create_kwargs = {
            "entrypoint": "/bin/bash",
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }
        if docker_platform:
            create_kwargs["platform"] = docker_platform

        container = client.containers.create(dockerhub_image_uri, **create_kwargs)

        tar_buf = _make_tar_archive(files)
        container.put_archive("/", tar_buf)

        container.start()
        result = container.wait()
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1

        output_files = ["stdout.log", "stderr.log", "apply.log", "apply_status.txt", "output.json"]
        for fname in output_files:
            content = _extract_file_from_container(container, f"/workspace/{fname}")
            local_path = os.path.join(workspace_dir, fname)
            # 当 output.json 提取不到内容时，不写入，避免产生空文件导致 json.load 报错
            if content is not None or fname != "output.json":
                with open(local_path, "w") as fp:
                    fp.write(content if content else "")

        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix)

        apply_status_file = os.path.join(output_dir, uid, f"{prefix}_apply_status.txt")
        apply_success = False
        if os.path.exists(apply_status_file):
            with open(apply_status_file) as f:
                apply_success = f.read().strip() == "SUCCESS"

        return {
            "test_output": output,
            "apply_success": apply_success,
        }
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWEAP Pro evaluations using Modal or local Docker with Docker Hub images and local scripts")
    parser.add_argument("--raw_sample_path", required=True, help="Path to the raw sample CSV file")
    parser.add_argument(
        "--patch_path", required=True, help="Path to the JSON file containing patches"
    )
    parser.add_argument("--output_dir", required=True, help="Directory to store evaluation outputs")
    parser.add_argument(
        "--dockerhub_username", required=True, help="Docker Hub username where sweap-images repository is located"
    )
    parser.add_argument(
        "--scripts_dir", required=True, help="Directory containing local run scripts (e.g., scripts/run_scripts)"
    )
    parser.add_argument(
        "--redo", action="store_true", help="Redo evaluations even if output exists"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of workers to run evaluations in parallel",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        default=None,
        help="Optional list of instance_ids to run. If omitted, all valid patches are run. Example: --instance_ids django__django-12345 project__repo-67890",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Support both JSONL and CSV input files
    if args.raw_sample_path.endswith(".jsonl"):
        raw_sample_df = pd.read_json(args.raw_sample_path, lines=True)
    else:
        raw_sample_df = pd.read_csv(args.raw_sample_path)
    
    # Replace nulls with empty strings
    raw_sample_df = raw_sample_df.fillna("")
    
    # use instance_id as index
    raw_sample_df = raw_sample_df.set_index("instance_id", drop=False)

    # each patch sample is a dict with keys: instance_id, patch, prefix
    with open(args.patch_path, "r") as f:
        patches_to_run = json.load(f)
    eval_results = {}
    apply_success_count = 0
    end_to_end_success_count = 0
    total_cases = 0


    # Optional filtering by --instance_ids. If provided, only patches whose
    # instance_id is in that list will be considered. Otherwise all patches
    # are considered (subject to presence in the raw sample data).
    requested_ids = set(args.instance_ids) if args.instance_ids else None
    valid_patches = []
    missing_instances = []
    skipped_by_request = []
    for patch_sample in patches_to_run:
        instance_id = patch_sample["instance_id"]
        # If user provided a list of instance_ids, skip patches not in that list
        if requested_ids is not None and instance_id not in requested_ids:
            skipped_by_request.append(instance_id)
            continue
        # Only keep patches that have a matching row in the raw sample data
        if instance_id in raw_sample_df.index:
            valid_patches.append(patch_sample)
        else:
            missing_instances.append(instance_id)

    if skipped_by_request:
        print(f"Note: {len(skipped_by_request)} patches were skipped because they were not listed in --instance_ids")
        for sid in skipped_by_request[:5]:
            print(f"  - skipped: {sid}")
        if len(skipped_by_request) > 5:
            print(f"  ... and {len(skipped_by_request) - 5} more")

    if missing_instances:
        print(f"Warning: Found {len(missing_instances)} patch instances not in raw sample data:")
        for missing_id in missing_instances[:5]:  # Show first 5
            print(f"  - {missing_id}")
        if len(missing_instances) > 5:
            print(f"  ... and {len(missing_instances) - 5} more")

    print(f"Proceeding with {len(valid_patches)} valid patches out of {len(patches_to_run)} total patches (after --instance_ids filtering)")

    detected_platform = "linux/amd64"

    eval_fn = eval_with_docker

    # Use ThreadPoolExecutor to run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Create a dictionary mapping futures to their patch samples for progress tracking
        future_to_patch = {
            executor.submit(
                eval_fn,
                patch_sample.get("model_patch", patch_sample.get("patch", "")),
                raw_sample_df.loc[patch_sample["instance_id"]],
                args.output_dir,
                args.dockerhub_username,
                args.scripts_dir,
                prefix=patch_sample.get("prefix", ""),
                redo=args.redo,
                docker_platform=detected_platform,
            ): patch_sample
            for patch_sample in valid_patches
        }

        # Track progress with tqdm and show running accuracy
        pbar = tqdm(concurrent.futures.as_completed(future_to_patch), total=len(valid_patches))
        for future in pbar:
            patch_sample = future_to_patch[future]
            try:
                output_bundle = future.result()
                instance_id = patch_sample["instance_id"]
                total_cases += 1  # ⭐ 总 case 计数

                if output_bundle is None:
                    print(f'Evaluation for {instance_id} returned None')
                    eval_results[instance_id] = False

                else:
                    apply_success = output_bundle["apply_success"]
                    test_output = output_bundle["test_output"]

                    # 统计 apply 成功
                    if apply_success:
                        apply_success_count += 1

                        # 只有 apply 成功才检查测试
                        if test_output is not None:
                            raw_sample = raw_sample_df.loc[instance_id]
                            passed_tests = {x["name"] for x in test_output["tests"] if x["status"] == "PASSED"}
                            f2p = set(eval(raw_sample["fail_to_pass"]))
                            p2p = set(eval(raw_sample["pass_to_pass"]))
                            end_to_end_success = (f2p | p2p) <= passed_tests
                        else:
                            end_to_end_success = False
                    else:
                        end_to_end_success = False

                    # ⭐ 统计端到端成功
                    if end_to_end_success:
                        end_to_end_success_count += 1

                    eval_results[instance_id] = end_to_end_success

                current_accuracy = sum(eval_results.values()) / len(eval_results)
                apply_rate_live = apply_success_count / total_cases if total_cases else 0

                pbar.set_description(
                    f"Apply: {apply_rate_live:.3%} | E2E: {current_accuracy:.3%}"
                )

            except Exception as exc:
                print(f'Evaluation for {instance_id} generated an exception: {exc}')
                eval_results[instance_id] = False
                total_cases += 1
                current_accuracy = sum(eval_results.values()) / len(eval_results)
                apply_rate_live = apply_success_count / total_cases if total_cases else 0

                pbar.set_description(
                    f"Apply: {apply_rate_live:.3%} | E2E: {current_accuracy:.3%}"
                )


                current_accuracy = sum(eval_results.values()) / len(eval_results)
                apply_rate_live = apply_success_count / total_cases if total_cases else 0

                pbar.set_description(
                    f"Apply: {apply_rate_live:.3%} | E2E: {current_accuracy:.3%}"
                )
            except Exception as exc:
                print(f'Evaluation for {patch_sample["instance_id"]} generated an exception: {exc}')
                eval_results[patch_sample["instance_id"]] = False
                # Update progress bar description with current accuracy
                current_accuracy = sum(eval_results.values()) / len(eval_results)
                apply_rate_live = apply_success_count / total_cases if total_cases else 0

                pbar.set_description(
                    f"Apply: {apply_rate_live:.3%} | E2E: {current_accuracy:.3%}"
                )
    # 之前写文件位置被调整，先计算指标，再写入包含 summary 的 JSON
    apply_rate = apply_success_count / total_cases if total_cases else 0
    end_to_end_rate = end_to_end_success_count / total_cases if total_cases else 0

    output_data = {
        "summary": {
            "total_cases": total_cases,
            "apply_success_count": apply_success_count,
            "end_to_end_success_count": end_to_end_success_count,
            "apply_rate": apply_rate,
            "end_to_end_rate": end_to_end_rate,
        },
        "results": eval_results,
    }

    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n================ FINAL METRICS ================")
    print(f"Patch Apply Success Rate: {apply_rate:.3%}")
    print(f"End-to-End Success Rate: {end_to_end_rate:.3%}")
    print("================================================")



if __name__ == "__main__":
    main()