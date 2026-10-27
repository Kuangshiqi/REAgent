import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path
import platform as py_platform
import tarfile
import time

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

def load_test_patches(jsonl_path):
    data = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            data[obj["instance_id"]] = obj["test_patch"]  # 长度为10的list
    return data

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
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(sample["instance_id"])
    instance_dockerfile = instance_docker(sample["instance_id"])

    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            if line.strip().startswith("ENV"):
                env_cmds.append(line.strip().replace("ENV", "export", 1))

    env_cmds = "\n".join(env_cmds)

    return f"""
set -e
{env_cmds}

cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/model_patch.diff
git apply -v /workspace/test_patch.diff

python reproduce_bug.py > /workspace/test_stdout.log 2> /workspace/test_stderr.log
echo $? > /workspace/exit_code.txt
"""


def create_dockerhub_tag(uid, repo_name=""):
    if repo_name:
        repo_base, repo_name_only = repo_name.lower().split("/")
        hsh = uid.replace("instance_", "")
        return f"{repo_base}.{repo_name_only}-{hsh}"
    else:
        image_name = "default"

    if "__" in uid and len(uid) > 9:
        tag_part = uid[9:]
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


def assemble_workspace_files(uid, scripts_dir, model_patch, test_patch, sample):
    entryscript_content = create_entryscript(sample)

    files = {
        "model_patch.diff": model_patch,
        "test_patch.diff": test_patch,
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
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None

def collect_outputs_local(workspace_dir, output_dir, uid, prefix):
    """
    output_dir 现在已经是 batch 目录，例如：
    output_dir/test_0/
    所以这里只需要建 instance 子目录
    """
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)

    def copy(src_name, dst_name=None):
        dst_name = dst_name or src_name
        try:
            with open(os.path.join(workspace_dir, src_name)) as f_in:
                with open(os.path.join(uid_dir, dst_name), "w") as f_out:
                    f_out.write(f_in.read())
        except FileNotFoundError:
            pass

    # ⭐ 这是你真正要的 print 输出
    copy("test_stdout.log", "stdout.txt")

    # 错误输出
    copy("test_stderr.log", "stderr.txt")

    # exit code 现在只是参考
    copy("exit_code.txt")

    return {"log_saved": True}


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


def eval_with_docker(model_patch, test_patch, sample, output_dir, dockerhub_username, scripts_dir, prefix="", redo=False, docker_platform=None):
    if docker is None:
        raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    # print(f"Running local-docker evaluation for {uid}")

    try:
        try:
            files, entryscript_content = assemble_workspace_files(
                uid, scripts_dir, model_patch, test_patch, sample
            )
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, "test", test_patch)
        write_patch_snapshot(output_dir, uid, "model", model_patch)

        # Run container via Docker SDK
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        # print(f"Using Docker Hub image: {dockerhub_image_uri}")

        client = docker.from_env(timeout=3600)
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            # If pull fails, fall back to a local image if present; otherwise, fail this run
            try:
                client.images.get(dockerhub_image_uri)
                # print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
                return None

        # DinD: use put_archive/get_archive instead of bind mount
        create_kwargs = {
            "detach": True,
            "entrypoint": "/bin/bash",
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }
        if docker_platform:
            create_kwargs["platform"] = docker_platform

        container = client.containers.create(dockerhub_image_uri, **create_kwargs)
        tar_buf = _make_tar_archive(files)
        container.put_archive("/", tar_buf)
        container.start()

        # 等待容器完成，但如果超过 timeout_seconds 则视为失败（patch 错误）
        timeout_seconds = 15 * 60
        start_ts = time.time()
        timed_out = False
        try:
            while True:
                container.reload()
                if getattr(container, "status", "") == "exited":
                    break
                if time.time() - start_ts > timeout_seconds:
                    timed_out = True
                    print(f"Timeout: instance {uid} exceeded {timeout_seconds} seconds, marking as failed.")
                    try:
                        container.kill()
                    except Exception:
                        pass
                    break
                time.sleep(1)

        # 容器 wait/reload 过程中出现异常也视为失败
        except Exception as wait_err:
            print(f"Error while waiting for container for {uid}: {wait_err}")
            try:
                container.kill()
            except Exception:
                pass
            timed_out = True

        # 如果正常退出，尝试获取状态码；超时则使用非零退出码标记失败
        status_code = 1
        if not timed_out:
            try:
                res = container.wait()
                status_code = res.get("StatusCode", 1) if isinstance(res, dict) else 1
            except Exception:
                status_code = 1
        else:
            # 超时：清空工作区的 stdout/stderr（保证后续 collect 输出为空）
            try:
                open(os.path.join(workspace_dir, "test_stdout.log"), "w").close()
                open(os.path.join(workspace_dir, "test_stderr.log"), "w").close()
                with open(os.path.join(workspace_dir, "exit_code.txt"), "w") as f:
                    f.write("124")
            except Exception:
                pass
            print(f"Entryscript timed out for {uid}, treating as failure.")

        # DinD: extract output files from container when not timed out
        if not timed_out:
            for fname in ["test_stdout.log", "test_stderr.log", "exit_code.txt"]:
                content = _extract_file_from_container(container, f"/workspace/{fname}")
                local_path = os.path.join(workspace_dir, fname)
                if content is not None:
                    with open(local_path, "w") as fp:
                        fp.write(content)
                # content 为 None 时不创建文件，便于 logread 将 apply 失败等识别为 Error_Patch

        # Collect outputs and logs, and save entryscript for reference
        collect_outputs_local(workspace_dir, output_dir, uid, prefix)
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)
        try:
            container.remove(force=True)
        except Exception:
            pass
        return {"status": "done"}
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SWEAP Pro evaluations with model patch + multi test patches"
    )
    parser.add_argument("--raw_sample_path", required=True)
    parser.add_argument("--model_patch_path", required=True)
    parser.add_argument("--test_patch_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dockerhub_username", default="jefzda")
    parser.add_argument("--scripts_dir", required=True)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--num_workers", type=int, default=50)
    parser.add_argument("--instance_ids", nargs="+", default=None)
    return parser.parse_args()


def load_test_patches(path):
    """jsonl: {instance_id, test_patch: [patch1,...,patch10]}"""
    test_patches = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            test_patches[obj["instance_id"]] = obj["test_patch"]
    return test_patches


def main():
    args = parse_args()

    # ========= 读取 raw sample =========
    if args.raw_sample_path.endswith(".jsonl"):
        raw_sample_df = pd.read_json(args.raw_sample_path, lines=True)
    else:
        raw_sample_df = pd.read_csv(args.raw_sample_path)

    raw_sample_df = raw_sample_df.fillna("")
    raw_sample_df = raw_sample_df.set_index("instance_id", drop=False)

    # ========= 读取模型补丁 =========
    with open(args.model_patch_path) as f:
        model_patches = {x["instance_id"]: x["patch"] for x in json.load(f)}

    # ========= 读取测试补丁 =========
    test_patches = load_test_patches(args.test_patch_jsonl)

    # ========= 过滤 instance =========
    instance_ids = [
        iid for iid in test_patches
        if iid in model_patches and iid in raw_sample_df.index
    ]

    if args.instance_ids:
        instance_ids = [iid for iid in instance_ids if iid in args.instance_ids]

    print(f"Total instances to evaluate: {len(instance_ids)}")

    detected_platform = "linux/amd64"
    eval_fn = eval_with_docker   # ⚠️ 你的原函数，不能删

    # ========= 10轮测试 =========
    for test_idx in range(10):
        print(f"\n================ TEST BATCH {test_idx} ================\n")

        batch_output_dir = os.path.join(args.output_dir, f"test_{test_idx}")
        os.makedirs(batch_output_dir, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}

            for iid in instance_ids:
                future = executor.submit(
                    eval_fn,
                    model_patches[iid],             # 第1步：模型补丁
                    test_patches[iid][test_idx],    # 第2步：测试补丁
                    raw_sample_df.loc[iid],
                    batch_output_dir,
                    args.dockerhub_username,
                    args.scripts_dir,
                    prefix=f"test{test_idx}",
                    redo=args.redo,
                    docker_platform=detected_platform,
                )
                futures[future] = iid

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                iid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] test {test_idx} instance {iid}: {e}")

    print("\nAll 10 test batches completed.")

    _pro_dir = Path(__file__).resolve().parent
    run_name = Path(args.output_dir).resolve().name
    summary_path = _pro_dir / "llm_result" / f"{run_name}_result.jsonl"
    from logread import summarize_pro_output_dir

    summarize_pro_output_dir(
        args.output_dir,
        instance_ids,
        args.model_patch_path,
        str(summary_path),
    )
    print(f"✅ Summary written to {summary_path}")


if __name__ == "__main__":
    main()