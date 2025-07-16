'''

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/app/ai-toolkit/config/whatever_you_want.yml

'''

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import modal
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, "/app/ai-toolkit")
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/app/ai-toolkit/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement

# define modal app
image = modal.Image.from_dockerfile("docker/Dockerfile").add_local_dir("config", "/app/ai-toolkit/config").add_local_dir("input/images", '/app/ai-toolkit/input/images')

# mount for the entire ai-toolkit directory
# example: "/Users/username/ai-toolkit" is the local directory, "/root/ai-toolkit" is the remote directory
# code_mount = modal.Mount.from_local_dir("D:/ai-toolkit", remote_path="/root/ai-toolkit")

# create the Modal app with the necessary mounts and volumes
# app = modal.App(name="flux-lora-training", image=image, mounts=[code_mount], volumes={MOUNT_DIR: model_volume})
app = modal.App(name="flux-lora-training", image=image, volumes={MOUNT_DIR: model_volume})

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # Set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

import argparse
from toolkit.job import get_job

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="A100", # gpu="H100"
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200,  # 2 hours, increase or decrease if needed
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):    
    # convert the config file list from a string to a list
    config_file_list = str(config_file_list_str).split(",")

    jobs_completed = 0
    jobs_failed = 0

    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, name)
            
            job.config['process'][0]['training_folder'] = MOUNT_DIR
            os.makedirs(MOUNT_DIR, exist_ok=True)
            print(f"Training outputs will be saved to: {MOUNT_DIR}")
            
            # run the job
            job.run()
            
            # commit the volume after training
            model_volume.commit()
            
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)

# WebUIアクセス用の新しい関数を追加（GPU対応でトレーニング可能）
@app.function(
    image=image,
    gpu="A100",  # ← GPUを有効化！WebUIからトレーニングできるように
    cpu=4,
    memory=32768,  # 32GB
    timeout=7200,  # 2時間
    volumes={MOUNT_DIR: model_volume},  # ← ボリュームもマウント
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def webui():
    import subprocess
    import time
    import os
    
    # Hugging Faceにログイン
    try:
        from huggingface_hub import login
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Successfully logged in to Hugging Face")
        else:
            print("⚠️ No HF_TOKEN found")
    except Exception as e:
        print(f"⚠️ Failed to login to Hugging Face: {e}")
    
    # AI-toolkit UIディレクトリに移動
    os.chdir("/app/ai-toolkit/ui")
    
    # トレーニング結果保存先を作成
    os.makedirs(MOUNT_DIR, exist_ok=True)
    
    # modal.forwardでポート8675を公開
    with modal.forward(8675) as tunnel:
        print(f"🌐 AI Toolkit WebUI is accessible at: {tunnel.url}")
        print(f"🔧 Starting UI server...")
        print(f"💾 Training outputs will be saved to: {MOUNT_DIR}")
        
        # npm run startでWebUIを起動
        env_vars = os.environ.copy()
        if hf_token:
            env_vars.update({
                "HF_TOKEN": hf_token,
                "HUGGINGFACE_HUB_TOKEN": hf_token,
                "HF_API_TOKEN": hf_token
            })
        
        process = subprocess.Popen(
            ["npm", "run", "start"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd="/app/ai-toolkit/ui",
            env=env_vars  # 環境変数を明示的に渡す
        )
        
        # サーバーが起動するまで待機
        print("⏳ Waiting for server to start...")
        time.sleep(30)
        print(f"✅ WebUI should now be accessible at: {tunnel.url}")
        print(f"🚀 You can now start training jobs from the WebUI!")
        
        # 2時間維持（トレーニング時間を考慮）
        try:
            while process.poll() is None:
                # プロセスのログを定期的に出力
                try:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(f"[WebUI] {stdout_line.decode().strip()}")
                    
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        print(f"[WebUI ERROR] {stderr_line.decode().strip()}")
                except:
                    pass
                
                time.sleep(10)
                # 定期的にボリュームをコミット
                model_volume.commit()
        except KeyboardInterrupt:
            print("🛑 Shutting down WebUI...")
            process.terminate()
        finally:
            # プロセス終了時の最終ログ
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                if stdout:
                    print(f"[WebUI Final STDOUT] {stdout.decode()}")
                if stderr:
                    print(f"[WebUI Final STDERR] {stderr.decode()}")
            
            # 最終コミット
            model_volume.commit()
            print("💾 Final volume commit completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # require at least one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if a job fails
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # optional name replacement for config file
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    # WebUIオプションを追加
    parser.add_argument(
        '--webui',
        action='store_true',
        help='Launch WebUI instead of training'
    )
    
    args = parser.parse_args()

    if args.webui:
        # WebUI起動
        webui.remote()
    else:
        # 従来通りのトレーニング実行
        config_file_list_str = ",".join(args.config_file_list)
        main.remote(config_file_list_str=config_file_list_str, recover=args.recover, name=args.name)