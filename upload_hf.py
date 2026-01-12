from huggingface_hub import create_repo, upload_folder

REPO_ID = "JohnnyZeppelin/zemantics"
FOLDER_PATH = "./runs"
HF_TOKEN = ""

# 1️⃣ 如果 repo 不存在就创建（存在不会报错）
create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    exist_ok=True
)

# 2️⃣ 上传整个 folder
upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
)

print("Repo created (if needed) and upload finished ✅")
