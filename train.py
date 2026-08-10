import math 
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.config import GPTConfig
from src.model import GPT
from src.dataset import prepare_datasets, MoveTokenizer

def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

if in_colab():
    print("Running in Google Colab environment.")
    data_root = "/content/drive/MyDrive/chess_project/"
    staging_root = "/content/drive/MyDrive/chess_project/staging/"
else:
    print("Running local.")
    data_root = "data/"
    staging_root = "staging/"

os.makedirs(staging_root, exist_ok=True)

# GPU check for cuda support.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
	device_name = torch.cuda.get_device_name(torch.cuda.current_device())
else:
	device_name = "System CPU"
print(f"GPU acceleration on : {device_name}")
config = GPTConfig()
torch.manual_seed(config.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(config.seed)
CHECKPOINT_PATH = staging_root + "training_checkpoint.pt"
CHECKPOINT_EVERY_N_STEPS = 5000  
def save_checkpoint(model, optimizer, epoch, step_in_epoch, best_val_loss, history):
	raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
	torch.save({
		'model_state_dict': raw_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
		'step_in_epoch': step_in_epoch,
		'best_val_loss': best_val_loss,
		'history': history,
	}, CHECKPOINT_PATH)
def load_checkpoint():
	if os.path.exists(CHECKPOINT_PATH):
		return torch.load(CHECKPOINT_PATH, map_location=device)
	return None
# Data Loader.
tokenizer = MoveTokenizer.from_vocab_file("vocab_fixed.json")
train_loader, val_loader, _ = prepare_datasets(
    file_path=data_root + "chessDataset_1.2m.txt",
    block_size=config.block_size,
    batch_size=config.batch_size,
    split_ratio=0.9,
    tokenizer=tokenizer,
)
config.vocab_size = tokenizer.vocab_size
tokenizer.save_vocab(staging_root + "vocab.json")
config.save(staging_root + "config.json")
model = GPT(config).to(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
start_epoch = 0
start_step_in_epoch = 0
best_val_loss = float('inf')
history = []
ckpt = load_checkpoint()
if ckpt is not None:
	model.load_state_dict(ckpt['model_state_dict'])
	optimizer.load_state_dict(ckpt['optimizer_state_dict'])
	start_epoch = ckpt['epoch']
	start_step_in_epoch = ckpt['step_in_epoch']
	best_val_loss = ckpt['best_val_loss']
	history = ckpt['history']
	print(f"Resumed from checkpoint: epoch {start_epoch+1}, step {start_step_in_epoch:,}")
else:
	print("No checkpoint found -- starting fresh.")
model = torch.compile(model)
try:
	for epoch in range(start_epoch, config.max_epochs):
		model.train() 
		total_train_loss = 0
		train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}", leave=False, mininterval=0.5)
		for step, (X, Y) in enumerate(train_pbar):
			if epoch == start_epoch and step < start_step_in_epoch:
				continue
			X, Y = X.to(device), Y.to(device)
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
				logits, loss = model(X, Y)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			total_train_loss += loss.item()
			if step % CHECKPOINT_EVERY_N_STEPS == 0 and step > 0:
				save_checkpoint(model, optimizer, epoch, step, best_val_loss, history)
		start_step_in_epoch = 0
		model.eval()
		total_val_loss = 0
		with torch.no_grad():
			for X_val, Y_val in val_loader:
				X_val, Y_val = X_val.to(device), Y_val.to(device)
				_, loss_val = model(X_val, Y_val)
				total_val_loss += loss_val.item()
		avg_train_loss = total_train_loss / len(train_loader)
		avg_val_loss = total_val_loss / len(val_loader)
		history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
			torch.save(raw_model.state_dict(), staging_root + "model_weights.pt")
		save_checkpoint(model, optimizer, epoch + 1, 0, best_val_loss, history)
	print(f"Training complete. Best val_loss={best_val_loss:.4f}")
except KeyboardInterrupt:
	print("\nInterrupted -- saving checkpoint before exiting...")
	save_checkpoint(model, optimizer, epoch, step, best_val_loss, history)
	print("Checkpoint saved. Re-run this script to resume.")
