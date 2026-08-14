import math 
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.config import GPTConfig
from src.model import GPT
from src.dataset import prepare_datasets, MoveTokenizer

def in_colab():
    try:
        import google.colab
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

def format_duration(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"

def save_checkpoint(model, optimizer, epoch, step_in_epoch, best_val_loss, history, elapsed_seconds):
	raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
	tmp_path = CHECKPOINT_PATH + ".tmp"
	torch.save({
		'model_state_dict': raw_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
		'step_in_epoch': step_in_epoch,
		'best_val_loss': best_val_loss,
		'history': history,
		'elapsed_seconds': elapsed_seconds,
	}, tmp_path)
	os.replace(tmp_path, CHECKPOINT_PATH)  # atomic on the same filesystem -- no partial-write window at the real path

def load_checkpoint():
	if os.path.exists(CHECKPOINT_PATH):
		return torch.load(CHECKPOINT_PATH, map_location=device)
	return None

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
elapsed_seconds = 0.0  # TRUE cumulative training wall-clock time, survives resumes
ckpt = load_checkpoint()
if ckpt is not None:
	model.load_state_dict(ckpt['model_state_dict'])
	optimizer.load_state_dict(ckpt['optimizer_state_dict'])
	start_epoch = ckpt['epoch']
	start_step_in_epoch = ckpt['step_in_epoch']
	best_val_loss = ckpt['best_val_loss']
	history = ckpt['history']
	elapsed_seconds = ckpt.get('elapsed_seconds', 0.0)  # .get() for backward compat with older checkpoints
	print(f"Resumed from checkpoint: epoch {start_epoch+1}, step {start_step_in_epoch:,}")
	print(f"Elapsed training time so far: {format_duration(elapsed_seconds)}")
else:
	print("No checkpoint found -- starting fresh.")
model = torch.compile(model)

session_start_time = time.time()  # this session's own start, NOT the run's true start

def current_total_elapsed():
    return elapsed_seconds + (time.time() - session_start_time)

try:
	for epoch in range(start_epoch, config.max_epochs):
		model.train() 
		total_train_loss = 0
		train_pbar = tqdm(
			train_loader, 
			desc=f"Epoch {epoch+1}/{config.max_epochs}", 
			leave=False,
			mininterval=0.5,
			bar_format="{l_bar}{bar}|[time={elapsed}{postfix}]"
		)
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
				save_checkpoint(model, optimizer, epoch, step, best_val_loss, history, current_total_elapsed())
			if step % 10 == 0:
				batches_per_sec = train_pbar.format_dict.get("rate")
				if batches_per_sec is not None and batches_per_sec > 0:
					tokens_per_batch = X.numel()
					tokens_per_sec = batches_per_sec * tokens_per_batch
					throughput_str = f"{tokens_per_sec:,.0f} tok/s"
				else:
					throughput_str = "estimating..."
				train_pbar.set_postfix(loss=f"{loss.item():.4f}", throughput=throughput_str)
		start_step_in_epoch = 0
		model.eval()
		total_val_loss = 0
		val_pbar = tqdm(val_loader, desc="Validating", leave=False, mininterval=0.5)
		with torch.no_grad():  
			for X_val, Y_val in val_pbar:
				X_val, Y_val = X_val.to(device), Y_val.to(device)
				_, loss_val = model(X_val, Y_val)
				total_val_loss += loss_val.item()
				val_pbar.set_postfix(val_loss=f"{loss_val.item():.4f}")
		avg_train_loss = total_train_loss / len(train_loader)
		avg_val_loss = total_val_loss / len(val_loader)
		try:
			train_ppl = math.exp(avg_train_loss)
			val_ppl = math.exp(avg_val_loss)
		except OverflowError:
			train_ppl = float('inf')
			val_ppl = float('inf')
		elapsed_seconds = current_total_elapsed()
		history.append({
			"epoch": epoch + 1,
			"train_loss": avg_train_loss,
			"train_ppl": train_ppl,
			"val_loss": avg_val_loss,
			"val_ppl": val_ppl,
			"elapsed_seconds": elapsed_seconds,
		})
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
			weights_tmp = staging_root + "model_weights.pt.tmp"
			torch.save(raw_model.state_dict(), weights_tmp)
			os.replace(weights_tmp, staging_root + "model_weights.pt")
			checkpoint_note = f"  ↳ New best val_loss ({avg_val_loss:.4f}), checkpoint saved."
		else:
			checkpoint_note = f"  ↳ val_loss did not improve (best: {best_val_loss:.4f})"
		save_checkpoint(model, optimizer, epoch + 1, 0, best_val_loss, history, elapsed_seconds)
		os.system('cls' if os.name == 'nt' else 'clear')
		print("\n" + "="*97)
		print(f"| {'Epoch':^5} | {'Train Loss':^10} | {'Train PPL':^10} | {'Val Loss':^10} | {'Val PPL':^10} | {'Duration':^13} | {'Total':^13} |")
		print("|" + "-"*95 + "|")
		prev_elapsed = 0.0
		for row in history:
			duration = row['elapsed_seconds'] - prev_elapsed
			prev_elapsed = row['elapsed_seconds']
			print(
				f"| {row['epoch']:^5} "
				f"| {row['train_loss']:^10.4f} "
				f"| {row['train_ppl']:^10.2f} "
				f"| {row['val_loss']:^10.4f} "
				f"| {row['val_ppl']:^10.2f} "
				f"| {format_duration(duration):^13} "
				f"| {format_duration(row['elapsed_seconds']):^13} |"
			)
		print("="*97 + "\n")
		print(checkpoint_note + "\n")
	print(f"Training complete. Best val_loss={best_val_loss:.4f}. Total elapsed: {format_duration(elapsed_seconds)}")
except KeyboardInterrupt:
	final_elapsed = current_total_elapsed()
	print(f"\nInterrupted -- saving checkpoint before exiting...")
	print(f"Actual elapsed training time before stopping: {format_duration(final_elapsed)}")
	save_checkpoint(model, optimizer, epoch, step, best_val_loss, history, final_elapsed)
	print("Checkpoint saved. Re-run this script to resume.")
