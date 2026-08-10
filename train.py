import math 
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.config import GPTConfig
from src.model import GPT
from src.dataset import prepare_datasets, MoveTokenizer

os.makedirs("staging", exist_ok=True)

# GPU check for cuda support.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
	device_name = torch.cuda.get_device_name(torch.cuda.current_device())
else:
	device_name = "System CPU"
print(f"GPU acceleration on : {device_name}")

# Loads config variables.
config = GPTConfig()
# Random seed for reproducibility.
torch.manual_seed(config.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(config.seed)


CHECKPOINT_PATH = "staging/training_checkpoint.pt"
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
    file_path="data/chessDataset_1.2m.txt",
    block_size=config.block_size,
    batch_size=config.batch_size,
    split_ratio=0.9,
    tokenizer=tokenizer,
)
config.vocab_size = tokenizer.vocab_size
tokenizer.save_vocab("staging/vocab.json")
config.save("staging/config.json")

# Setup model.
model = GPT(config).to(device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)

# Resume from checkpoint if one exists
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
		train_pbar = tqdm(
			train_loader, 
			desc=f"Epoch {epoch+1}/{config.max_epochs}", 
			leave=False,
			mininterval=0.5,
			bar_format="{l_bar}{bar}|[time={elapsed}{postfix}]"
		)
		
		for step, (X, Y) in enumerate(train_pbar):
			# --- skip already-completed steps when resuming mid-epoch ---
			if epoch == start_epoch and step < start_step_in_epoch:
				continue
			# --- end ---

			X, Y = X.to(device), Y.to(device)
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
				logits, loss = model(X, Y)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			total_train_loss += loss.item()

			# --- periodic mid-epoch checkpoint ---
			if step % CHECKPOINT_EVERY_N_STEPS == 0 and step > 0:
				save_checkpoint(model, optimizer, epoch, step, best_val_loss, history)
			# --- end ---
		  
			if step % 10 == 0:
				batches_per_sec = train_pbar.format_dict.get("rate")
				if batches_per_sec is not None and batches_per_sec > 0:
					tokens_per_batch = X.numel()
					tokens_per_sec = batches_per_sec * tokens_per_batch
					throughput_str = f"{tokens_per_sec:,.0f} tok/s"
				else:
					throughput_str = "estimating..."
					
				train_pbar.set_postfix(
					loss=f"{loss.item():.4f}", 
					throughput=throughput_str
				)

		# After finishing this epoch, next resume point is the START of the next one
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
		history.append({
			"epoch": epoch + 1,
			"train_loss": avg_train_loss,
			"train_ppl": train_ppl,
			"val_loss": avg_val_loss,
			"val_ppl": val_ppl
		})
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
			torch.save(raw_model.state_dict(), "staging/model_weights.pt")
			checkpoint_note = f"  ↳ New best val_loss ({avg_val_loss:.4f}), checkpoint saved."
		else:
			checkpoint_note = f"  ↳ val_loss did not improve (best: {best_val_loss:.4f})"

		# --- end-of-epoch checkpoint too, marking the next resume point ---
		save_checkpoint(model, optimizer, epoch + 1, 0, best_val_loss, history)
		# --- end ---

		os.system('cls' if os.name == 'nt' else 'clear')
		print("\n" + "="*63)
		print(f"| {'Epoch':^5} | {'Train Loss':^10} | {'Train PPL':^10} | {'Val Loss':^10} | {'Val PPL':^10} |")
		print("|" + "-"*61 + "|")
		for row in history:
			print(
				f"| {row['epoch']:^5} "
				f"| {row['train_loss']:^10.4f} "
				f"| {row['train_ppl']:^10.2f} "
				f"| {row['val_loss']:^10.4f} "
				f"| {row['val_ppl']:^10.2f} |"
			)
		print("="*63 + "\n")
		print(checkpoint_note + "\n")

	print(f"Training complete. Best checkpoint saved with val_loss={best_val_loss:.4f}")

except KeyboardInterrupt:
	# --- checkpoint on manual interrupt ---
	print("\nInterrupted -- saving checkpoint before exiting...")
	save_checkpoint(model, optimizer, epoch, step, best_val_loss, history)
	print("Checkpoint saved. Re-run this script to resume.")
