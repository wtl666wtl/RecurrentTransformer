import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recqwen import RecurrentQwenForCausalLM

src_path = "Qwen/Qwen2.5-0.5B-Instruct"
dst_path = "./model/RecQwen-0.5B-8888"

base = AutoModelForCausalLM.from_pretrained(
    src_path, torch_dtype=torch.float32, attn_implementation="flash_attention_2", trust_remote_code=True
)

cfg = base.config
cfg.architectures = ["RecurrentQwen.RecurrentQwenForCausalLM"]
cfg.auto_map = {
  "AutoConfig": "RecurrentQwen.RecurrentQwenConfig",
  "AutoModelForCausalLM": "RecurrentQwen.RecurrentQwenForCausalLM"
}
cfg.prelude_layer        = 8
cfg.coda_layer           = 8
cfg.mean_recurrence      = 8
cfg.mean_backprop_depth  = 8
model = RecurrentQwenForCausalLM(cfg).to(base.dtype)
model.generation_config = base.generation_config

model.model.embed_tokens      = base.model.embed_tokens
for i in range(len(base.model.layers)):
    model.model.layers[i] = base.model.layers[i]
model.model.norm              = base.model.norm
model.model.rotary_emb              = base.model.rotary_emb
model.lm_head.weight.data.copy_(base.lm_head.weight.data)

model.save_pretrained(dst_path)
tok = AutoTokenizer.from_pretrained(src_path, trust_remote_code=True)
tok.save_pretrained(dst_path)
print("📦  已导出 ->", dst_path)