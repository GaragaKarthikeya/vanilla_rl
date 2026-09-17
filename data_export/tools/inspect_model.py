from common import *
import zipfile
from c345 import make_env, load_model
for s in SEEDS:
    z = zipfile.ZipFile(REPO / CKPT[s]); d = json.loads(z.read("data"))
    keys = ["learning_rate","gamma","gae_lambda","clip_range","clip_range_vf","ent_coef","vf_coef","max_grad_norm","n_steps","batch_size","n_epochs","n_envs","num_timesteps","_total_timesteps","normalize_advantage","target_kl","seed","policy_kwargs","lr_schedule","_n_updates","use_sde","device"]
    print(s, {k: (d[k] if not isinstance(d.get(k), dict) or ':serialized:' not in d[k] else {kk:vv for kk,vv in d[k].items() if kk!=':serialized:'}) for k in keys if k in d})
env = make_env("diffeq1")
m = load_model(42, env)
print(m.policy)
tot = sum(p.numel() for p in m.policy.parameters() if p.requires_grad)
fe = sum(p.numel() for p in m.policy.features_extractor.parameters())
print("total_trainable", tot, "features_extractor", fe, "share_features_extractor", m.policy.share_features_extractor)
print("lr schedule at 0 / 1:", m.lr_schedule(1.0), m.lr_schedule(0.0), "clip", m.clip_range(1.0), m.clip_range(0.0))
