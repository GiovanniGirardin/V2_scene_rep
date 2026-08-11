# Next steps — Validazione locale CPU e preparazione training GPU

Questo documento definisce il percorso da completare **prima** di lanciare un training lungo su GPU.

Obiettivo della fase locale: verificare che ambiente, reward, buffer, update SAC/SLT, checkpoint ed evaluation siano coerenti. Non usare una curva CPU breve come misura dell’efficacia dell’SLT.

---

## Stato di partenza

Le correzioni già applicate alla pipeline sono:

- il reward di progresso usa `position_delta` invece di differenziare `distance_travelled` SMARTS;
- l’encoder MST condiviso ha un solo optimizer Adam;
- SLT genera target futuri dal `target_encoder`;
- i target SAC/SLT vengono aggiornati dopo gli update SAC e SLT;
- lo scenario left-turn configurato è una directory SMARTS con missione ego valida;
- i test sintetici SAC + SLT e i rollout SMARTS reali hanno completato senza errori.

Il profilo locale corrente è `configs/default.yaml`:

```yaml
project.device: auto
smarts.scenario: /home/giovanni/SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_t_agents_1
reward.mode: progress
sac.batch_size: 16
sac.warmup_steps: 5000
slt.enabled: true
slt.future_horizon: 3
slt.loss_weight: 0.1
training.total_steps: 100000
```

`total_steps: 100000` non deve essere eseguito integralmente su CPU limitata.

---

# Fase 0 — Preparazione dell’ambiente locale

## 0.1 Attivare l’ambiente Python

```bash
source venv/bin/activate
python --version
python -c "import torch, smarts, gymnasium; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('smarts:', smarts.__version__)"
```

### Criterio di accettazione

- Python 3.10+;
- import di `torch`, `smarts` e `gymnasium` riusciti;
- `cuda: False` è normale sulla macchina locale CPU.

## 0.2 Verificare SUMO

Configurare i path soltanto per la sessione corrente, sostituendo il percorso se necessario:

```bash
export SUMO_HOME="/percorso/venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
test -d "$SUMO_HOME" && echo "SUMO_HOME valido"
```

### Criterio di accettazione

- `SUMO_HOME` punta a una directory esistente;
- SMARTS avvia il simulatore nel test della fase successiva.

## 0.3 Verificare il percorso scenario

```bash
grep -nE 'scenario:|use_dummy:|device:' configs/default.yaml
```

### Criterio di accettazione

- `use_dummy: false`;
- `scenario` punta alla directory SMARTS presente localmente;
- il percorso non deve essere copiato immutato su GPU remota: andrà adattato alla nuova macchina.

---

# Fase 1 — Validazione ambiente e reward

## 1.1 Rollout deterministico

```bash
python scripts/debug_smarts_env.py \
  --config configs/default.yaml \
  --steps 30 \
  --episodes 1 \
  --speed-mps 5.0
```

### Verificare

- `motion` ha forma `(4, 4, 5)` per il default:
  - 1 ego + massimo 3 vicini;
  - history di 4 frame;
  - 5 feature cinematiche.
- `waypoints` ha forma `(4, 3, 10, 3)`.
- l’ego è attivo in `agent_mask`;
- quando il veicolo avanza, `progress > 0`;
- reward e return sono finiti (`not NaN`, `not inf`);
- ogni terminazione espone un `terminal_reason`.

### Criterio di accettazione

Il rollout termina correttamente o raggiunge il limite di step senza exception. Il progresso non deve essere sistematicamente zero durante movimento dell’ego.

## 1.2 Rollout casuale

```bash
python scripts/debug_smarts_env.py \
  --config configs/default.yaml \
  --steps 100 \
  --episodes 3 \
  --random-actions
```

### Verificare

- collisioni/off-route sono riportati correttamente;
- un episodio successivo parte da un nuovo reset;
- non si osservano errori del tipo key error, shape mismatch o SMARTS sensor failure persistente.

### Criterio di accettazione

Tre episodi completano senza crash. Eventi negativi sono normali con azioni casuali; errori di integrazione non lo sono.

## 1.3 Ispezione visuale opzionale

```bash
python scripts/debug_smarts_env.py \
  --config configs/default.yaml \
  --steps 300 \
  --speed-mps 5.0 \
  --render \
  --sleep-sec 0.1
```

### Verificare

- ego, mappa e traffico sono visualizzati;
- la velocità impostata viene applicata;
- l’azione keep-lane non genera comandi invertiti.

### Criterio di accettazione

La visualizzazione è utile ma non bloccante: in un ambiente senza display/Envision funzionante, la validazione headless è sufficiente.

---

# Fase 2 — Test funzionali PyTorch

## 2.1 Compilazione moduli

```bash
python -m py_compile \
  scene_rep/envs/smarts_env.py \
  scene_rep/models/sac.py \
  scene_rep/models/slt.py \
  scene_rep/training/trainer.py \
  scripts/debug_smarts_env.py
```

## 2.2 Integrità diff

```bash
git diff --check
git status --short
```

### Criterio di accettazione

- nessun errore Python;
- nessun whitespace error Git;
- i file modificati sono noti e intenzionali.

## 2.3 Smoke test sintetico SAC + SLT

Eseguire il test completo degli update senza attendere SMARTS:

```bash
python - <<'PY'
import torch

from scene_rep.models.sac import SACAgent
from scene_rep.utils.config import load_config

cfg = load_config("configs/default.yaml")
cfg["project"]["device"] = "cpu"
cfg["model"]["dropout"] = 0.0
cfg["sac"]["batch_size"] = 2
cfg["slt"]["batch_size"] = 2

agent = SACAgent(cfg)

batch_size = 2
horizon = cfg["slt"]["future_horizon"] + 1
agents = cfg["observation"]["max_neighbors"] + 1
history = cfg["observation"]["history_len"]
routes = cfg["observation"]["max_candidate_routes"]
waypoints = cfg["observation"]["waypoint_len"]
motion_dim = cfg["model"]["motion_dim"]
waypoint_dim = cfg["model"]["waypoint_dim"]

def observation(n):
    return {
        "motion": torch.randn(n, agents, history, motion_dim),
        "waypoints": torch.randn(n, agents, routes, waypoints, waypoint_dim),
        "agent_mask": torch.ones(n, agents),
        "route_mask": torch.ones(n, agents, routes),
    }

sac_batch = {
    "obs": observation(batch_size),
    "next_obs": observation(batch_size),
    "actions": torch.rand(batch_size, 2) * 2 - 1,
    "rewards": torch.randn(batch_size, 1),
    "dones": torch.zeros(batch_size, 1),
}
sequence_batch = {
    "motion": torch.randn(batch_size, horizon, agents, history, motion_dim),
    "waypoints": torch.randn(
        batch_size, horizon, agents, routes, waypoints, waypoint_dim
    ),
    "agent_mask": torch.ones(batch_size, horizon, agents),
    "route_mask": torch.ones(batch_size, horizon, agents, routes),
    "actions": torch.rand(batch_size, horizon, 2) * 2 - 1,
}

print(agent.update(sac_batch))
print(agent.update_slt(sequence_batch))
agent.soft_update_targets()
print("PASS")
PY
```

### Criterio di accettazione

- stampa metriche SAC e SLT finite;
- termina con `PASS`;
- nessun errore su optimizer, shape o backward graph.

---

# Fase 3 — Breve run integrato su CPU

## 3.1 Non usare il default invariato per un test breve

Il default ha:

```yaml
warmup_steps: 5000
total_steps: 100000
```

Su CPU ciò renderebbe il test troppo costoso e non informativo.

Creare una config di validazione **versionata**, ad esempio `configs/local_validation.yaml`, con `inherit: default.yaml` e override mirati:

```yaml
inherit: default.yaml

project:
  name: V2_scene_rep_local_validation
  device: cpu

sac:
  replay_size: 1000
  batch_size: 4
  warmup_steps: 20

slt:
  batch_size: 4
  sequence_replay_size: 256
  loss_weight: 0.01

training:
  total_steps: 100
  eval_every_steps: 50
  save_every_steps: 50
  log_every_steps: 10
  checkpoint_dir: checkpoints/local_validation
```

> Se 100 step sono troppo lenti, ridurre temporaneamente `total_steps` a 50. Non modificare il default per trasformarlo in un esperimento non riproducibile.

## 3.2 Avviare il run

```bash
python scripts/train.py --config configs/local_validation.yaml
```

### Osservare

- il warm-up completa;
- la replay SAC supera batch size;
- il `FutureQueue` genera sequenze dopo abbastanza transizioni consecutive;
- le metriche SAC compaiono senza NaN/inf;
- le metriche SLT compaiono quando la sequence replay ha batch sufficiente;
- i checkpoint `sac_step_*.pt` vengono salvati;
- il trainer chiude SMARTS senza exception.

### Criterio di accettazione

Un run di 50–100 step completa con almeno un checkpoint e senza `NaN`, `inf`, errore di shape o errore di optimizer.

> Questo run non deve dimostrare apprendimento. Serve soltanto a validare l’integrazione completa SMARTS → buffer → SAC → SLT → checkpoint.

---

# Fase 4 — Checkpoint ed evaluation locale

Dopo il run breve, identificare un checkpoint:

```bash
find checkpoints/local_validation -name 'sac_step_*.pt' -type f | sort
```

Valutarne uno:

```bash
python scripts/evaluate.py \
  --config configs/local_validation.yaml \
  --checkpoint checkpoints/local_validation/sac_step_50.pt \
  --episodes 2 \
  --out logs/local_validation_evaluation.csv \
  --trace-actions \
  --log-every-steps 10
```

Adattare il numero di step al checkpoint realmente prodotto.

### Criterio di accettazione

- il checkpoint viene caricato;
- evaluation completa almeno due episodi;
- il CSV viene creato o aggiornato;
- azioni, reward e progresso sono finiti.

---

# Fase 5 — Ablation da preparare per GPU

Dopo che tutte le fasi CPU sono approvate, creare config GPU esplicite. Non confrontare run che differiscono in più fattori contemporaneamente.

## 5.1 Esperimenti minimi

| ID | MST | SLT | Scopo |
|---|---:|---:|---|
| `gpu_sac_only` | sì | no | baseline SAC + MST |
| `gpu_sac_slt_001` | sì | sì, `loss_weight: 0.01` | primo confronto conservativo |
| `gpu_sac_slt_01` | sì | sì, `loss_weight: 0.1` | test dell’impostazione corrente |
| `gpu_no_mst` | no, se supportato | no | ablation encoder, solo dopo baseline stabile |

Ogni config deve fissare:

- stesso scenario;
- stessa modalità reward e relativi coefficienti;
- stessa action repeat;
- stessa dimensione observation/model;
- stesso budget step;
- stessi seed, oppure una lista condivisa di seed.

## 5.2 Seed

Eseguire almeno 3 seed per ogni ablation quando il budget lo permette:

```text
42, 43, 44
```

Confrontare media e variabilità; non dedurre un miglioramento da un solo seed.

## 5.3 Metriche da salvare

Per step/episodio:

- return;
- episode length;
- success rate;
- collision rate;
- off-route rate;
- stagnation rate;
- timeout rate;
- critic loss;
- actor loss;
- alpha;
- Q-value medio;
- SLT loss e SLT weighted loss;
- throughput (steps/s) e memoria GPU, se disponibile.

Per checkpoint:

- evaluation deterministica su un numero fisso di episodi;
- stesso scenario e seed policy/ambiente, quando supportato.

---

# Fase 6 — Preparazione macchina GPU

## 6.1 Requisiti

- GPU NVIDIA performante;
- driver/CUDA compatibili con la build di PyTorch;
- SMARTS, SUMO e dipendenze Python installati;
- spazio per checkpoint, log e output SUMO;
- scenari SMARTS copiati o montati;
- display non necessario per training headless.

## 6.2 Controlli iniziali

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import smarts, gymnasium; print('SMARTS import OK')"
```

### Criterio di accettazione

- PyTorch rileva CUDA;
- SMARTS si importa;
- il percorso scenario della config GPU esiste sulla macchina remota;
- un rollout headless con `debug_smarts_env.py` funziona prima del training.

## 6.3 Rendere portabile la config

Non copiare percorsi assoluti locali senza verificarli. Creare un YAML GPU dedicato con:

```yaml
project:
  device: cuda

smarts:
  headless: true
  scenario: "/percorso/reale/su/gpu/scenario"
```

Conservare config, commit Git e output `pip freeze` insieme ai risultati.

---

# Fase 7 — Protocollo training GPU

## 7.1 Ordine consigliato degli scenari

1. `loop` per sanity check operativo;
2. `double_merge` per test intermedio;
3. `left_turn` per task difficile;
4. `roundabout` e `unprotected_left_turn` solo dopo risultati stabili.

Lo scenario left-turn corrente è valido, ma una policy keep-lane può collidere rapidamente: non è il primo benchmark consigliato per attribuire merito o colpa all’SLT.

## 7.2 Ordine delle ablation

1. baseline SAC + MST, SLT disabilitato;
2. SAC + MST + SLT con `loss_weight: 0.01`;
3. SAC + MST + SLT con `loss_weight: 0.1`;
4. eventuale sweep di:
   - `future_horizon`: 3, 5, 10;
   - `loss_weight`: 0.003, 0.01, 0.03, 0.1;
   - `updates_per_step`;
   - batch size SLT.

Cambiare una sola famiglia di iperparametri alla volta.

## 7.3 Condizioni di stop

Interrompere e analizzare il run se si osserva uno dei seguenti:

- NaN/inf in loss, Q-values o azioni;
- reward/progress incompatibili con lo spostamento dell’ego;
- sequence buffer mai popolato;
- SLT loss non finita;
- crollo persistente dei Q-values subito dopo l’attivazione SLT;
- checkpoint non caricabile;
- simulator crash ricorrente.

---

# Decisione di avanzamento

Procedere al training GPU lungo soltanto se tutte le condizioni seguenti sono vere:

- [ ] rollout SMARTS deterministico e casuale passano;
- [ ] reward/progress sono coerenti con il movimento;
- [ ] test sintetico SAC + SLT passa;
- [ ] run integrato CPU breve completa;
- [ ] checkpoint creato e ricaricato con evaluation;
- [ ] nessuna loss/azione/Q-value non finita;
- [ ] config GPU portabile e versionata;
- [ ] baseline SAC-only pianificata prima dell’ablation SLT;
- [ ] seed e metriche da confrontare definiti.

Quando tutti i punti sono soddisfatti, il codice è pronto per la fase sperimentale GPU.
