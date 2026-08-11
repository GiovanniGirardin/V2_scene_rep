# V2 Scene Rep Transformer

Implementazione PyTorch di **Soft Actor-Critic (SAC)** per guida autonoma in **SMARTS**, con:

- **MST (Multi-Stage Transformer)** come encoder della scena;
- **SLT (Sequential Latent Transformer)** come loss ausiliaria per representation learning;
- osservazioni multi-agente con storia cinematica e waypoint;
- replay buffer SAC e replay separata per sequenze SLT;
- strumenti per debug SMARTS, training, checkpoint e valutazione.

Il progetto è SMARTS-only. Non è una copia line-by-line della repository TensorFlow originale: riprende l’idea di combinare representation learning e reinforcement learning in un’implementazione PyTorch moderna.

Repository originale: <https://github.com/georgeliu233/Scene-Rep-Transformer>

---

## Stato attuale

Il profilo locale di riferimento è:

```text
configs/default.yaml
```

Usa SMARTS reale (`use_dummy: false`), reward basato sul progresso, MST e SLT abilitati.

> **CPU:** usare il profilo default per validare l’ambiente, gli update e i checkpoint. Un training conclusivo da 100k step con SMARTS + Transformer non è realistico su CPU limitata. Il confronto quantitativo sarà eseguito successivamente su GPU.

Per il piano operativo completo consultare [`next_steps.md`](next_steps.md).

---

## Architettura

```text
SMARTS
  │
  ▼
ObservationAdapter
  ├─ motion:    [A, H, 5]
  ├─ waypoints: [A, R, W, D_w]
  ├─ agent_mask
  └─ route_mask
  │
  ▼
MSTEncoder ──────────────► Actor ──────────────► azione SAC
  │
  ├──────────────────────► Critic 1 / Critic 2
  │
  └──────────────────────► SLT (solo training)
                              │
                              ▼
                   predizione di latenti futuri
                   condizionata dalle azioni
```

### SAC

`scene_rep/models/sac.py` contiene:

- actor gaussiano squashed;
- due critic Q;
- temperatura entropica appresa (`alpha`);
- target encoder e target critic con Polyak averaging;
- encoder MST condiviso da critic e SLT.

L’actor riceve un latente detached: l’actor loss non aggiorna l’encoder. L’MST è invece aggiornato dal critic e, con SLT attivo, dalla loss ausiliaria.

### MST

`scene_rep/models/mst_encoder.py` integra:

- storia del moto di ego e vicini;
- waypoint e route candidate;
- maschere per agenti/route in padding.

L’output è un latente `[B, latent_dim]`.

### SLT

`scene_rep/models/slt.py` è usato solo in training:

1. riceve una sequenza di latenti e azioni;
2. usa un Transformer causale;
3. predice i latenti futuri;
4. ottimizza negative cosine similarity in stile SimSiam;
5. applica stop-gradient ai target.

La pipeline usa `target_encoder` per generare i target SLT. Il target encoder e i target critic sono aggiornati solo dopo gli update SAC e SLT del passo.

### Ordine degli update

```text
update SAC (critic + MST, actor, alpha)
                 │
                 ▼
update SLT (MST + SLT)
                 │
                 ▼
EMA target encoder + target critics
```

L’MST ha un solo optimizer Adam (`encoder_optimizer`). L’optimizer SLT gestisce solo i parametri dell’SLT, evitando due stati Adam diversi sugli stessi pesi MST.

---

## Osservazioni e azioni

### Motion

```text
motion: [num_agents, history_len, 5]
[x, y, vx, vy, heading]
```

- Slot `0`: ego.
- Slot successivi: vicini ordinati per distanza.
- `agent_mask` distingue dati reali e padding.

### Waypoints

```text
waypoints: [num_agents, max_candidate_routes, waypoint_len, waypoint_dim]
```

Con `waypoint_dim: 3`:

```text
[x, y, heading]
```

Con `coordinate_frame: absolute` e `waypoint_dim: 5`:

```text
[x, y, heading, is_ego, is_neighbor]
```

Sono disponibili due frame:

- `ego`: coordinate relative all’ego;
- `absolute`: coordinate globali SMARTS e cronologia stabile per vehicle ID.

### Azioni

La policy produce:

```text
[speed_normalized, lane_signal] ∈ [-1, 1]^2
```

`ActionAdapter` converte:

- speed normalizzata → `[0, max_speed_mps]`;
- lane signal → `-1` (sinistra), `0` (mantieni), `+1` (destra).

---

## Reward e terminazioni

Il wrapper SMARTS è `scene_rep/envs/smarts_env.py`.

### Reward `progress`

```text
reward =
  step_penalty
  + progress_reward_scale * position_delta
  + position_delta_reward_scale * position_delta
  + reward/penalty terminali
```

`position_delta` è lo spostamento euclideo dell’ego tra osservazioni consecutive.

SMARTS 2.x può esporre `distance_travelled` come valore per osservazione/non monotono; il wrapper non calcola più il progresso differenziando quel valore. Usa direttamente `position_delta`, il cui significato per-step è stabile.

### Informazioni di episodio

`info` contiene:

- `success`, `collision`, `off_route`, `stagnation`, `timeout`;
- `terminal_reason`;
- `progress`, `position_delta`, `distance_travelled`;
- azione SMARTS adattata.

I timeout vengono inseriti come non terminali nella replay SAC, in modo da permettere il bootstrap.

---

## Struttura

```text
V2_scene_rep/
├── configs/
│   ├── default.yaml
│   ├── runpod_debug.yaml
│   ├── runpod_4090.yaml
│   ├── runpod_4090_seed42_config.yaml
│   └── scenarios/
├── scene_rep/
│   ├── data/
│   │   ├── replay_buffer.py
│   │   ├── future_queue.py
│   │   └── sequence_buffer.py
│   ├── envs/
│   │   ├── action_adapter.py
│   │   ├── observation_adapter.py
│   │   └── smarts_env.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── rollout.py
│   ├── models/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── mst_encoder.py
│   │   ├── sac.py
│   │   ├── slt.py
│   │   └── transformer_blocks.py
│   ├── training/
│   │   ├── augmentation.py
│   │   ├── checkpointing.py
│   │   ├── logger.py
│   │   └── trainer.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── torch_utils.py
├── scripts/
│   ├── debug_smarts_env.py
│   ├── evaluate.py
│   ├── evaluate_checkpoints.py
│   ├── plot_training.py
│   └── train.py
├── checkpoints/
├── logs/
├── requirements.txt
└── next_steps.md
```

---

## Installazione

### Prerequisiti

- Linux;
- Python 3.10+;
- SMARTS 2.x;
- SUMO;
- scenario SMARTS disponibile localmente.

Il `requirements.txt` corrisponde all’ambiente usato e include, tra le altre:

```text
smarts==2.0.1
eclipse-sumo==1.26.0
gymnasium==1.3.0
torch==2.11.0
```

### Ambiente Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pyproject.toml` dichiara dipendenze minime del package; per eseguire SMARTS installare anche `requirements.txt`.

### SUMO

Se SUMO è installato nel virtualenv, impostare il percorso corretto della macchina:

```bash
export SUMO_HOME="/percorso/venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

### Percorsi scenario

I YAML includono percorsi assoluti, ad esempio:

```yaml
smarts:
  scenario: "/home/giovanni/SMARTS/scenarios/sumo/intersections/1_to_2lane_left_turn_t_agents_1"
```

Prima di eseguire su un’altra macchina, aggiornare `smarts.scenario`.

---

## Configurazioni

### `configs/default.yaml`

Profilo locale/CPU di riferimento:

- SMARTS reale;
- reward `progress`;
- MST + SLT attivi;
- `future_horizon: 3`;
- batch SAC e SLT: `16`;
- warm-up: `5000` step;
- training configurato per `100000` step.

Non usare automaticamente tutti i 100k step su CPU: seguire prima [`next_steps.md`](next_steps.md).

### `configs/runpod_debug.yaml`

Smoke test breve. Usa reward sparse e pochi step: verifica che il codice non crashi, ma non dimostra apprendimento.

### `configs/runpod_4090.yaml`

Profilo GPU con scenario left-turn, osservazioni assolute, action repeat, batch maggiori e training lungo. Richiede una GPU performante e percorsi scenario validi nella macchina remota.

### Ereditarietà YAML

Il config loader supporta:

```yaml
inherit: default.yaml
```

Le chiavi del file figlio sovrascrivono ricorsivamente quelle del file base. Non esistono override di singoli parametri da CLI: ogni esperimento deve avere un file YAML versionato.

---

## Verifica SMARTS

Prima del training:

```bash
python scripts/debug_smarts_env.py \
  --config configs/default.yaml \
  --steps 30 \
  --episodes 1 \
  --speed-mps 5.0
```

Controllare:

- forme di `motion` e `waypoints`;
- `agent_mask` dell’ego attiva;
- reward e `progress > 0` quando il veicolo avanza;
- terminazioni plausibili.

Azioni casuali:

```bash
python scripts/debug_smarts_env.py \
  --config configs/default.yaml \
  --steps 100 \
  --random-actions
```

Rendering Envision:

```bash
python scripts/debug_smarts_env.py \
  --config configs/default.yaml \
  --steps 300 \
  --speed-mps 5.0 \
  --render \
  --sleep-sec 0.1
```

---

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

Senza argomenti, `train.py` usa `configs/default.yaml`.

Flusso:

1. reset ambiente;
2. warm-up con azioni casuali;
3. inserimento transizioni nel `ReplayBuffer`;
4. costruzione finestre consecutive tramite `FutureQueue`;
5. inserimento nel `SequenceBuffer`;
6. update SAC;
7. update SLT opzionale;
8. update target EMA;
9. logging e checkpoint.

Output:

- checkpoint: `training.checkpoint_dir`;
- metriche episodiche: `logs/training_episodes.csv`;
- checkpoint: `sac_step_<step>.pt`.

---

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/sac_step_2000.pt \
  --episodes 10 \
  --out logs/evaluation_results.csv
```

Per tracciare azioni e reward:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/sac_step_2000.pt \
  --episodes 3 \
  --trace-actions \
  --log-every-steps 10
```

L’evaluation usa azioni deterministiche e aggiunge una riga ai risultati CSV.

---

## Scenari

| Scenario | Uso |
|---|---|
| `loop` | sanity check di rollout, osservazioni e azioni |
| `double_merge` | task intermedio con traffico e cambi corsia |
| `left_turn` | task più difficile, da affrontare dopo validazione su task più semplici |
| `roundabout` | interazioni multi-veicolo |
| `unprotected_left_turn` | scenario avanzato e sensibile al traffico |

Non confrontare curve ottenute con scenario, reward, seed o `action_repeat` differenti.

---

## Troubleshooting

### Scenario non trovato

Controllare `smarts.scenario`: deve essere una directory SMARTS valida.

### SMARTS non trova SUMO

Controllare `SUMO_HOME`, `PYTHONPATH` e l’installazione `eclipse-sumo`.

### L’ego avanza ma reward/progress sono quasi zero

Eseguire `debug_smarts_env.py`. Il wrapper corrente usa `position_delta` per il progresso; un veicolo in movimento deve mostrare `progress > 0`.

### SLT peggiora SAC

Verificare:

1. stesso seed/scenario/reward per le ablation;
2. critic loss e SLT loss finite;
3. sequenze SLT effettivamente presenti;
4. `loss_weight` iniziale conservativo, ad esempio `0.01`;
5. target encoder, optimizer condiviso dell’MST e update target post-SLT presenti nel codice corrente.

### Nessun apprendimento in un run breve

Un breve run CPU è un controllo funzionale, non una prova di fallimento o successo. In particolare, `runpod_debug.yaml` è uno smoke test e non un benchmark.

---

## Riferimenti

- Originale: <https://github.com/georgeliu233/Scene-Rep-Transformer>
- SMARTS: <https://github.com/huawei-noah/SMARTS>
- Piano operativo: [`next_steps.md`](next_steps.md)
