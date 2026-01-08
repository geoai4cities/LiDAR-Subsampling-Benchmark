# Generated Config Summary - SemanticKITTI

**Generated:** 12 configs
**Tier:** tier1
**GPU Types:** 140gb
**Seeds:** 1 (for non-deterministic methods)

## Method Properties

| Method | Deterministic | Seed Required | Output Directory |
|--------|---------------|---------------|------------------|
| RS | No | Yes | `RS_loss{XX}_seed{N}/` |
| IDIS | Yes | No | `IDIS_loss{XX}/` |
| IDIS_R5 | Yes | No | `IDIS_R5_loss{XX}/` |
| IDIS_R15 | Yes | No | `IDIS_R15_loss{XX}/` |
| IDIS_R20 | Yes | No | `IDIS_R20_loss{XX}/` |

## Configuration Breakdown

| Category | Methods | Loss Levels | Seeds | Configs/GPU |
|----------|---------|-------------|-------|-------------|
| Deterministic | 4 | 6 | N/A | 6 |
| Non-deterministic | 1 | 6 | 1 | 6 |
| **Total** | 5 | 6 | - | **12** |

## Parameters

### Deterministic Methods (4 total)
- IDIS, IDIS_R5, IDIS_R15, IDIS_R20
- No seed needed - same input always produces same output

### Non-deterministic Methods (1 total)
- RS
- Seeds: 1

### Loss Levels (6 total)
- **Baseline:** 0% (original data)
- **Subsampled:** 10, 30, 50, 70, 90%

## Config Naming Convention

### Deterministic Methods
```
ptv3_semantickitti_{METHOD}_loss{LOSS}_{GPU}.py
```

### Non-deterministic Methods
```
ptv3_semantickitti_{METHOD}_loss{LOSS}_seed{SEED}_{GPU}.py
```

**Examples:**
- `ptv3_semantickitti_IDIS_loss50_140gb.py` - IDIS (deterministic)
- `ptv3_semantickitti_RS_loss50_seed1_140gb.py` - RS (non-deterministic)
- `ptv3_semantickitti_FPS_loss50_seed2_140gb.py` - FPS (non-deterministic)

## Output Directory

```
PTv3/SemanticKITTI/configs/semantickitti/generated
```
