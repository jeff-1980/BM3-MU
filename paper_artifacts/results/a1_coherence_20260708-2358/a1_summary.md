# A1 channel-coherence analysis summary
Output dir: `results/a1_coherence_20260708-2358`
## Coverage
- CWRU: 16 (fault_type x load) cells, DE/FE, fault bands from DE_BEARING geometry (project convention).
- XJTU: 5 (condition x fault_type) cells over 11 valid OR/IR bearings, H/V, fault-phase windows only.

## Fault-band coherence, by dataset
- CWRU: mean=0.412, min=0.307, max=0.492 across 16 cells.
- XJTU: mean=0.417, min=0.337, max=0.554 across 5 cells.

- Sanity control: CWRU `normal` cells (no seeded fault) mean fault-band SNR_A=0.62 dB vs faulted cells mean SNR_A=2.65 dB.

## Gain vs. coherence (task 3)
Matched (coherence, gain) pairs -- see `gain_vs_coherence_points.csv` and the design-notes docstring in this script for exactly how each point was constructed (granularity mismatch between available gain data and computed coherence cells forced this specific pairing; documented, not hidden).
- CWRU / SNR=0dB: coherence=0.013, gain=+0.14 pp
- CWRU / SNR=-2dB: coherence=0.008, gain=+0.05 pp
- CWRU / SNR=-4dB: coherence=0.004, gain=+0.70 pp
- CWRU / SNR=-6dB: coherence=0.001, gain=+1.75 pp
- CWRU / SNR=-8dB: coherence=0.001, gain=+2.53 pp
- XJTU / LOBO(Cond3): coherence=0.412, gain=-15.53 pp
- XJTU / Cross(Cond2->Cond3): coherence=0.429, gain=+14.01 pp

**Spearman (all 7 points): r=-0.214, p=0.645.**
**Spearman (CWRU-only SNR sweep, 5 points): r=-0.900, p=0.037.**

N is very small (5-7 points, 2 of them from a different dataset/metric than the other 5); this is directional evidence only, not a confirmed relationship. No claim beyond what these numbers show is made.
