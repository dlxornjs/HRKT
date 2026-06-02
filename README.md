# HRKT-SAKT
This repository provides a PyTorch reference implementation of **HRKT-SAKT**, 
the SAKT-based version of Hierarchical Recurrent Knowledge Tracing (HRKT).

HRKT is designed for efficient long-sequence knowledge tracing by separating 
local interaction modeling and global knowledge evolution through a hierarchical 
recurrent Transformer architecture.

## Architecture

HRKT consists of three main components:

1. **Bottom Layer**  
   Processes long interaction sequences chunk by chunk using recurrent memory tokens.
   It captures local question-response interaction patterns within each chunk.

2. **Top Layer**  
   Collects chunk-level write memories and models long-term knowledge evolution
   through memory self-attention and cross-attention.

3. **Fusion Layer**  
   Adaptively combines local representations from the Bottom Layer and global
   representations from the Top Layer using a gated fusion mechanism.

<p align="center">
  <img src="figures/hrkt_overview.png" width="700">
</p>

<p align="center">
  <img src="figures/hrkt_internal_architecture.png" width="700">
</p>

## File Structure

```text
HRKT/
├── README.md
├── figures/
│   ├── fig2_hrkt_overview.png
│   └── fig3_hrkt_internal_architecture.png
└── models/
    └── hrkt_sakt.py
