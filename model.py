from adapters import SAKTAdapter
from hrkt import HRKT


class HRKT_SAKT(HRKT):
    """
    HRKT-SAKT model.

    This class wraps the SAKT adapter with the HRKT framework.
    """

    def __init__(
        self,
        num_q,
        max_len,
        d_model=64,
        num_heads=4,
        dropout=0.2,
        chunk_len=50,
        mem_len=4,
        top_dropout=0.3,
        top_d_ff_ratio=4,
    ):
        base_model = SAKTAdapter(
            num_q=num_q,
            d=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        super().__init__(
            base_model=base_model,
            chunk_len=chunk_len,
            mem_len=mem_len,
            dropout=dropout,
            top_dropout=top_dropout,
            top_d_ff_ratio=top_d_ff_ratio,
        )

        self.num_q = num_q
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
