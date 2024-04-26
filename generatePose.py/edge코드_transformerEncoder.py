class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int, # 모델 차원
        nhead: int, # 멀티헤드 어텐션의 헤드 수
        dim_feedforward: int = 2048,    # 피드포워드 네트워크 차원
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,    # 활성화 함수
        layer_norm_eps: float = 1e-5,   # 레이어 정규화 epsilon 값
        batch_first: bool = False,  # 배치 차원 우선 여부
        norm_first: bool = True,    # 정규화 먼저 적용할지 여부
        device=None,
        dtype=None,
        rotary=None,     # 로터리 포지셔닝 임베딩 객체
    ) -> None:
        super().__init__()  # nn.Module의 생성자 호출
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )    # 멀티헤드 어텐션 구현
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary    # 로터리 포지셔닝 임베딩
        self.use_rotary = rotary is not None    # 로터리 포지셔닝 사용 여부 결정

        # 로터리 위치 임베딩은 회전하는 오브젝트의 위치를 나타내는 데 사용되는 임베딩 기술
        # 로터리 위치 임베딩을 생성하는 가장 일반적인 방법 중 하나는 회전하는 오브젝트의 각도를 캡처하는 것
        # 복잡한 회전을 다루기 위해 쿼터니언(quaternion)이나 회전 매트릭스(rotation matrix)와 같은 고급 기법을 사용

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
