class DanceDecoder(nn.Module):
    def __init__(
            self,
            nfeats: int,  # 특징의 개수
            seq_len: int = 150,  # 5 seconds, 30 fps
            latent_dim: int = 256,  # 잠재 공간의 차원
            ff_size: int = 1024,  # 피드포워드 네트워크의 차원
            num_layers: int = 4,  # 디코더 스택 내의 레이어 수
            num_heads: int = 4,
            dropout: float = 0.1,
            cond_feature_dim: int = 4800,  # 조건부 특징의 차원
            activation: Callable[[Tensor], Tensor] = F.gelu,  # relu의 비슷한 형태로 입력값이 음수일때 0이 되지않고 부드럽게 감소한다
            use_rotary=True,  # 회전 위치 인코딩의 사용 여부를 결정
            **kwargs
    ) -> None:

        super().__init__()  # 부모 클래스인 nn.Module의 생성자를 호출하여, PyTorch 모듈의 기본 초기화를 수행

        output_feats = nfeats  # 모델 출력의 차원 수를 결정

        # positional embeddings
        self.rotary = None  # 회전 위치 임베딩(rotary positional embedding) 인스턴스를 저장할 속성 초기값은 None으로 설정
        self.abs_pos_encoding = nn.Identity()  # 절대 위치 임베딩(absolute positional encoding)을 저장할 속성
        # nn.Identity()는 기본적으로 입력을 변경 없이 그대로 전달하는 연산을 의미
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:  # 조건문은 use_rotary 매개변수의 값에 따라 위치 임베딩 방식을 결정
            self.rotary = RotaryEmbedding(dim=latent_dim)  # 회전 위치 임베딩은 각 토큰의 위치 정보를 회전을 통해 인코딩하는 방식
        else:
            self.abs_pos_encoding = PositionalEncoding(  # 절대 위치 임베딩은 각 위치에 고유한 벡터를 할당하여 위치 정보를 제공
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing     # 이 코드 부분은 시간 임베딩 처리를 위한 멀티 레이어 퍼셉트론(MLP)을 초기화
        self.time_mlp = nn.Sequential(  # 사인 및 코사인 함수를 사용하여 위치 정보를 인코딩하는 SinusoidalPosEmb 클래스의 인스턴스를 생성
            
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),  # ReLU와 비슷하지만 부드러운 곡선을 가지고 있어 기울기 소실 문제를 덜 겪으며, 더 나은 학습 성능을 보임
        )  # 시간적순서가 중요해서 sequential을 사용하여 각 요소들을 순차적으로 처리한다

        self.to_time_cond = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim), )  # 신경망 내에서 시간 임베딩을 조건부 정보로 변환하는 작은 네트워크를 정의
        #  time_mlp에서 생성된 시간 임베딩의 확장된 차원을 다시 원래의 잠재 차원으로 줄임 이는 시간 임베딩을 조건부 정보로 적합한 형태로 압축하고 정제하는 역할
        self.to_time_tokens = nn.Sequential(
            # 시간 관련 정보를 토큰화하는 레이어를 구성하는 과정, 시간 임베딩을 입력으로 받아, 이를 특정한 형태의 출력으로 변환하는 역할을 수행
            nn.Linear(latent_dim * 4, latent_dim * 2),
            # 2 time tokens #  여기서 생성된 출력은 시간 토큰, 각 토큰은 latent_dim 차원을 가지며, 시간적 정보를 요약한 것
            Rearrange("b (r d) -> b r d", r=2),  # Rearrange는 텐서의 차원을 재배열하는 데 사용
        )  # 입력된 2차원 벡터를 두 개의 시간 토큰을 갖는 3차원 텐서로 변환, b는 배치 크기를, r은 각 배치에 대한 토큰 수, d는 각 토큰의 차원을 나타냅
        # 이 연산은 모델이 더 쉽게 각 시간 토큰을 개별적으로 처리할 수 있게 해주며, 후속 레이어나 연산에서 각 토큰을 독립적인 입력으로 활용할 수 있도록 함
        # self.to_time_tokens은 복잡한 시간 데이터를 요약하고 구조화하는 방식으로 작동, 이는 모델이 시간에 따른 동적 변화를 보다 효과적으로 처리하고 이해할 수 있게 도와줌
        # 예를 들어, 댄스 동작이나 비디오 프레임과 같이 시간적 순서가 중요한 데이터에서, 모델이 특정 시점의 중요한 특징을 강조하거나 기억하는 데 사용

        # null embeddings for guidance dropout      # 가이드 드롭아웃(guide dropout)을 위한 널(null) 임베딩과 관련된 초기화 작업
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))  # nn.Parameter를 사용하여 학습 가능한 텐서를 정의,
        # 이 텐서는 임의의 정규 분포(torch.randn)에서 생성된 값으로 초기화
        # 이 파라미터는 훈련 중에 조건부 정보가 부족할 때 사용될 수 있는 "널" 임베딩으로, 드롭아웃과 유사한 기능을 하여 모델이 더 강인하게 정보를 처리하도록 도움
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))  # (1, latent_dim)의 크기를 갖는 학습 가능한 텐서
        # (hidden state)의 널 버전으로, 특정 상황에서 초기 상태 또는 참조 상태로 사용될 수 있음
        self.norm_cond = nn.LayerNorm(latent_dim)  # 입력 텐서의 평균과 분산을 조정하여, 훈련 중에 레이어의 입력이 더 안정적인 분포를 갖도록 도움

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()  # nn.Sequential()로 초기화된 빈 컨테이너, 조건부 인코딩을 위한 트랜스포머 인코더 레이어를 순차적으로 추가하여 구성
        for _ in range(2):
            self.cond_encoder.append(  # TransformerEncoderLayer 인스턴스를 cond_encoder에 추가
                TransformerEncoderLayer(
                    d_model=latent_dim,  # 모델의 특징 차원을 설정 latent_dim 중요
                    nhead=num_heads,
                    dim_feedforward=ff_size,  # 피드포워드 네트워크의 차원
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )  # 입력 데이터는 먼저 input_projection을 통해 적절한 차원으로 매핑되고, 이어서 cond_encoder에서 더 복잡한 특징 추출과 정보 통합이 이루어짐
            )
        # conditional projection    # 조건부 투영
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),  # 입력 데이터의 평균과 분산을 조정, 이는 훈련의 안정성과 수렴 속도를 향상
            nn.Linear(latent_dim, latent_dim),  # 입력을 받아 동일한 차원의 출력을 생성하지만, 내부적인 특징 변환을 수행
            nn.SiLU(),  # swish, 부드러운 비선형성 제공
            nn.Linear(latent_dim, latent_dim),  # 최종적인 조건부 정보의 표현을 다듬어, 다음 처리 단계로 넘기기 전에 최적화
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(
            decoderstack)  # DecoderLayerStack 클래스를 사용하여, 위에서 생성한 decoderstack 모듈 리스트를 디코더 스택으로 설정.
        # 이 스택은 입력 시퀀스를 순차적으로 각 디코더 레이어를 통과시키며, 최종적으로 변환된 시퀀스를 출력함

        self.final_layer = nn.Linear(latent_dim, output_feats)  # 디코더의 최종 출력을 원래의 특징 차원(output_feats)으로 변환
        # 입력 시퀀스에 대해 조건부 정보를 통합하고, 시퀀스를 효과적으로 디코딩한 다음, 원하는 출력 형태로 변환하는 복잡한 처리 과정을 수행할 수 있음

    def guided_forward(self, x, cond_embed, times,
                       guidance_weight):  # 입력에 따라 모델의 출력을 어떻게 조절할지에 대한 추가적인 제어를 가능하게 하는 방법을 제공
        #  cond_embed: 조건부 정보를 담은 임베딩, times: 시간적 차원이나 순서 정보,
        #  guidance_weight: 조건부 정보를 얼마나 강하게 사용할지를 결정하는 스케일링 인자
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)  # 무조건적 순전파 (Unconditioned Forward):
        # 모델의 forward 메소드를 호출하되, cond_drop_prob=1로 설정하여 조건부 정보를 전혀 사용하지 않도록 함. 즉, 이 호출은 입력 x와 시간 times에만 기반한 출력을 생성하며, 조건부 정보는 무시된다.
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)  # 조건부 순전파 (Conditioned Forward)
        # 동일한 입력에 대해 cond_drop_prob=0로 설정하여 조건부 정보를 완전히 사용하는 경우의 출력을 생성
        return unc + (
                    conditioned - unc) * guidance_weight  # unc와 conditioned 사이의 차이에 가이던스 가중치를 곱한 후, 이를 unc에 추가하여 최종 출력을 계산
        # guidance_weight는 conditioned 출력이 최종 결과에 미치는 영향을 조절

    # 가이드된 순전파 방식은 모델이 조건부 정보를 얼마나 반영할지를 동적으로 조절할 수 있게 해줌
    def forward(
            self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)  # 입력 데이터 x를 self.input_projection을 사용하여 잠재 차원으로 투영
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)  # 위치 정보(self.abs_pos_encoding)를 추가하여 시퀀스 데이터에 시간적 정보를 부여

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)  # 주어진 확률(1 - cond_drop_prob)에 따라
        # 배치 사이즈(batch_size)만큼의 True 또는 False 값을 갖는 마스크를 생성
        # cond_drop_prob가 0이면 모든 조건부 정보가 사용되고, 1이면 모든 조건부 정보가 사용X
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")  # b -> b 1 1"의 패턴은 원래 1차원 벡터였던 마스크를 3차원 텐서로 변환
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(
            cond_embed)  # 조건부 임베딩(cond_embed)을 self.cond_projection을 사용하여 잠재 차원(latent_dim)으로 매핑.
        # 이 선형 레이어는 조건부 정보를 모델의 내부 차원과 호환되도록 차원을 변환, 이는 후속 처리를 위해 조건부 정보를 적절한 형태로 표준화하는 첫 번째 단계임

        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)  # 위치 임베딩을 추가하여 각 토큰의 순서와 위치 정보를 모델에 제공
        cond_tokens = self.cond_encoder(cond_tokens)  # 위치 임베딩이 추가된 조건부 토큰을 self.cond_encoder를 통해 인코딩함

        null_cond_embed = self.null_cond_embed.to(
            cond_tokens.dtype)  # null_cond_embed의 데이터 타입을 현재 조건부 토큰(cond_tokens)의 데이터 타입과 일치시킴
        # 이는 연산 중에 데이터 타입의 불일치로 인한 문제를 방지
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)
        # keep_mask_embed는 각 조건부 토큰이 유지될지 여부를 결정하는 마스크, torch.where 함수는 조건에 따라 두 개의 입력 값 중 하나를 선택하는 조건부 텐서 연산을 수행
        # keep_mask_embed 값이 True인 경우 해당 위치에서 cond_tokens의 값을 유지
        # keep_mask_embed 값이 False인 경우 해당 위치에서 null_cond_embed의 값을 사용
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)  # 조건부 토큰(cond_tokens)에 대해 평균 풀링을 적용
        # 모델이 조건부 정보를 효과적으로 요약하고, 이를 모델의 다른 부분에 사용할 수 있도록 함
        #  여러 조건부 토큰들 간의 정보를 평균화하여 각 배치 항목에 대한 하나의 대표적인 특징 벡터를 생성
        # 평균 풀링은 시퀀스나 다양한 조건부 토큰들의 정보를 간략화하고, 중요한 정보를 보존하면서 데이터의 차원을 축소하는 방법
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)
        # 평균 풀링된 조건부 토큰을 입력으로 받아 추가적인 변환을 수행
        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)  # 시간 정보 times를 self.time_mlp를 통해 시간 임베딩으로 변환

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)  # 생성된 시간 임베딩을 self.to_time_cond와 self.to_time_tokens를 통해 조건부 주의와 FiLM 조절 정보로 변환
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        # 널 조건부 숨겨진 상태(null_cond_hidden)를 현재 시간 임베딩 텐서 t의 데이터 타입으로 변환, 속 연산에서 데이터 타입의 일관성을 보장하기 위함
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        # keep_mask_hidden의 각 요소에 따라 cond_hidden과 null_cond_hidden 중 하나를 선택
        t += cond_hidden  # 조건부 드롭아웃을 고려하여 시간 임베딩과 조건부 히든 상태(cond_hidden)를 통합
        # 모델이 입력 시퀀스를 처리할 때 시간적 컨텍스트와 조건부 정보를 동시에 고려할 수 있도록 함
        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)  # 텐서의 뒤에서 두 번째 차원을 기준으로 결합하라는 의미
        cond_tokens = self.norm_cond(c)  # 조건부 토큰과 시간 토큰을 결합하고, self.norm_cond를 통해 정규화
        # 각 입력 특징에 대한 정보가 한 레이어 내에서 통합되어, 모델이 이 두 유형의 정보를 동시에 고려할 수 있도록 함
        # torch.cat을 통해 생성된 결합된 토큰 c에 레이어 정규화를 적용
        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)  # 입력 x, 조건부 토큰 cond_tokens, 시간임베딩 t 사용하여 DecoderLayerStack 통과

        output = self.final_layer(output)
        return output