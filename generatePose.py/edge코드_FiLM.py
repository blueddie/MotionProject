class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""
    # Feature-wise linear modulation (FiLM) 생성기 클래스를 정의합니다. FiLM은 특정 차원의 특성에 선형 변환을 적용합니다.

    # FiLM은 신경망에서 주로 조건부 학습을 위해 사용되는 기법 중 하나.
    # FiLM은 입력 특성들에 대해 각각 선형 변환을 독립적으로 적용하여 신경망의 특정 층의 출력을 조절
    # 이를 통해 신경망이 주어진 조건에 따라 다르게 반응하도록 만들 수 있음.




    def __init__(self, embed_channels):
        super().__init__()  # nn.Module 클래스의 생성자를 호출하여 모듈 초기화를 진행합니다.
        self.embed_channels = embed_channels  # 임베딩 차원의 크기를 인스턴스 변수로 저장합니다.
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )  # Mish 활성화 함수와 선형 변환을 연속적으로 적용하는 블록을 생성합니다. 출력 차원은 입력의 두 배입니다.

    def forward(self, position):
        pos_encoding = self.block(position)  # 입력 position에 대해 정의된 블록(활성화 함수와 선형 변환)을 적용합니다.
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")  # 결과 텐서의 차원을 재배열합니다. 채널 차원을 확장하여 3D 텐서로 만듭니다.
        scale_shift = pos_encoding.chunk(2, dim=-1)  # 재배열된 텐서를 채널 차원을 따라 2개로 나누어 스케일과 시프트를 생성합니다.
        return scale_shift  # 스케일과 시프트의 튜플을 반환합니다.

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift          # 입력된 scale_shift 튜플에서 스케일(scale)과 시프트(shift) 값을 추출합니다.
    return (scale + 1) * x + shift      # 입력 텐서 x의 각 요소에 스케일을 적용하고 시프트를 더합니다.
    # 각 요소에 대해 계산된 스케일 값에 1을 더한 후 입력 x에 곱하고, 그 결과에 시프트 값을 더합니다.
    # 스케일 값에 1을 더하는 이유는 기본적인 입력 값 x에 어느 정도 영향을 유지하면서 변형을 가하는 것입니다.
    # 예를 들어, 스케일이 0일 경우에도 원본 입력 x의 영향을 받습니다
    # 이 함수는 주로 신경망 내에서 다양한 조건에 따라 입력 텐서의 각 특성을 동적으로 조절할 때 사용됩니다.
    # 이러한 기능은 이미지 스타일 전환, 조건부 이미지 생성, 음성 인식 등의 응용에서 매우 유용하게 사용됩니다.


