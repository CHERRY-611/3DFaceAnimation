import torch
import torch.nn.functional as F

def compute_total_loss(bs_pred11, bs_pred12, emo_logits, data, lambda_weights=None):
    """
    - bs_pred11: 예측된 blendshape (cross-recon1)
    - bs_pred12: 예측된 blendshape (cross-recon2)
    - emo_logits: 감정 분류 로짓 (multi-label)
    - data: 데이터 배치 (target 포함)
    - lambda_weights: loss weight dict (선택)
    """
    # 기본 가중치
    if lambda_weights is None:
        lambda_weights = {
            "cross": 1.0,
            "self": 1.0,
            "vel": 0.5,
            "cls": 0.1
        }

    # target
    target11 = data["target11"].to(bs_pred11.device)  # [B, T, 52]
    target12 = data["target12"].to(bs_pred12.device)

    # 1. Cross Reconstruction Loss
    loss_cross = F.mse_loss(bs_pred11, target11) + F.mse_loss(bs_pred12, target12)

    # 2. Self Reconstruction Loss
    loss_self = F.mse_loss(bs_pred11, target12)  # pred11 from input12 → target12
    # 이 구성은 상황 따라 달라질 수 있음

    # 3. Velocity Loss (시간 축 smoothness 유도)
    def velocity_loss(pred, target):
        return F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])

    loss_vel = velocity_loss(bs_pred11, target11) + velocity_loss(bs_pred12, target12)

    # 4. Classification Loss (감정 분류 → BCE)
    # 감정 레이블은 현재 없음 → pseudo-label 대체 or loss 0 처리
    if "emotion_label" in data:
        emotion_label = data["emotion_label"].float().to(bs_pred11.device)
        loss_cls = F.binary_cross_entropy_with_logits(emo_logits, emotion_label)
    else:
        loss_cls = torch.tensor(0.0, device=bs_pred11.device)

    # 전체 합산
    total = (
        lambda_weights["cross"] * loss_cross +
        lambda_weights["self"] * loss_self +
        lambda_weights["vel"] * loss_vel +
        lambda_weights["cls"] * loss_cls
    )

    return total