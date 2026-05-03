from opacus import PrivacyEngine

def make_private(model, optimizer, data_loader):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )
    return model, optimizer, data_loader, privacy_engine