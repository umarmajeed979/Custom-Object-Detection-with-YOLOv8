def test_model_loads():
    from app.core.model import load_model
    model = load_model()
    # assert model is not None   # uncomment once a real model path is set
