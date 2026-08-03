def test_preprocess_image_shape():
    import io
    from PIL import Image
    from app.utils.image_utils import preprocess_image

    buf = io.BytesIO()
    Image.new("RGB", (300, 300)).save(buf, format="PNG")
    result = preprocess_image(buf.getvalue())
    assert result.shape == (224, 224, 3)
