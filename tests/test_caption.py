"""The prompt as FIBO 2.0 reads it, and the pictures Qwen3-VL sees (caption.py)."""

import json

import torch

from forge_fibo2.native import caption, phases, text_images

MARKER = caption.VISION_MARKER


def chat_body(chat):
    assert chat.startswith("<|im_start|>user\n") and chat.endswith("<|im_end|>\n<|im_start|>assistant\n")
    return chat[len("<|im_start|>user\n"): -len("<|im_end|>\n<|im_start|>assistant\n")]


def test_empty_prompt_is_the_null_caption():
    assert caption.build("") == (caption.NULL_CAPTION, [])
    assert caption.build("   \n") == (caption.NULL_CAPTION, [])
    assert json.loads(chat_body(caption.NULL_CAPTION))["objects"] == []


def test_plain_text_goes_in_as_written():
    chat, texts = caption.build("a red fox in the snow, {not json}")
    assert chat_body(chat) == "a red fox in the snow, {not json}"
    assert texts == []


def test_json_caption_is_cleaned_and_compacted():
    prompt = json.dumps({
        "short_description": "A fox",
        "objects": [{"description": "fox", "pose": "", "aesthetic_score": 9}, {}],
        "lighting": {"conditions": "dusk", "direction": None},
        "aesthetics": {"composition": "centered", "preference_score": 0.8},
        "aesthetic_score": 7,
        "context": "",
        "artistic_style": "naturalistic",
    }, indent=2)
    body = chat_body(caption.build(prompt)[0])
    assert body == ('{"short_description":"A fox","objects":[{"description":"fox","aesthetic_score":9}],'
                    '"lighting":{"conditions":"dusk"},"aesthetics":{"composition":"centered"},'
                    '"artistic_style":"naturalistic"}')


def test_short_description_and_empty_text_render_keep_both():
    body = chat_body(caption.build('{"short_description": "A wall", "text_render": []}')[0])
    assert body == '{"short_description":"A wall","text_render":[]}'


def test_text_render_entries_get_their_pictures():
    prompt = json.dumps({"short_description": "A sign", "text_render": [
        {"text": "OPEN 24H", "location": "center"}, {"text": ""}, {"location": "top"}, {"text": "Café"}]})
    chat, texts = caption.build(prompt)
    body = json.loads(chat_body(chat))
    assert texts == ["OPEN 24H", "Café"]
    # the cleaning drops the entry whose only field is empty; the one without text gets no picture
    assert [e.get("image_tokens") for e in body["text_render"]] == [MARKER, None, MARKER]
    # the caption a generation starts from: the same entries, their markers empty, no pictures
    chat, texts = caption.build(prompt, text_images=False)
    assert texts == []
    assert [e.get("image_tokens") for e in json.loads(chat_body(chat))["text_render"]] == ["", None, ""]


def test_edit_wraps_the_instruction():
    body = json.loads(chat_body(caption.build("make it night", edit_images=1)[0]))
    assert body == {"image": MARKER, "edit_instruction": "make it night"}
    body = json.loads(chat_body(caption.build("swap <image_1> and <image_2>", edit_images=2)[0]))
    assert body == {"image_1": MARKER, "image_2": MARKER, "edit_instruction": "swap <image_1> and <image_2>"}


def test_edit_with_a_caption_of_the_result():
    prompt = json.dumps({"edit_instruction": "add a sign", "short_description": "A shop",
                         "text_render": [{"text": "SALE"}]})
    chat, texts = caption.build(prompt, edit_images=1)
    assert texts == ["SALE"]
    assert json.loads(chat_body(chat)) == {
        "image": MARKER, "structured_caption": {"short_description": "A shop",
                                                "text_render": [{"text": "SALE", "image_tokens": MARKER}]},
        "edit_instruction": "add a sign"}
    # the image markers come first: the images to edit, then the text pictures, in the token order
    assert chat.index(MARKER) < chat.index('"text_render"')


def test_trained_sizes():
    assert caption.trained_size(1.0) == (1024, 1024)
    assert caption.trained_size(16 / 9) == (1424, 800)
    assert caption.trained_size(9 / 16) == (800, 1424)


def test_text_picture():
    picture = caption.text_picture("Hello   world")
    assert picture.shape[0] == 1 and picture.shape[-1] == 3
    assert picture.max() == 1.0 and picture.min() < 0.5  # black text on white
    assert caption.to_qwen_grid(picture).shape[1] % 32 == 0


def test_accents_come_from_the_fallback_fonts():
    font = text_images.default_font()
    aileron, dejavu = text_images.fallback_fonts()
    # Pillow's built-in font has no accented letters; training drew them with Aileron
    assert not text_images.has_glyph(font, "é")
    assert text_images.has_glyph(aileron, "é") and text_images.has_glyph(aileron, "Ã")
    runs = text_images.font_runs("PÃO Café", [font, aileron, dejavu])
    assert runs == [("P", font), ("Ã", aileron), ("O Caf", font), ("é", aileron)]
    # a character Aileron lacks goes to DejaVu; one no font has stays with the built-in font (its notdef box)
    assert text_images.font_runs("Ж", [font, aileron, dejavu]) == [("Ж", dejavu)]
    assert text_images.font_runs("中", [font, aileron, dejavu]) == [("中", font)]


def test_accented_picture_has_no_notdef_box():
    # the built-in font alone draws "é" as its notdef box; with the fallback it is a real glyph, so the picture of
    # "é" differs from the picture of an unknown character
    accented = text_images.draw_text_image("Café")
    unknown = text_images.draw_one_font(["Caf￾"], text_images.default_font())
    assert accented.size != unknown.size or list(accented.getdata()) != list(unknown.getdata())


def test_qwen_grid_and_thumbnail():
    image = torch.rand(1, 1000, 1500, 3)
    thumb = caption.thumbnail(image)
    assert thumb.shape == (1, 512, 768, 3)
    grid = caption.to_qwen_grid(thumb)
    assert grid.shape == (1, 512, 768, 3)
    # 8-bit values, as the processor's
    assert torch.equal(grid, (grid * 255).round() / 255)
    assert caption.to_qwen_grid(torch.rand(1, 10, 10, 3)).shape == (1, 64, 64, 3)


def test_trained_sizes_on_another_step():
    width, height = caption.trained_size(4 / 3, 64)
    assert width % 64 == 0 and height % 64 == 0
    assert 0.9 <= width * height / 2 ** 20 <= 1.1


def test_size_advice():
    # about 1 MP in the picture's proportions: fine
    assert caption.size_advice((1600, 1200), (1184, 880)) is None
    assert caption.size_advice((1024, 1024), (1024, 1024)) is None
    # a 4:3 picture into a square result, or a small one: the trained size for 4:3
    assert caption.size_advice((1600, 1200), (1024, 1024)) == caption.trained_size(4 / 3)
    assert caption.size_advice((1024, 1024), (512, 512)) == (1024, 1024)
    assert caption.size_advice((1600, 1200), (512, 512), 64) == caption.trained_size(4 / 3, 64)


def test_edit_images_sizes():
    one = caption.edit_images([torch.rand(1, 600, 1000, 4)])
    assert one[0].shape == (1, 592, 992, 3)
    # a single image is cropped to the result's size
    assert caption.edit_images([torch.rand(1, 600, 1000, 3)], 1024, 1024)[0].shape == (1, 1024, 1024, 3)
    same = torch.rand(1, 512, 768, 3)
    assert torch.equal(caption.edit_images([same])[0], same)
    # several images: each to its trained size (5:3 -> 1360 x 816, about 1.1 MP)
    two = caption.edit_images([torch.rand(1, 600, 1000, 3), torch.rand(1, 900, 900, 3)])
    assert [tuple(i.shape) for i in two] == [(1, 816, 1360, 3), (1, 1024, 1024, 3)]


def test_same_proportions():
    assert caption.same_proportions((800, 1424), (800, 1424))
    assert caption.same_proportions((1600, 1200), (1184, 880))
    assert not caption.same_proportions((800, 1424), (1424, 800))
    assert not caption.same_proportions((800, 1424), (1024, 1024))


def test_fill_box_is_where_resize_and_fill_puts_the_picture():
    # Forge's resize_image mode 2: a portrait into 16:9 sits in the middle at full height
    assert caption.fill_box((800, 1424), (1424, 800)) == (488, 0, 937, 800)
    # a landscape into a square at full width
    assert caption.fill_box((1424, 800), (1024, 1024)) == (0, 225, 1024, 800)
    assert caption.fill_box((1024, 1024), (1024, 1024)) == (0, 0, 1024, 1024)


def test_pad_fits_the_picture_inside_on_grey():
    image = torch.ones(1, 1424, 800, 3)
    out = caption.pad(image, 1424, 800)
    assert out.shape == (1, 800, 1424, 3)
    left, _, right, _ = caption.fill_box((800, 1424), (1424, 800))
    assert torch.allclose(out[0, :, :left], torch.tensor(128 / 255))
    assert torch.allclose(out[0, :, right:], torch.tensor(128 / 255))
    assert (out[0, :, left + 2:right - 2] == 1.0).all()
    same = torch.rand(1, 64, 96, 3)
    assert caption.pad(same, 96, 64) is same
    # edit_images: one image fitted instead of cropped, several unchanged by fill
    assert caption.edit_images([image], 1424, 800, fill=True)[0].shape == (1, 800, 1424, 3)
    two = caption.edit_images([torch.rand(1, 600, 1000, 3), torch.rand(1, 900, 900, 3)], 1424, 800, fill=True)
    assert [tuple(i.shape) for i in two] == [(1, 816, 1360, 3), (1, 1024, 1024, 3)]


def test_fit_crops_the_center():
    image = torch.zeros(1, 100, 300, 3)
    image[:, :, 100:200] = 1.0
    out = caption.fit(image, 64, 64)
    assert out.shape == (1, 64, 64, 3)
    assert out.mean() > 0.95


def test_phases_pack_and_split():
    early, late = torch.full((2, 3), 1.0), torch.full((4, 3), 2.0)
    packed = phases.pack(late, early)
    assert packed.shape == (6, 4)
    assert phases.split(packed[None], 0.95)[0].tolist() == early.tolist()
    assert phases.split(packed[None], 0.9)[0].tolist() == late.tolist()
    single = phases.pack(late)
    for sigma in (1.0, 0.5):
        assert phases.split(single[None], sigma)[0].tolist() == late.tolist()
