"""Masked edits: the region the inpaint mask marks greyed out (masking.py)."""

import numpy as np
import torch
from PIL import Image

from forge_fibo2.native import masking


def test_grey_out_through_the_binary_mask():
    image = torch.rand(1, 32, 48, 3)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:16, 10:20] = 255
    mask[20, 30] = 127  # under the threshold: kept
    mask[21, 31] = 128  # at it: greyed
    out = masking.grey_out(image, Image.fromarray(mask))
    grey = torch.full((3,), 128 / 255)
    assert torch.equal(out[0, 8:16, 10:20], grey.expand(8, 10, 3))
    assert torch.equal(out[0, 21, 31], grey)
    assert torch.equal(out[0, 20, 30], image[0, 20, 30])
    assert torch.equal(out[0, :8], image[0, :8])


def test_mask_of_another_size_is_brought_to_the_picture():
    image = torch.zeros(1, 32, 32, 3)
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 32, 64))  # left half
    out = masking.grey_out(image, mask)
    assert (out[0, :, :16] > 0).all() and (out[0, :, 16:] == 0).all()


def test_fill_mask_marks_the_sides_and_reaches_into_the_picture():
    mask = np.array(masking.fill_mask((100, 40), (30, 0, 70, 40), overlap=4))
    assert mask.shape == (40, 100)
    assert (mask[:, :34] == 255).all() and (mask[:, 66:] == 255).all()
    assert (mask[:, 34:66] == 0).all()
    # no overlap on the edges the picture reaches: top and bottom stay the picture's
    mask = np.array(masking.fill_mask((40, 100), (0, 30, 40, 70), overlap=4))
    assert (mask[34:66] == 0).all() and (mask[:34] == 255).all() and (mask[66:] == 255).all()
    # a picture that fills the result: nothing to fill
    assert not np.array(masking.fill_mask((64, 64), (0, 0, 64, 64))).any()


def test_state():
    masking.set_mask(None)
    assert masking.current() is None
    mask = Image.new("L", (8, 8))
    masking.set_mask(mask)
    assert masking.current() is mask
    masking.set_mask(None)
