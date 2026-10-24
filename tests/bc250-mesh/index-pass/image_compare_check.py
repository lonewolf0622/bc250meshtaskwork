#!/usr/bin/env python3
"""Exercise exact probes and guarded interpolated readback checks without a GPU."""
from benchmark import compare_image, image_acceptance


def main():
    reference = bytes((40, 80, 120, 255)) * (256 * 256 - 1) + bytes(4)
    assert compare_image(reference, reference)['exact']
    for channel, delta, accepted in ((0, 1, True), (1, -1, True), (2, 2, False), (3, -1, False)):
        image = bytearray(reference)
        image[channel] += delta
        mesh = compare_image(image, reference, allow_rounding=True)
        assert mesh['accepted'] == accepted, (channel, delta, mesh)
        assert not compare_image(image, reference)['accepted'], 'IDX must remain exact'
        assert mesh['different_pixels'] == 1 and mesh['max_channel_error'] == abs(delta)
    image = bytearray(reference)
    image[-4] = 1
    assert not compare_image(image, reference, allow_rounding=True)['accepted'], 'background changed'
    image[-1] = 1
    assert not compare_image(image, reference, allow_rounding=True)['accepted'], 'coverage changed'
    image = bytearray(reference)
    image[0] += 1
    case = dict(name='heavy_pp128')
    assert not image_acceptance(case, 'mesh-before', image, reference)['accepted'], 'unverified Mesh accepted'
    for mode in ('existing-before', 'stored'):
        assert not image_acceptance(case, mode, image, reference)['accepted'], 'unverified IDX accepted'
        verified = dict(case, verified_flat_oracle=True)
        assert image_acceptance(verified, mode, image, reference)['accepted']
        assert not image_acceptance(dict(verified, flat_oracle=True), mode, image, reference)['accepted']
    assert not image_acceptance(dict(case, flat_oracle=True), 'mesh-before', image, reference)['accepted']
    print('PASS exact probes, guarded IDX rounding, RGB bounds, alpha, background and coverage checks')


if __name__ == '__main__':
    main()
