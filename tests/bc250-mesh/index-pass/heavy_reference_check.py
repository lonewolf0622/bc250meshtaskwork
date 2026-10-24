#!/usr/bin/env python3
"""Check actual compiled heavy VS outputs against a separate integer oracle."""
from pathlib import Path
from reference_check import Shader, f32


def expected(k, vertices, draw, indirect, steps, ballot, seed, oracle_corner=0):
    primitive, corner = k // 3 % vertices, k % 3
    corner = (corner + oracle_corner) % 3
    group = k // (vertices * 3)
    if indirect and draw == 2:
        group %= 16
    modulus = vertices - 1
    first = primitive % 2 if primitive % 4 == 3 else (primitive * 5 + group) % modulus
    offset = 1 + primitive * 7 % (modulus - 1)
    third = 1 + (offset + primitive * 3 % (modulus - 2)) % (modulus - 1)
    vertex = (first, (first + offset) % modulus, (first + third) % modulus)[corner]
    h = ((vertex + 1) * 747796405 ^ (group + seed) * 2891336453 ^
         (64 * 277803737 if ballot else 0)) & 0xffffffff
    for n in range(steps):
        h = ((h ^ (h >> 16)) * 2246822519) & 0xffffffff
        h = ((h ^ (h >> 13)) * 3266489917) & 0xffffffff
        h = ((h ^ (h >> 16)) + n * 668265263 + 374761393) & 0xffffffff
    color = tuple(f32(((h >> bit) & 255) / 255) for bit in (0, 8, 16)) + (1.0,)
    cols = 8 if vertices == 64 else 16
    x = f32(f32(f32((group % 8) / 4) - 1) + f32(f32((vertex % cols) / cols) * 0.25))
    y = f32(f32(f32((group // 8) / 4) - 1) + f32(f32((vertex // cols) / 8) * 0.25))
    result = {('%_', (0,)): (f32(x + f32(0.01)), f32(y + f32(0.01)), 0.5, 1.0),
              ('%vtx', ()): color}
    if vertices == 128:
        result[('%color', ())] = (f32(((primitive * 37 + group * 11) & 255) / 255),
                                 f32(((primitive * 13 + group * 7) & 255) / 255),
                                 f32((primitive & 7) / 7), 1.0)
    return result


def main():
    here = Path(__file__).resolve().parent
    total = 0
    flat_total = 0
    for name, vertices, ballot in (('heavy64', 64, False), ('heavy_pp128', 128, False),
                                    ('heavy_ballot64', 64, True)):
        shader = Shader(here / (name + '.vert.spv'))
        flat = Shader(here / (name + '_flat.vert.spv'))
        # All triangle corners in group zero, then grid-remapping boundaries.
        corners = list(range(vertices * 3))
        corners += [group * vertices * 3 + corner for group in (15, 16, 63)
                    for corner in (0, 1, 2, vertices * 3 - 1)]
        for steps in (0, 16, 64, 256, 1024):
            for k in corners:
                for draw, indirect in ((0, 0), (2, 1)):
                    actual = shader.run(k, draw, indirect, steps=steps)
                    assert actual == expected(k, vertices, draw, indirect, steps, ballot, 7), (name, steps, k, draw)
                    total += 1
            # Seed changes must reach the observable outputs too.
            for seed in (1, 19):
                actual = shader.run(4, 0, 0, steps=steps, seed=seed)
                assert actual == expected(4, vertices, 0, 0, steps, ballot, seed), (name, steps, seed)
                total += 1
            print(f'PASS {name} steps={steps}: compiled positions, vertex colors and primitive attributes', flush=True)
            # Full corner coverage at short loops, boundary samples at long loops.
            samples = corners if steps <= 64 else corners[:12] + corners[-15:]
            for rotation in range(3):
                for k in samples:
                    for draw, indirect in ((0, 0), (2, 1)):
                        actual = flat.run(k, draw, indirect, steps=steps, oracle_corner=rotation)
                        assert actual == expected(k, vertices, draw, indirect, steps, ballot, 7, rotation), (name, steps, k, draw, rotation)
                        flat_total += 1
            print(f'PASS {name} steps={steps}: three compiled flat corner rotations', flush=True)
    print(f'PASS {total} heavy compiled-reference comparisons', flush=True)
    print(f'PASS {flat_total} flat compiled-reference comparisons', flush=True)


if __name__ == '__main__':
    main()
