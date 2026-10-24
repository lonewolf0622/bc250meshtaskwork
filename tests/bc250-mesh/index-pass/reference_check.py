#!/usr/bin/env python3
"""Execute the compiled VS references on CPU; check indirect grid equivalence.

This deliberately small SPIR-V evaluator supports only the instructions these
reference shaders use, and fails on anything else. It checks geometry and all
varyings without creating a Vulkan device or accessing GPU nodes.
"""
import operator
from pathlib import Path
import struct
import subprocess


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def elementwise(fn, a, b):
    if isinstance(a, tuple):
        return tuple(fn(x, y) for x, y in zip(a, b if isinstance(b, tuple) else (b,) * len(a)))
    return fn(a, b)


class Shader:
    def __init__(self, path):
        assembly = subprocess.check_output(['spirv-dis', str(path)], text=True)
        self.code, self.constants, self.variables = [], {}, {}
        self.builtins, self.outputs = {}, {}
        for line in assembly.splitlines():
            words = line.split()
            if not words or words[0].startswith(';'):
                continue
            dst = None
            if len(words) > 2 and words[1] == '=':
                dst, words = words[0], words[2:]
            op, args = words[0], words[1:]
            if op == 'OpFunction':
                self.start = len(self.code)
            if op == 'OpConstant':
                self.constants[dst] = f32(float(args[1])) if args[0] == '%float' else int(args[1])
            elif op in ('OpConstantTrue', 'OpConstantFalse'):
                self.constants[dst] = op == 'OpConstantTrue'
            elif op == 'OpConstantComposite':
                self.constants[dst] = tuple(self.constants[a] for a in args[1:])
            elif op == 'OpVariable':
                self.variables[dst] = (dst, ())
                if args[1] == 'Output':
                    self.outputs[dst] = dst
            elif op == 'OpDecorate' and args[1] == 'BuiltIn':
                self.builtins[args[2]] = args[0]
            self.code.append((dst, op, args))
        self.labels = {dst: n for n, (dst, op, _) in enumerate(self.code) if op == 'OpLabel'}

    def run(self, vertex, draw, indirect, steps=0, seed=7, oracle_corner=0):
        values = self.constants | self.variables
        memory = {(self.builtins['VertexIndex'], ()): vertex}
        memory.update({('%dispatch', (0,)): seed, ('%dispatch', (1,)): indirect,
                       ('%dispatch', (2,)): steps, ('%dispatch', (3,)): oracle_corner})
        if 'DrawIndex' in self.builtins:
            memory[(self.builtins['DrawIndex'], ())] = draw
            memory[('%dispatch', (1,))] = indirect
        binary = {
            'OpIAdd': operator.add, 'OpISub': operator.sub, 'OpIMul': operator.mul,
            'OpUDiv': operator.floordiv, 'OpUMod': operator.mod,
            'OpBitwiseAnd': operator.and_, 'OpShiftRightLogical': operator.rshift,
            'OpBitwiseXor': operator.xor, 'OpShiftLeftLogical': operator.lshift,
            'OpULessThan': operator.lt,
            'OpIEqual': operator.eq, 'OpINotEqual': operator.ne,
            'OpFAdd': lambda x, y: f32(x + y), 'OpFSub': lambda x, y: f32(x - y),
            'OpFDiv': lambda x, y: f32(x / y),
            'OpVectorTimesScalar': lambda x, y: f32(x * y),
        }
        pc, block, previous = self.start, None, None
        while True:
            dst, op, args = self.code[pc]
            pc += 1
            if op in ('OpFunction', 'OpVariable', 'OpSelectionMerge', 'OpLoopMerge'):
                continue
            if op == 'OpLabel':
                previous, block = block, dst
            elif op == 'OpLoad':
                values[dst] = memory[values[args[1]]]
            elif op == 'OpStore':
                memory[values[args[0]]] = values[args[1]]
            elif op == 'OpAccessChain':
                base, indices = values[args[1]]
                values[dst] = (base, indices + tuple(values[a] for a in args[2:]))
            elif op in binary:
                values[dst] = elementwise(binary[op], values[args[1]], values[args[2]])
                if args[0] == '%uint':
                    values[dst] &= 0xffffffff
            elif op == 'OpBitcast':
                values[dst] = values[args[1]]
            elif op == 'OpConvertUToF':
                values[dst] = f32(values[args[1]])
            elif op == 'OpCompositeConstruct':
                values[dst] = tuple(values[a] for a in args[1:])
            elif op == 'OpCompositeExtract':
                values[dst] = values[args[1]][int(args[2])]
            elif op == 'OpSelect':
                values[dst] = values[args[2] if values[args[1]] else args[3]]
            elif op == 'OpPhi':
                choices = dict(zip(args[2::2], args[1::2]))
                values[dst] = values[choices[previous]]
            elif op == 'OpBranch':
                pc = self.labels[args[0]]
            elif op == 'OpBranchConditional':
                pc = self.labels[args[1] if values[args[0]] else args[2]]
            elif op == 'OpReturn':
                return {key: val for key, val in memory.items() if key[0] in self.outputs}
            else:
                raise RuntimeError('unsupported reference instruction: ' + op)


def main():
    here = Path(__file__).resolve().parent
    total = 0
    for name in ('j64', 'ballot_vertex64', 'shared_vertex64', 'pp64'):
        direct = Shader(here / (name + '.vert.spv'))
        indirect = Shader(here / (name + '_ind.vert.spv'))
        # Check every corner in all 64 groups. The empty middle record leaves
        # the second nonempty record with DrawID 2, not DrawID 1.
        expected = [direct.run(k, 0, 0) for k in range(64 * 192)]
        for draw, enabled in ((0, 1), (2, 1), (2, 0)):
            for k in range(64 * 192):
                x = k // 192 % 16 if draw == 2 and enabled else k // 192
                mapped = x * 192 + k % 192
                assert indirect.run(k, draw, enabled) == expected[mapped], (name, draw, enabled, k)
                total += 1
        print('PASS ' + name + ': positions and all varyings match logical grid', flush=True)
    print(f'PASS {total} compiled-reference vertex comparisons', flush=True)


if __name__ == '__main__':
    main()
