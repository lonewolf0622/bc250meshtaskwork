"""CPU-only checks of the shipped environment and exact argument forwarding."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='directmesh release ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(Path(__file__).with_name('launch.sh'), self.root / 'launch.sh')
        (self.root / 'radeon_icd.json').write_text('{}\n')

    def run_profile(self, profile='standard', extra=None):
        env = dict(PATH='/usr/bin', HOME=self.temp.name)
        env.update(extra or {})
        code = 'import os,json,sys;print(json.dumps(dict(env=dict(os.environ),args=sys.argv[1:])))'
        result = subprocess.run(['bash', str(self.root / 'launch.sh'), '--profile', profile, '--',
                                 sys.executable, '-c', code, 'one argument with spaces', 'literal $() `text`'],
                                env=env, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_shipping_defaults_and_arguments(self):
        result = self.run_profile()
        env = result['env']
        self.assertEqual(env['VK_DRIVER_FILES'], str(self.root / 'radeon_icd.json'))
        self.assertEqual(env['VK_ICD_FILENAMES'], env['VK_DRIVER_FILES'])
        self.assertEqual(env['RADV_DIRECTMESH'], '1')
        self.assertEqual(env['RADV_BC250_MESH_IDXPASS'], '1')
        self.assertEqual(env['RADV_BC250_MESH_IDXPASS_SETUP_BATCH'], '1')
        self.assertEqual(env['RADV_BC250_MESH_IDXPASS_STORE'], '0')
        self.assertEqual(env['RADV_BC250_MESH_IDXPASS_BATCH_TRIM'], '1')
        self.assertEqual(result['args'], ['one argument with spaces', 'literal $() `text`'])

    def test_explicit_settings_including_zero_win(self):
        explicit = dict(VK_DRIVER_FILES='/chosen/icd.json', VK_ICD_FILENAMES='/chosen/legacy.json',
                        RADV_DIRECTMESH='0', RADV_BC250_MESH_IDXPASS='0',
                        RADV_BC250_MESH_IDXPASS_SETUP_BATCH='0', RADV_BC250_MESH_IDXPASS_STORE='0',
                        RADV_BC250_MESH_IDXPASS_BATCH_TRIM='0')
        for profile in ('standard', 'mesh', 'stored'):
            env = self.run_profile(profile, explicit)['env']
            self.assertEqual({k: env[k] for k in explicit}, explicit)

    def test_explicit_empty_value_is_preserved(self):
        env = self.run_profile(extra={'RADV_BC250_MESH_IDXPASS_STORE': ''})['env']
        self.assertEqual(env['RADV_BC250_MESH_IDXPASS_STORE'], '')

    def test_mesh_and_stored_are_explicit_profiles(self):
        self.assertEqual(self.run_profile('mesh')['env']['RADV_BC250_MESH_IDXPASS'], '0')
        env = self.run_profile('stored')['env']
        self.assertEqual((env['RADV_BC250_MESH_IDXPASS'], env['RADV_BC250_MESH_IDXPASS_STORE']), ('1', '1'))

    def test_bad_profile_fails_without_running_command(self):
        result = subprocess.run(['bash', str(self.root / 'launch.sh'), '--profile', 'unknown', '--',
                                 'touch', str(self.root / 'should-not-exist')], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'should-not-exist').exists())


if __name__ == '__main__':
    unittest.main()
