import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pandas as pd
from PIL import Image

import test_H2_Human_Image_Aware_Plan_at6_on_Charxiv as runner
from rig_eval import load_predictions


class HumanRunnerTests(unittest.TestCase):
    def test_frozen_real_plan_and_tamper(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'frozen'
            shutil.copytree(runner.human.DEFAULT_FROZEN_DIR, dest)
            plans, _ = runner.load_frozen_plans(dest / 'plans.json')
            self.assertEqual(len(plans), 5)
            (dest / 'plans.json').write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_frozen_plans(dest / 'plans.json')

    def test_source_and_frozen_output_protected(self):
        source = runner.human.DEFAULT_FROZEN_DIR / 'plans.json'
        for output in (source.parent, source.parent / 'child', source.parent.parent,
                       runner.human.DEFAULT_ROOT / 'model__repeated-single-at6'):
            with self.assertRaises(ValueError):
                runner.assert_output_safe(output, source)

    def test_q1_then_five_resume_preserves_model_failures_and_retries_transport(self):
        plans, frozen = runner.load_frozen_plans(runner.human.DEFAULT_FROZEN_DIR / 'plans.json')
        frame = pd.DataFrame([{'index': i, 'raw_question': p['question'], 'inst_category': 1}
                              for i, p in plans.items()])
        count = 0
        def fake_call(client, **kwargs):
            nonlocal count
            count += 1
            self.assertEqual(kwargs['max_tokens'], 12288)
            self.assertEqual(len(kwargs['messages']), 2)
            self.assertEqual(kwargs['extra_body'], runner.static.step_extra_body())
            if count == 3:
                return None, 1, 'synthetic transport failure', 1
            content = json.dumps({'elements': [{'description': f'clue {count}',
                                               'bbox': [10, 20, 30, 40]}]})
            response = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason='length' if count == 2 else 'stop')],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30))
            return response, 1, None, 1
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner.pd, 'read_csv', return_value=frame), \
                patch.object(runner.static, 'source_image_sha256',
                    side_effect=lambda row, path: frozen['input_image_sha256'][str(row['index'])]), \
                patch.object(runner.static, 'load_image', return_value=Image.new('RGB', (50, 50))), \
                patch.object(runner.static, 'call_model', side_effect=fake_call), \
                patch.object(runner.static, 'draw_bboxes_on_image'), \
                patch.object(runner, 'OpenAI', return_value=MagicMock()):
            output = Path(tmp) / 'human-run'
            tsv = Path(tmp) / 'input.tsv'
            tsv.write_text('fixture', encoding='utf-8')
            base = ['--output-dir', str(output), '--tsv-path', str(tsv)]
            self.assertEqual(runner.main(base + ['--indices', '1']), 2)
            self.assertEqual(count, 6)
            first = runner.read_json(output / 'step_outputs.json')
            self.assertEqual(first[1]['status'], 'truncated')
            self.assertEqual(first[1]['elements'], [])
            self.assertFalse(runner.read_json(output / 'run_status.json')['complete'])
            self.assertEqual(runner.main(base + ['--smoke', '--resume']), 0)
            self.assertEqual(count, 31)  # Only the transport failure is resampled.
            saved = runner.read_json(output / 'step_outputs.json')
            self.assertEqual(saved[0], first[0])
            self.assertEqual(saved[1], first[1])
            self.assertEqual(len(saved[2]['prior_request_failures']), 1)
            self.assertTrue(runner.read_json(output / 'run_status.json')['complete'])
            self.assertEqual(len(load_predictions(output / 'rig_results.json')), 5)
            for k in range(1, 7):
                results = runner.read_json(output / 'prefix_results' / f'k{k}' / 'rig_results.json')
                self.assertEqual(len(results), 5)
                self.assertTrue(all(r['h2_mode'] == runner.MODE for r in results))
            self.assertEqual(runner.main(base + ['--indices', '1', '--resume']), 0)
            self.assertEqual(count, 31)
            self.assertEqual(len(runner.read_json(output / 'rig_results.json')), 5)
            with self.assertRaisesRegex(ValueError, 'api_base'):
                runner.main(base + ['--indices', '1', '--resume', '--api-base', 'http://changed/v1'])


if __name__ == '__main__':
    unittest.main()
