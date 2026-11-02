import unittest
import numpy as np
from inference_animation_model import example,plan,state

class TeachingInferenceTests(unittest.TestCase):
    def test_actual_routes_of_two_objects(self):
        np.testing.assert_array_equal(example()['leaves'],[[0,1,0,1],[1,0,1,0]])

    def test_reference_sum_and_expected_answer(self):
        m=example();raw=m['bias']+np.stack([m['full'][t,m['leaves'][:,t]] for t in range(4)]).sum(axis=0)
        np.testing.assert_allclose(raw,[[.6,.05,1.],[-.65,1.,-1.3]],atol=1e-14)
        np.testing.assert_array_equal(raw>=0,[[1,1,1],[0,1,0]])

    def test_factorized_example_is_exact_by_construction(self):
        for c in (3,144):
            m=example(c);np.testing.assert_allclose(m['coordinates']@m['basis']+m['residual'],m['full'],atol=1e-14)
            z=np.zeros((2,2));e=np.zeros((2,c))
            for t in range(4):z+=m['coordinates'][t,m['leaves'][:,t]];e+=m['residual'][t,m['leaves'][:,t]]
            np.testing.assert_allclose(z@m['basis']+e+m['bias'],state(m,plan('packed',True)[-1])['raw'],atol=1e-14)

    def test_every_variant_finishes_same_ensemble(self):
        for wide in (False,True):
            m=example(144 if wide else 3)
            expected=state(m,plan('native')[-1])
            for engine in ('native','legacy','packed'):
                for compact in (False,True):
                    stages=plan(engine,compact,wide)
                    result=state(m,stages[-1])
                    np.testing.assert_allclose(result['probability'],expected['probability'],atol=1e-14)
                    self.assertEqual(stages[-1].phase,'cpu_out')
                    for stage in stages:
                        self.assertEqual(len(stage.completed),len(set(stage.completed)))
                        self.assertTrue(set(stage.completed)<=set(range(4)))

    def test_cpu_answer_does_not_appear_before_download(self):
        for engine in ('native','legacy','packed'):
            stages=plan(engine,True,True)
            self.assertTrue(all(not s.cpu_ready for s in stages[:-2]))
            self.assertTrue(all(s.cpu_ready for s in stages[-2:]))

    def test_packed_stages_stay_in_one_launch(self):
        stages=plan('packed',True,True)
        self.assertEqual(len({s.kernel for s in stages if s.phase=='gpu'}),1)
        native=[s for s in plan('native') if s.id.startswith(('walk','gather'))]
        self.assertEqual(len({s.kernel for s in native}),8)

    def test_legacy_gather_and_walk_share_block_launch(self):
        stages=plan('legacy',True,False)
        self.assertEqual(stages[3].kernel,stages[4].kernel)
        wide=plan('legacy',True,True)
        self.assertTrue(any(s.id.startswith('decode') for s in wide))
        self.assertFalse(any(s.id.startswith('decode') for s in stages))

if __name__=='__main__':unittest.main()
