import unittest


class TestCRICMeraDxTransfer(unittest.TestCase):
    def test_validates_xudata_source_checkpoint(self):
        from experiments.train_cric_meradx_transfer import validate_source_checkpoint

        payload = {
            "schema_version": "xudata-to-cric-zeroshot-checkpoint-v1",
            "model_name": "mera_dx",
            "backbone_name": "swin_tiny_patch4_window7_224",
            "model_state": {"base_head.weight": None},
        }
        self.assertTrue(validate_source_checkpoint(payload))

    def test_rejects_target_or_wrong_model_checkpoint(self):
        from experiments.train_cric_meradx_transfer import validate_source_checkpoint

        with self.assertRaises(ValueError):
            validate_source_checkpoint({"schema_version": "cric-fiveclass-training-v1"})

    def test_inverse_sqrt_sampler_weights_are_class_balancing_weights(self):
        from experiments.train_cric_meradx_transfer import inverse_sqrt_sample_weights

        weights = inverse_sqrt_sample_weights([0, 0, 0, 1, 1, 2])
        self.assertEqual(len(weights), 6)
        self.assertAlmostEqual(weights[0], weights[1])
        self.assertGreater(weights[3], weights[0])
        self.assertGreater(weights[5], weights[3])

    def test_target_metadata_forbids_zero_shot_claims(self):
        from experiments.train_cric_meradx_transfer import transfer_metadata_flags

        flags = transfer_metadata_flags()
        self.assertTrue(flags["target_training_performed"])
        self.assertTrue(flags["target_parameter_updates"])
        self.assertFalse(flags["target_test_accessed_during_training"])

    def test_loss_call_supports_legacy_trainer_signature(self):
        from experiments.train_cric_meradx_transfer import call_meradx_loss

        calls = []

        def legacy_loss(model_name, output, labels):
            calls.append((model_name, output, labels))
            return "loss", {}

        result = call_meradx_loss(legacy_loss, "output", "labels", "full")
        self.assertEqual(result, ("loss", {}))
        self.assertEqual(calls, [("mera_dx", "output", "labels")])

    def test_direct_ce_requires_diagnosis_log_probabilities(self):
        from experiments.train_cric_meradx_transfer import call_meradx_loss

        with self.assertRaises(ValueError):
            call_meradx_loss(lambda *args, **kwargs: None, {}, "labels", "direct_ce")


if __name__ == "__main__":
    unittest.main()
