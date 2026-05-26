import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.pipeline_actions_mixin import PipelineActionsMixin


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _StartupDummy(PipelineActionsMixin):
    def __init__(self, aligner="HSMM OTO"):
        self.aligner_var = _Var(aligner)
        self.is_running = False
        self.scheduled = False
        self.started = False

    def _load_mfa_startup_repair_state(self):
        return {}

    def _after_safe(self, func, delay_ms=0):
        self.scheduled = True
        self._scheduled_func = func

    def _run_in_thread(self, func):
        self.started = True


class _PromptDummy(PipelineActionsMixin):
    def __init__(self, approve=False):
        self.mfa_path = ""
        self.approve = approve
        self.prompts = []
        self.repair_started = False
        self.status_ready = None

    def _get_language(self):
        return "japanese"

    def _confirm_mfa_install_action(self, language="korean", reason="install_required"):
        self.prompts.append((language, reason))
        return self.approve

    def _run_mfa_diagnose_repair(self):
        self.repair_started = True

    def _update_mfa_status(self, ready):
        self.status_ready = bool(ready)


class UiMfaInstallGateTests(unittest.TestCase):
    def test_startup_mfa_repair_is_disabled_by_default(self):
        dummy = _StartupDummy(aligner="MFA")
        with mock.patch.dict(os.environ, {}, clear=True):
            dummy._schedule_startup_mfa_auto_repair()
        self.assertFalse(dummy.scheduled)
        self.assertFalse(dummy.started)

    def test_startup_mfa_repair_requires_env_and_mfa_selection(self):
        with mock.patch.dict(os.environ, {"UTOA_ENABLE_STARTUP_MFA_AUTO_REPAIR": "1"}, clear=True):
            hsmm = _StartupDummy(aligner="HSMM OTO")
            hsmm._schedule_startup_mfa_auto_repair()
            self.assertFalse(hsmm.scheduled)

            mfa = _StartupDummy(aligner="MFA")
            mfa._schedule_startup_mfa_auto_repair()
            self.assertTrue(mfa.scheduled)

    @mock.patch("ui.pipeline_actions_mixin.find_mfa_executable", return_value="")
    @mock.patch(
        "ui.pipeline_actions_mixin.diagnose_mfa_runtime",
        return_value={"ready": False, "checks": {"mfa_executable": False}},
    )
    def test_explicit_mfa_selection_prompts_before_repair(self, _diag, _find):
        dummy = _PromptDummy(approve=False)
        dummy._prompt_mfa_install_for_explicit_selection()
        dummy._prompt_mfa_install_for_explicit_selection()

        self.assertEqual(dummy.prompts, [("japanese", "missing_runtime")])
        self.assertFalse(dummy.repair_started)

    @mock.patch("ui.pipeline_actions_mixin.find_mfa_executable", return_value="")
    @mock.patch(
        "ui.pipeline_actions_mixin.diagnose_mfa_runtime",
        return_value={"ready": False, "checks": {"mfa_executable": False}},
    )
    def test_explicit_mfa_selection_starts_repair_after_approval(self, _diag, _find):
        dummy = _PromptDummy(approve=True)
        dummy._prompt_mfa_install_for_explicit_selection()

        self.assertEqual(dummy.prompts, [("japanese", "missing_runtime")])
        self.assertTrue(dummy.repair_started)


if __name__ == "__main__":
    unittest.main()
