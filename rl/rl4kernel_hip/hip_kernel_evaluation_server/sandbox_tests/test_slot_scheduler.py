# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from sandbox_core.protocol import EvalRequest
from sandbox_core.result import EvalRunResult
from server_adapters.master import GPUSlotManager, _result_to_response


def _request(name: str) -> EvalRequest:
    return EvalRequest(
        kernel_name=name,
        hip_code="__global__ void k() {}",
        hip_ref_code="__global__ void k() {}",
        pytorch_functional_code="def get_inputs():\n    return []\n",
    )


def _manager(tmpdir: str) -> GPUSlotManager:
    return GPUSlotManager(
        {
            "master": {"gpus": [0, 1, 2, 3]},
            "workers": [{"host": "10.1.1.1", "port": 9, "gpus": 4}],
            "settings": {
                "error_log_dir": tmpdir,
                "perf_iterations": 1,
                "enable_ref_compile_cache": False,
                "enable_ref_golden_cache": False,
                "enable_ref_perf_cache": False,
            },
        }
    )


class SlotSchedulerTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_queue_spreads_concurrent_singleton_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            manager.slot_policy = "queue"
            seen = []

            async def fake_execute(slot, task, batch_tmp_dir):
                del task, batch_tmp_dir
                seen.append((slot.node_type, slot.node_host, slot.gpu_id))
                await asyncio.sleep(0.05)
                return _result_to_response("k", EvalRunResult(True, True, True, 1.0, {}))

            manager.execute_on_slot = fake_execute

            async def go():
                await asyncio.gather(
                    *[
                        manager.dispatch_batch([_request(f"k{i}")], os.path.join(tmpdir, f"b{i}"))
                        for i in range(8)
                    ]
                )

            self._run(go())
            self.assertEqual(len(seen), 8)
            self.assertEqual(len(set(seen)), 8)

    def test_legacy_singleton_batches_share_slot_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            manager.slot_policy = "legacy"
            seen = []

            async def fake_execute(slot, task, batch_tmp_dir):
                del task, batch_tmp_dir
                seen.append(slot.gpu_id)
                return _result_to_response("k", EvalRunResult(True, True, True, 1.0, {}))

            manager.execute_on_slot = fake_execute

            async def go():
                await asyncio.gather(
                    *[manager.dispatch_batch([_request(f"k{i}")], tmpdir) for i in range(8)]
                )

            self._run(go())
            self.assertEqual(seen, [0] * 8)

    def test_queue_reuses_a_slot_after_release(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(os.path.join(tmpdir, "err"))
            os.makedirs(manager.error_log_dir, exist_ok=True)
            manager.slots = manager.slots[:2]
            manager.slot_policy = "queue"
            seen = []

            async def fake_execute(slot, task, batch_tmp_dir):
                del task, batch_tmp_dir
                seen.append(slot.gpu_id)
                await asyncio.sleep(0.01)
                return _result_to_response("k", EvalRunResult(True, True, True, 1.0, {}))

            manager.execute_on_slot = fake_execute

            async def go():
                await manager.dispatch_batch([_request(f"k{i}") for i in range(4)], tmpdir)

            self._run(go())
            self.assertEqual(sorted(seen), [0, 0, 1, 1])

    def test_failed_task_returns_its_slot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _manager(tmpdir)
            manager.slots = manager.slots[:1]
            manager.slot_policy = "queue"
            calls = []

            async def fake_execute(slot, task, batch_tmp_dir):
                del slot, batch_tmp_dir
                calls.append(task.kernel_name)
                if task.kernel_name == "boom":
                    raise RuntimeError("boom")
                return _result_to_response(task.kernel_name, EvalRunResult(True, True, True, 1.0, {}))

            manager.execute_on_slot = fake_execute

            async def go():
                first = await manager.dispatch_batch([_request("boom")], tmpdir)
                second = await manager.dispatch_batch([_request("ok")], tmpdir)
                return first, second

            first, second = self._run(go())
            self.assertEqual(calls, ["boom", "ok"])
            self.assertIn("boom", first[0].reason or "")
            self.assertTrue(second[0].compile_ok)


if __name__ == "__main__":
    unittest.main()
