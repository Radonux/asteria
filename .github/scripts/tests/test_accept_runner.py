"""The forced command's allocation: a runner name's -c<N> suffix."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "ci" / "dcs" / "accept-runner.sh"


class AcceptRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home)
        (self.home / "asteria-ci" / "jobs").mkdir(parents=True)
        (self.home / "asteria" / "ci" / "dcs").mkdir(parents=True)
        stubs = self.home / "stubs"
        stubs.mkdir()
        # sbatch records its arguments; git's pull is a no-op.
        (stubs / "sbatch").write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" > "$HOME/sbatch-args"\necho 4242\n'
        )
        (stubs / "git").write_text("#!/bin/bash\nexit 0\n")
        for stub in stubs.iterdir():
            stub.chmod(0o755)
        self.path = f"{stubs}:{os.environ['PATH']}"

    def accept(self, command: str) -> subprocess.CompletedProcess:
        env = {"HOME": str(self.home), "PATH": self.path, "SSH_ORIGINAL_COMMAND": command}
        return subprocess.run(
            ["bash", str(SCRIPT)], input="jitconfig", env=env, capture_output=True, text=True
        )

    def sbatch_args(self) -> list[str]:
        return (self.home / "sbatch-args").read_text().splitlines()

    def jobs(self) -> list[str]:
        return sorted(p.name for p in (self.home / "asteria-ci" / "jobs").iterdir())

    def test_no_suffix_keeps_the_scripts_allocation(self) -> None:
        result = self.accept("asteria/dcs-1-1-s000")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.sbatch_args()
        self.assertEqual(args[:2], ["--parsable", "--job-name=asteriadcs-1-1-s000"])
        self.assertEqual(args[2], str(self.home / "asteria" / "ci" / "dcs" / "runner-job.sbatch"))
        self.assertFalse(any(a.startswith(("--cpus-per-task", "--mem")) for a in args))

    def test_suffix_asks_for_cores_and_4_gb_each(self) -> None:
        for suffix, cores, mem in (("c64", "64", "256G"), ("c064", "64", "256G"), ("c3", "3", "12G"), ("c128", "128", "512G")):
            with self.subTest(suffix=suffix):
                result = self.accept(f"asteria/dcs-1-1-s000-{suffix}")
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.sbatch_args()
                self.assertEqual(args[2:4], [f"--cpus-per-task={cores}", f"--mem={mem}"])
                self.assertEqual(args[4], str(self.home / "asteria" / "ci" / "dcs" / "runner-job.sbatch"))

    def test_suffix_survives_the_names_cut(self) -> None:
        result = self.accept("asteria/dcs-" + "x" * 80 + "-c64")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.sbatch_args()
        self.assertEqual(len(args[1]), len("--job-name=") + 64)
        self.assertEqual(args[2], "--cpus-per-task=64")

    def test_out_of_range_is_refused_without_submitting(self) -> None:
        for suffix in ("c2", "c0", "c129", "c1000", "c0064"):
            with self.subTest(suffix=suffix):
                result = self.accept(f"asteria/dcs-1-1-s000-{suffix}")
                self.assertEqual(result.returncode, 1)
                self.assertIn("outside 3 to 128", result.stderr)
                self.assertFalse((self.home / "sbatch-args").exists())
                self.assertEqual(self.jobs(), [])


if __name__ == "__main__":
    unittest.main()
