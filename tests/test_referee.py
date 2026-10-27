from neuronoid.emulator import Key, Mouse


class ScriptedPlayer:
    """Игрок по нашим знаниям (только для теста судьи): проходит меню и стоит."""

    def __init__(self):
        self.t = 0

    def act(self, frame):
        self.t += 1
        script = {150: [Key.RETURN], 220: [Key.M], 930: [Key.SPACE]}
        return script.get(self.t, []), Mouse()


def test_judge_in_subprocess(tmp_path):
    import json
    import subprocess
    import sys

    code = (
        "import json, sys; sys.path.insert(0, r'%s');"
        "from test_referee import ScriptedPlayer;"
        "from neuronoid.referee import judge;"
        "print(json.dumps(judge(ScriptedPlayer, games=1, seed=0, max_frames=12000)))"
    ) % str(__import__("pathlib").Path(__file__).parent)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    game = out["games"][0]
    assert game["started"] and game["game_over"] and game["round"] == 1 and game["score"] >= 0
    assert game["won"] is False
