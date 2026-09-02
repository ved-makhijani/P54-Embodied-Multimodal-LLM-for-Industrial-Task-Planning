import os

from simulation_backend.vision.scene_representation import get_current_scene


class _Entry:
    def __init__(self, label):
        self.label = label


class _Registry:
    def all_entries(self):
        return [_Entry("red block"), _Entry("left tray")]


class _Simulation:
    registry = _Registry()

    def __init__(self):
        self.called = False

    def get_live_scene(self, verbose=False):
        self.called = True
        return {
            "objects": [
                {"label": "red block", "position": (0.45, -0.2, 0.05)},
                {"label": "left tray", "position": [0.65, 0.45, 0.01]},
            ]
        }


def test_get_current_scene_uses_live_simulation_and_returns_json_scene(monkeypatch):
    monkeypatch.delenv("VISION_DETECTOR", raising=False)
    sim = _Simulation()

    scene = get_current_scene(sim=sim)

    assert sim.called is True
    assert scene == {
        "objects": [
            {"label": "red block", "position": [0.45, -0.2, 0.05]},
            {"label": "left tray", "position": [0.65, 0.45, 0.01]},
        ]
    }
    assert all(isinstance(obj["position"], list) for obj in scene["objects"])
    assert scene["objects"][0]["position"][2] == 0.05
    assert os.environ["VISION_DETECTOR"] == "yolo"
