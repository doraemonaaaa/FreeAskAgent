"""Contract tests for VLNAgent._candidates_from_external (CWP injection seam)."""
import numpy as np
import pytest

from agentflow.agents.vln_agent_4 import VLNAgent
from agentflow.agents.models_embodied_v2.skiils.protocol import SOM_MAX_CANDIDATES
from agentflow.agents.models_embodied_v2.memory.spatial_memory.candidates import Candidate


CAM = np.eye(4)


def build(external):
    # The method reads nothing from self; call it unbound.
    return VLNAgent._candidates_from_external(
        None, external, intrinsics=None, camera_to_world=CAM,
        image_shape=(480, 640))


def test_fields_and_turn_options():
    out = build([{"world_xyz": [1.0, 0.0, -2.0], "bearing_deg": 12.5,
                  "pixel_uv": [320, 400], "note": "cwp opening"}])
    openings = [c for c in out if c.kind != "turn"]
    turns = [c for c in out if c.kind == "turn"]
    assert len(openings) == 1 and len(turns) == 3
    c = openings[0]
    assert isinstance(c, Candidate)
    assert c.world_xyz == (1.0, 0.0, -2.0)
    assert c.pixel_uv == (320, 400)
    assert c.bearing_deg == 12.5
    assert c.note == "cwp opening"
    assert c.distance_m == pytest.approx(np.hypot(1.0, 2.0))
    assert [t.label for t in turns] == ["L", "R", "B"]
    assert [t.bearing_deg for t in turns] == [-90.0, 90.0, 180.0]


def test_caps_at_som_max_candidates():
    many = [{"world_xyz": [float(i), 0.0, -1.0]} for i in range(SOM_MAX_CANDIDATES + 4)]
    out = build(many)
    assert sum(1 for c in out if c.kind != "turn") == SOM_MAX_CANDIDATES


def test_defaults_without_uv_and_note():
    out = build([{"world_xyz": [0.0, 0.0, -3.0]}])
    c = out[0]
    assert c.pixel_uv is None or isinstance(c.pixel_uv, tuple)
    assert "3.0 m" in c.note
    assert c.label == ""  # relabel() downstream assigns numbers
